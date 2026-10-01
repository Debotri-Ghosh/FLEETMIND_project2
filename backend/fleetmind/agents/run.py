"""Run the agent pipeline and persist decisions to Postgres.

Usage:
    python -m fleetmind.agents.run --window-id 42
    python -m fleetmind.agents.run --all-pending
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import insert, select

from fleetmind.agents.graph import build_agent_graph
from fleetmind.agents.state import AgentState
from fleetmind.core.config import get_settings
from fleetmind.db.models import (
    AgentDecision,
    Depot,
    Experiment,
    ModelPrediction,
    SignalWindow,
)
from fleetmind.db.session import create_db_engine, make_session_factory

MLRUNS = Path("mlruns/agents")


def _prediction_for_window(session, window_id: int) -> dict | None:
    """Get (or create) a model prediction for the window so we have a FK target."""
    row = session.execute(
        select(ModelPrediction).where(ModelPrediction.window_id == window_id)
    ).scalar_one_or_none()
    if row is not None:
        return {"id": row.id, "predicted_label": row.predicted_label, "confidence": row.confidence}

    # No prediction yet — create a placeholder from the window's label so the
    # agent pipeline still has something to chew on.
    win = session.execute(
        select(SignalWindow.id, SignalWindow.raw_values, SignalWindow.label).where(
            SignalWindow.id == window_id
        )
    ).one_or_none()
    if win is None:
        return None
    result = session.execute(
        insert(ModelPrediction).values(
            window_id=window_id,
            model_version="v0-placeholder",
            predicted_label=win.label,
            confidence=None,
        )
    )
    session.flush()
    pred_id = result.inserted_primary_key[0]
    return {"id": pred_id, "predicted_label": win.label, "confidence": None}


def _depot_id_for_window(session, window_id: int) -> int:
    from fleetmind.db.models import Experiment

    return session.execute(
        select(Experiment.depot_id)
        .join(SignalWindow, SignalWindow.experiment_id == Experiment.id)
        .where(SignalWindow.id == window_id)
    ).scalar_one()


def run_agent_pipeline(
    session,
    app,
    window_id: int,
    depot_id: int,
    predicted_label: str,
    confidence: float | None,
    signal_values: list[float],
    load_hp: int | None = None,
) -> AgentState:
    """Run the full agent pipeline for one window and persist the decision.

    `load_hp` is the site's motor load (depots.load_hp) and is what the risk
    agent's load factor is keyed by. It defaults to None (neutral factor)
    rather than being derived from depot_id -- see LOAD_RISK_FACTOR.
    """
    state: AgentState = {
        "window_id": window_id,
        "depot_id": depot_id,
        "load_hp": load_hp,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "signal_values": signal_values,
    }
    result = app.invoke(state)

    # Persist to agent_decisions
    prediction_row = _prediction_for_window(session, window_id)
    if prediction_row is not None:
        session.execute(
            insert(AgentDecision).values(
                prediction_id=prediction_row["id"],
                agent_name="fleetmind-multi-agent",
                decision_text=result.get("decision_text", ""),
                risk_score=result.get("risk_score"),
            )
        )
        session.flush()
        result["agent_decision_id"] = None  # flushed but not fetched; fine for CLI

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FleetMind agent pipeline.")
    parser.add_argument("--window-id", type=int, default=None)
    parser.add_argument("--all-pending", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = make_session_factory(engine)
    session = factory()

    if args.window_id:
        rows = [(args.window_id,)]
    elif args.all_pending:
        # windows that have no agent decision yet
        rows = [
            (row[0],)
            for row in session.execute(
                select(SignalWindow.id)
                .outerjoin(ModelPrediction, ModelPrediction.window_id == SignalWindow.id)
                .outerjoin(AgentDecision, AgentDecision.prediction_id == ModelPrediction.id)
                .where(AgentDecision.id.is_(None))
                .order_by(SignalWindow.id)
            ).all()
        ]
    else:
        raise SystemExit("Pass --window-id N or --all-pending")

    if not rows:
        print("No windows to process.")
        return

    MLRUNS.mkdir(parents=True, exist_ok=True)
    batch_size = 50
    app = build_agent_graph()  # build ONCE, reuse for all windows
    for i, (window_id,) in enumerate(rows, 1):
        win = session.execute(
            select(
                SignalWindow.raw_values,
                SignalWindow.label,
                Experiment.depot_id,
                Depot.load_hp,
            )
            .join(Experiment, Experiment.id == SignalWindow.experiment_id)
            .join(Depot, Depot.id == Experiment.depot_id)
            .where(SignalWindow.id == window_id)
        ).one_or_none()
        if win is None:
            continue

        run_agent_pipeline(
            session=session,
            app=app,
            window_id=window_id,
            depot_id=win.depot_id,
            predicted_label=win.label,
            confidence=None,
            signal_values=list(win.raw_values or []),
            load_hp=win.load_hp,
        )
        if i % batch_size == 0:
            session.commit()
            print(f"  committed {i}/{len(rows)} ...")

    session.commit()
    session.close()
    print(f"Done. {len(rows)} decisions written to agent_decisions.")


if __name__ == "__main__":
    main()
