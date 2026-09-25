"""One-call inference pipeline over windows that have arrived.

Chains the three steps that used to be three separate CLI invocations:

    predict (incremental) -> agent chain -> equipment/service-request sync

Both the replay worker and the operator console's manual trigger call this,
so there is exactly one code path for "process whatever has arrived", and a
tick with no new arrivals is a cheap no-op.

Note on inputs to the agent chain: this feeds each agent run the *model's*
predicted label and confidence, i.e. the real sensor -> model -> decision
flow. (`agents/run.py`'s standalone CLI instead passes the window's
ground-truth label; that path is left untouched so the documented batch
results stay reproducible, but it is not what a live pipeline should do.)
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from fleetmind.agents.graph import build_agent_graph
from fleetmind.agents.run import run_agent_pipeline
from fleetmind.agents.sync_equipment_status import sync_equipment_status
from fleetmind.db.models import (
    AgentDecision,
    Depot,
    Experiment,
    ModelPrediction,
    SignalWindow,
)
from fleetmind.ml.predict import predict_depot

MLRUNS = Path("mlruns")


def _predict_arrived(session, model_dir: Path) -> int:
    """Incrementally predict every depot that has a trained checkpoint."""
    depot_ids = sorted(session.execute(select(Depot.id)).scalars().all())
    total = 0
    for depot_id in depot_ids:
        checkpoint = model_dir / f"depot_{depot_id}" / "cnn_state_dict.pt"
        if not checkpoint.exists():
            continue
        total += predict_depot(depot_id, session, checkpoint, incremental=True)
    return total


def _run_agents_for_new_predictions(session, agent_app=None) -> int:
    """Run the agent chain over predictions that have no decision yet."""
    rows = session.execute(
        select(
            ModelPrediction.window_id,
            ModelPrediction.predicted_label,
            ModelPrediction.confidence,
            Experiment.depot_id,
            SignalWindow.raw_values,
        )
        .join(SignalWindow, SignalWindow.id == ModelPrediction.window_id)
        .join(Experiment, Experiment.id == SignalWindow.experiment_id)
        .outerjoin(AgentDecision, AgentDecision.prediction_id == ModelPrediction.id)
        .where(AgentDecision.id.is_(None))
        .order_by(ModelPrediction.window_id)
    ).all()
    if not rows:
        return 0

    app = agent_app if agent_app is not None else build_agent_graph()
    for row in rows:
        run_agent_pipeline(
            session=session,
            app=app,
            window_id=row.window_id,
            depot_id=row.depot_id,
            predicted_label=row.predicted_label,
            confidence=row.confidence,
            signal_values=[float(v) for v in (row.raw_values or [])],
        )
    return len(rows)


def process_arrived_windows(
    session, model_dir: Path | None = None, agent_app=None
) -> dict[str, int]:
    """Predict -> decide -> sync for everything that has arrived but is unprocessed.

    Commits once at the end so a tick is atomic: either the whole batch lands
    or none of it does. Returns per-stage counts for the API/UI to report.
    """
    model_dir = model_dir if model_dir is not None else MLRUNS

    predictions = _predict_arrived(session, model_dir)
    session.flush()
    decisions = _run_agents_for_new_predictions(session, agent_app=agent_app)
    session.flush()

    filed: list[tuple[str, str]] = []
    status_changes = sync_equipment_status(session, filed_requests_log=filed)

    session.commit()
    return {
        "predictions_written": predictions,
        "decisions_written": decisions,
        "equipment_status_changes": len(status_changes),
        "service_requests_filed": len(filed),
    }
