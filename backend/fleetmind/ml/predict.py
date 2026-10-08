"""Run trained per-depot CNN models on all windows, write ModelPrediction rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sqlalchemy import delete, select

from fleetmind.core.config import get_settings
from fleetmind.db.models import (
    AgentDecision,
    Depot,
    Experiment,
    ModelPrediction,
    SignalWindow,
)
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.ml.model import LABEL_MAP, BearingCNN, to_fft_magnitude

MLRUNS = Path("mlruns")


def predict_depot(depot_id: int, session, model_path: Path, incremental: bool = False) -> int:
    """Run this depot's trained CNN over its windows and write ModelPrediction rows.

    Two modes, deliberately different:

    * ``incremental=False`` (default, the documented offline batch workflow):
      wipe every existing prediction/decision for this depot and reprocess all
      of its windows, ignoring ``released_at`` entirely. Reproducible full runs.
    * ``incremental=True`` (used by the replay pipeline): process only windows
      that have *arrived* (``released_at IS NOT NULL``) and do not already have
      a prediction. Destroys nothing, so it is safe to call repeatedly on a
      tick -- that is what makes a live, incremental workflow possible.
    """
    model = BearingCNN(num_classes=len(LABEL_MAP))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    if not incremental:
        depot_window_ids = (
            session.execute(
                select(SignalWindow.id)
                .join(Experiment, Experiment.id == SignalWindow.experiment_id)
                .where(Experiment.depot_id == depot_id)
            )
            .scalars()
            .all()
        )
        if depot_window_ids:
            session.execute(
                delete(AgentDecision).where(
                    AgentDecision.prediction_id.in_(
                        select(ModelPrediction.id).where(
                            ModelPrediction.window_id.in_(depot_window_ids)
                        )
                    )
                )
            )
            session.execute(
                delete(ModelPrediction).where(ModelPrediction.window_id.in_(depot_window_ids))
            )

    query = (
        select(SignalWindow.id, SignalWindow.raw_values, SignalWindow.label)
        .join(Experiment, Experiment.id == SignalWindow.experiment_id)
        .where(Experiment.depot_id == depot_id)
        .order_by(SignalWindow.id)
    )
    if incremental:
        query = query.outerjoin(
            ModelPrediction, ModelPrediction.window_id == SignalWindow.id
        ).where(SignalWindow.released_at.is_not(None), ModelPrediction.id.is_(None))
    windows = session.execute(query).all()

    count = 0
    for window_id, raw_values, _true_label in windows:
        values = [float(v) for v in (raw_values or [])]
        if len(values) < 2048:
            values += [0.0] * (2048 - len(values))
        else:
            values = values[:2048]
        values = to_fft_magnitude(values)
        x = torch.tensor(values, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, 1).squeeze(0)
            pred_idx = int(probs.argmax())
        pred_label = LABEL_MAP[pred_idx]
        confidence = float(probs[pred_idx])
        pred = ModelPrediction(
            window_id=window_id,
            model_version=f"cnn_depot_{depot_id}",
            predicted_label=pred_label,
            confidence=confidence,
            class_probabilities=[float(v) for v in probs],
        )
        session.add(pred)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depot-id", type=int, default=None)
    parser.add_argument("--all-depots", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = make_session_factory(engine)
    session = factory()

    depots = {d.id: d.name for d in session.execute(select(Depot.id, Depot.name)).all()}
    if args.all_depots:
        depot_ids = sorted(depots.keys())
    elif args.depot_id:
        depot_ids = [args.depot_id]
    else:
        raise SystemExit("Pass --depot-id N or --all-depots")

    total = 0
    for did in depot_ids:
        ckpt = MLRUNS / f"depot_{did}" / "cnn_state_dict.pt"
        if not ckpt.exists():
            print(f"  [skip] no checkpoint for depot {did} ({depots.get(did, '?')}) at {ckpt}")
            continue
        n = predict_depot(did, session, ckpt)
        total += n
        print(f"  [ok] depot {did} ({depots.get(did, '?')}): {n} predictions")
    session.commit()
    session.close()
    print(f"\nWrote {total} model predictions.")


if __name__ == "__main__":
    main()
