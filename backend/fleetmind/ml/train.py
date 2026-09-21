"""Train a local 1D-CNN fault classifier for ONE depot.

Usage:
    python -m fleetmind.ml.train --depot-id 1 --epochs 20 --batch-size 64

Reads `signal_windows` joined to experiments->depots from Postgres,
splits by EXPERIMENT (whole .mat recordings held out -- review-honest, see
augmentation/exp_split.py), trains BearingCNN with class-weighted loss, and
writes metrics + artifacts to mlruns/<depot_id>/.

FIX (2026-09-19): this file previously split individual WINDOWS 80/20
(`stratified_split`, now removed), so windows from the same recording could
land on both sides of the split -- the model memorized recordings instead of
learning to generalize, inflating val/test accuracy to 100%. It now reuses
the same experiment-level split as augmentation/run.py and federated/run.py,
and uses class-weighted CrossEntropyLoss (see ml/model.py:class_weights) to
address the real class imbalance that unweighted loss was masking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sqlalchemy.orm import Session

from fleetmind.augmentation.exp_split import stratified_experiment_split
from fleetmind.augmentation.run import load_experiment_rows
from fleetmind.core.config import get_settings
from fleetmind.db.models import Depot
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.ml.model import LABEL_MAP, BearingCNN, class_weights

try:
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "torch is required for training: install 'fleetmind[ml]' "
        "(pip install 'fleetmind[ml]' in your venv, or run inside the API container)."
    ) from exc

MLRUNS = Path("mlruns")


class _TorchDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        values, label_index = self.rows[idx]
        x = torch.tensor(values, dtype=torch.float32).view(1, -1)
        return x, label_index


def load_depot_split(session: Session, depot_id: int, frac: float = 0.8, seed: int = 42):
    """Experiment-level train/val split for a depot.

    Returns (train_rows, val_rows) as [(values, label_index), ...], using the
    same stratified_experiment_split as augmentation/run.py so all training
    paths in the repo agree on what "honest evaluation" means.
    """
    all_rows = load_experiment_rows(session, depot_id)  # (experiment_id, values, label_index)
    exp_fault = {e: LABEL_MAP[lab] for e, _, lab in all_rows}
    train_exp, val_exp = stratified_experiment_split(
        sorted(exp_fault.items()), frac=frac, seed=seed
    )
    se_train, se_val = set(train_exp), set(val_exp)
    from fleetmind.ml.model import to_fft_magnitude

    train_rows = [(to_fft_magnitude(v), lab) for e, v, lab in all_rows if e in se_train]
    val_rows = [(to_fft_magnitude(v), lab) for e, v, lab in all_rows if e in se_val]
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local 1D-CNN for one depot.")
    parser.add_argument("--depot-id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = make_session_factory(engine)
    session = factory()

    depot_name = {d.id: d.name for d in session.query(Depot.id, Depot.name).all()}
    train_rows, val_rows = load_depot_split(session, args.depot_id, frac=0.8, seed=args.seed)
    session.close()

    if not train_rows:
        print(f"No signal_windows found for depot_id={args.depot_id}.")
        print("Run ingestion first: python -m fleetmind.ingestion.ingest")
        return

    print(f"Depot {args.depot_id} ({depot_name.get(args.depot_id, '?')})")
    print(f"  train={len(train_rows)}  val={len(val_rows)}  (experiment-level split)")

    train_loader = DataLoader(_TorchDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(_TorchDataset(val_rows), batch_size=args.batch_size)

    model = BearingCNN(num_classes=len(LABEL_MAP))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_rows, len(LABEL_MAP)))

    out_dir = MLRUNS / f"depot_{args.depot_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += yb.size(0)
        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)
        print(f"  epoch {epoch:2d}/{args.epochs}  loss={running / n:.4f}  val_acc={acc:.3f}")

    torch.save(model.state_dict(), out_dir / "cnn_state_dict.pt")
    metrics = {
        "depot_id": args.depot_id,
        "best_val_acc": best_acc,
        "epochs": args.epochs,
        "train_windows": len(train_rows),
        "val_windows": len(val_rows),
        "split": "experiment-level (fixed 2026-09-19)",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model + metrics to {out_dir.resolve()}")
    print(f"Best val accuracy: {best_acc:.3%}")


if __name__ == "__main__":
    main()
