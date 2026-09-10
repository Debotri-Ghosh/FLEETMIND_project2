"""Train a local 1D-CNN fault classifier for ONE depot.

Usage:
    python -m fleetmind.ml.train --depot-id 1 --epochs 20 --batch-size 64

Reads `signal_windows` joined to experiments->depots from Postgres,
splits stratified 80/20 per depot, trains BearingCNN, and writes metrics +
artifacts to mlruns/<depot_id>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.core.config import get_settings
from fleetmind.db.models import Depot, Experiment, SignalWindow
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.ml.model import LABEL_MAP, BearingCNN

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


def load_depot_rows(session: Session, depot_id: int):
    """Return [(values_list, label_index), ...] for all windows of a depot."""
    rows = session.execute(
        select(SignalWindow.raw_values, SignalWindow.label)
        .join(Experiment, Experiment.id == SignalWindow.experiment_id)
        .where(Experiment.depot_id == depot_id)
        .order_by(SignalWindow.id)
    ).all()
    out = []
    for raw_values, label in rows:
        values = [float(v) for v in (raw_values or [])]
        if label in LABEL_MAP:
            out.append((values, LABEL_MAP.index(label)))
    return out


def stratified_split(rows, frac=0.8, seed=42):
    """Naive 80/20 split stratified by label index (keeps per-class ratio)."""
    gen = torch.Generator().manual_seed(seed)
    by_class: dict[int, list] = {}
    for i, (_, label_index) in enumerate(rows):
        by_class.setdefault(label_index, []).append(i)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        perm = torch.tensor(idxs, dtype=torch.long)
        perm = perm[torch.randperm(len(perm), generator=gen)]
        cut = int(len(perm) * frac)
        train_idx += perm[:cut].tolist()
        val_idx += perm[cut:].tolist()
    train = [rows[i] for i in train_idx]
    val = [rows[i] for i in val_idx]
    return train, val


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

    rows = load_depot_rows(session, args.depot_id)
    depot_name = {d.id: d.name for d in session.query(Depot.id, Depot.name).all()}
    session.close()

    if not rows:
        print(f"No signal_windows found for depot_id={args.depot_id}.")
        print("Run ingestion first: python -m fleetmind.ingestion.ingest")
        return

    print(f"Depot {args.depot_id} ({depot_name.get(args.depot_id, '?')}): {len(rows)} windows")

    train_rows, val_rows = stratified_split(rows, frac=0.8, seed=args.seed)
    print(f"  train={len(train_rows)}  val={len(val_rows)}")

    train_loader = DataLoader(_TorchDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(_TorchDataset(val_rows), batch_size=args.batch_size)

    model = BearingCNN(num_classes=len(LABEL_MAP))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

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
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model + metrics to {out_dir.resolve()}")
    print(f"Best val accuracy: {best_acc:.3%}")


if __name__ == "__main__":
    main()
