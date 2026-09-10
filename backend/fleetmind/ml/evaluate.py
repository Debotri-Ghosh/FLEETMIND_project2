"""Evaluate a depot's trained 1D-CNN on its held-out test windows.

Usage:
    python -m fleetmind.ml.evaluate --depot-id 1
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fleetmind.core.config import get_settings
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.ml.model import LABEL_MAP, BearingCNN
from fleetmind.ml.train import _TorchDataset, load_depot_rows, stratified_split


def evaluate(depot_id: int, checkpoint_dir: Path) -> dict:
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    factory = make_session_factory(engine)
    session = factory()

    rows = load_depot_rows(session, depot_id)
    session.close()
    if not rows:
        raise SystemExit(f"No windows for depot_id={depot_id}")

    _, val_rows = stratified_split(rows, frac=0.8, seed=42)
    val_loader = DataLoader(_TorchDataset(val_rows), batch_size=64)

    model = BearingCNN(num_classes=len(LABEL_MAP))
    state = torch.load(checkpoint_dir / "cnn_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss_sum += loss_fn(logits, yb).item() * xb.size(0)
            preds = logits.argmax(1)
            all_preds += preds.tolist()
            all_labels += yb.tolist()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    results = {
        "depot_id": depot_id,
        "test_windows": total,
        "accuracy": correct / total,
        "loss": loss_sum / total,
        "predictions": all_preds,
        "labels": all_labels,
    }
    (checkpoint_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--depot-id", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()
    ckpt = args.checkpoint_dir or Path("mlruns") / f"depot_{args.depot_id}"
    evaluate(args.depot_id, ckpt)
