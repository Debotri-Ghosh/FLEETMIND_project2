"""Retrain per site with real best-epoch selection, to `mlruns/depot_N_v2`.

    python -m fleetmind.decision.retrain                 # all four sites
    python -m fleetmind.decision.retrain --depot-id 3
    python -m fleetmind.decision.retrain --epochs 20 --seed 42

What this fixes
---------------
`ml/train.py` tracks `best_acc = max(best_acc, acc)` across epochs and then calls
`torch.save()` **after** the loop, with no best-epoch guard -- so the weights on
disk are the last epoch's while `metrics.json` reports the best epoch's score.
The published figures therefore describe models that were never written to disk.

This module keeps an actual snapshot of the best epoch's weights, chosen on a
**selection set that is neither trained on nor tested on**. Selecting an epoch on
the test set would make the test accuracy an optimistic estimate of itself, which
is the subtler version of the same mistake.

Two checkpoints come out of one training run
--------------------------------------------
* `cnn_state_dict.pt`      -- the best-on-selection epoch (the v2 result)
* `cnn_state_dict_last.pt` -- the final epoch (a control)

The control exists because v2 trains on roughly a third less data than v1 did
(the selection set was carved out of v1's training recordings). Comparing
v2-last against v1 isolates that data reduction; comparing v2-best against
v2-last isolates best-epoch selection. Without both, a result could not be
attributed.

**The test set is never opened here.** `compare.py` is the only module that reads
it, and `tests/test_decision_phase1.py` asserts that this module's training and
selection recordings are disjoint from it.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

from fleetmind.db.models import Depot
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.decision import split3
from fleetmind.ml.model import LABEL_MAP, BearingCNN, class_weights, to_fft_magnitude
from fleetmind.ml.train import _TorchDataset

MLRUNS = Path("mlruns")
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3


def out_dir_for(depot_id: int, mlruns: Path = MLRUNS) -> Path:
    """`mlruns/depot_N_v2` -- deliberately NOT `mlruns/depot_N`.

    The v1 checkpoints produced all 4,367 stored predictions and decisions, so
    overwriting them would silently invalidate every figure derived from them.
    """
    return mlruns / f"depot_{depot_id}_v2"


def _rows(windows: list[tuple[int, int, int, list[float], str]]) -> list[tuple[list[float], int]]:
    """(fft_magnitude, label_index) -- the protected input transform, always."""
    return [
        (to_fft_magnitude(values), LABEL_MAP.index(label)) for _, _, _, values, label in windows
    ]


def _accuracy(model: BearingCNN, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += int((model(xb).argmax(1) == yb).sum().item())
            total += int(yb.size(0))
    return correct / max(total, 1)


def retrain_depot(
    session: Session,
    depot_id: int,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = 42,
    mlruns: Path = MLRUNS,
    write: bool = True,
) -> dict:
    """Train one site, select the best epoch on the selection set, save both arms."""
    split = split3.three_way_split(session, depot_id)

    # Guard, not a comment: the test recordings must never enter this function's
    # data. A bug that leaked them would otherwise be invisible in the output.
    test_ids = set(split.test_experiment_ids)
    if test_ids & set(split.train_experiment_ids + split.selection_experiment_ids):
        raise AssertionError("test recordings leaked into training or selection")

    train_rows = _rows(split3.windows_for(session, split.train_experiment_ids))
    selection_rows = _rows(split3.windows_for(session, split.selection_experiment_ids))
    if not train_rows or not selection_rows:
        raise ValueError(f"site {depot_id}: empty train or selection set")

    torch.manual_seed(seed)
    model = BearingCNN(num_classes=len(LABEL_MAP))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_rows, len(LABEL_MAP)))

    train_loader = DataLoader(_TorchDataset(train_rows), batch_size=batch_size, shuffle=True)
    selection_loader = DataLoader(_TorchDataset(selection_rows), batch_size=batch_size)

    history: list[dict] = []
    best_acc = -1.0
    best_epoch = -1
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(xb.size(0))
            seen += int(xb.size(0))

        acc = _accuracy(model, selection_loader)
        history.append(
            {"epoch": epoch, "train_loss": running / max(seen, 1), "selection_accuracy": acc}
        )
        # Strict `>` so the EARLIEST best epoch wins a tie -- the less-overfit one.
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"  epoch {epoch:2d}/{epochs}  loss={running / max(seen, 1):.4f}  "
            f"selection_acc={acc:.4f}" + ("  <- best so far" if epoch == best_epoch else "")
        )

    if best_state is None:
        raise RuntimeError("no epoch was ever recorded as best; training did not run")

    last_state = copy.deepcopy(model.state_dict())
    last_acc = history[-1]["selection_accuracy"]

    report = {
        "depot_id": depot_id,
        "epochs": epochs,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "best_epoch": best_epoch,
        "best_selection_accuracy": best_acc,
        "last_epoch": epochs,
        "last_selection_accuracy": last_acc,
        "best_minus_last_on_selection": best_acc - last_acc,
        "selection_accuracy_history": [h["selection_accuracy"] for h in history],
        "history": history,
        "split": split.as_dict(),
        "note": (
            "best_epoch was chosen on the SELECTION set only; the test set was not "
            "opened by this module. cnn_state_dict.pt is the best epoch, "
            "cnn_state_dict_last.pt is the final epoch (control for the smaller "
            "training set)."
        ),
    }

    if write:
        out = out_dir_for(depot_id, mlruns)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out / "cnn_state_dict.pt")
        torch.save(last_state, out / "cnn_state_dict_last.pt")
        (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["out_dir"] = str(out)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depot-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlruns", default="mlruns")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    session = make_session_factory(create_db_engine())()
    try:
        depot_ids = (
            [args.depot_id]
            if args.depot_id is not None
            else [int(r[0]) for r in session.execute(select(Depot.id).order_by(Depot.id)).all()]
        )
        for depot_id in depot_ids:
            name = session.execute(select(Depot.name).where(Depot.id == depot_id)).scalar()
            print(f"\n=== site {depot_id} -- {name} ===")
            r = retrain_depot(
                session,
                depot_id,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                mlruns=Path(args.mlruns),
                write=not args.no_write,
            )
            s = r["split"]
            print(
                f"  recordings {s['recordings']}  windows {s['windows']}\n"
                f"  best epoch {r['best_epoch']} at selection_acc "
                f"{r['best_selection_accuracy']:.4f}; last epoch {r['last_epoch']} at "
                f"{r['last_selection_accuracy']:.4f} "
                f"(best - last = {r['best_minus_last_on_selection']:+.4f})"
            )
            if "out_dir" in r:
                print(f"  wrote {r['out_dir']}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
