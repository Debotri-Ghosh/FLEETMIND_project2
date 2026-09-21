"""Client-side training helpers shared by the Flower NumPyClient wrappers.

Phase 4 design (M.Tech-demo scale, CPU):
- Feature spec is FIXED: 2048 raw samples -> the same BearingCNN used in
  Phase 2 (`fleetmind.ml.model.BearingCNN`).
- Each depot client owns its windows (experiment-split train partition) and
  runs local SGD for `local_epochs` per round.
- FedAvg: vanilla local SGD (mu=0). FedProx: local objective gains the
  proximal term (mu/2)||w - w_global||^2 (Li et al., MLSys 2020).
- Parameter transport uses plain float lists (Flower NumPyClient style) so
  the code works with flwr>=1.9 without pulling in the whole server stack
  for unit tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fleetmind.ml.model import LABEL_MAP, BearingCNN, class_weights


class _RowsDataset(Dataset):
    def __init__(self, rows: list[tuple[list[float], int]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        values, label = self.rows[idx]
        x = torch.tensor(values, dtype=torch.float32).view(1, -1)
        return x, label


def _state_params(model: BearingCNN) -> list:
    return [p for _, p in model.named_parameters()]


def get_parameters(model: BearingCNN) -> list:
    """Learnable parameters only (excludes BatchNorm running stats)."""
    return [p.detach().cpu().numpy().tolist() for p in _state_params(model)]


def set_parameters(model: BearingCNN, parameters: list) -> None:
    """Load learnable parameters only; BatchNorm buffers stay client-local."""
    import numpy as np

    with torch.no_grad():
        for p, arr in zip(_state_params(model), parameters, strict=True):
            p.copy_(torch.as_tensor(np.asarray(arr, dtype="float32")))


def local_train(
    model: BearingCNN,
    train_rows: list[tuple[list[float], int]],
    *,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 3e-3,
    mu: float = 0.0,
    seed: int = 42,
) -> dict[str, float]:
    """Run local SGD; mu>0 adds the FedProx proximal term. Returns {'loss': ...}."""
    torch.manual_seed(seed)
    loader = DataLoader(_RowsDataset(train_rows), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # FIX (2026-09-19): unweighted loss let federated averaging collapse to
    # always predicting the majority class (OuterRace) -- see ml/model.py:
    # class_weights and federated/run.py's gain_vs_baseline going negative.
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_rows, len(LABEL_MAP)))
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0.0 else None
    total, n = 0.0, 0
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if mu > 0.0 and global_params is not None:
                prox = sum(
                    ((p - gp) ** 2).sum()
                    for p, gp in zip(model.parameters(), global_params, strict=True)
                )
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return {"loss": total / max(n, 1)}


def evaluate_model(
    model: BearingCNN,
    test_rows: list[tuple[list[float], int]],
    *,
    batch_size: int = 64,
) -> dict:
    """Accuracy + per-class recall on held-out rows (no gradient)."""
    from collections import Counter

    loader = DataLoader(_RowsDataset(test_rows), batch_size=batch_size)
    model.eval()
    correct, total = 0, 0
    cc: Counter[int] = Counter()
    tc: Counter[int] = Counter()
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            for p, t in zip(preds.tolist(), yb.tolist(), strict=True):
                tc[t] += 1
                if p == t:
                    cc[t] += 1
    recall = {
        LABEL_MAP[c]: (cc.get(c, 0) / tc[c] if tc.get(c, 0) else 0.0) for c in range(len(LABEL_MAP))
    }
    return {"accuracy": correct / max(total, 1), "per_class_recall": recall}
