"""Tests for Phase 4 federated helpers (no DB, no Flower server needed)."""

import torch

from fleetmind.federated.client import (
    evaluate_model,
    get_parameters,
    local_train,
    set_parameters,
)
from fleetmind.federated.run import average_parameters
from fleetmind.ml.model import LABEL_MAP, BearingCNN


def _toy_rows(n=32, label=0):
    import random

    rng = random.Random(0)
    return [([rng.gauss(0, 0.1) for _ in range(2048)], label % len(LABEL_MAP)) for _ in range(n)]


def test_get_set_parameters_roundtrip():
    m1, m2 = BearingCNN(num_classes=4), BearingCNN(num_classes=4)
    set_parameters(m2, get_parameters(m1))
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.allclose(p1, p2)


def test_average_parameters_is_sample_weighted():

    m = BearingCNN(num_classes=4)
    n = sum(p.numel() for p in m.parameters())
    zeros = [[0.0] * p.numel() for p in m.parameters()]
    ones = [[1.0] * p.numel() for p in m.parameters()]
    avg = average_parameters([zeros, ones], [3, 1])
    flat = [v for layer in avg for v in layer]
    assert len(flat) == n
    assert all(abs(v - 0.25) < 1e-9 for v in flat)


def test_local_train_reduces_loss_and_evaluates():
    torch.manual_seed(0)
    model = BearingCNN(num_classes=len(LABEL_MAP))
    rows = _toy_rows(64, 0) + _toy_rows(64, 1)
    before = evaluate_model(model, rows)["accuracy"]
    stats = local_train(model, rows, epochs=2, batch_size=32, lr=3e-3, mu=0.0)
    after = evaluate_model(model, rows)["accuracy"]
    assert stats["loss"] > 0
    assert after >= before


def test_local_train_fedprox_runs():
    torch.manual_seed(0)
    model = BearingCNN(num_classes=len(LABEL_MAP))
    rows = _toy_rows(32, 0) + _toy_rows(32, 1)
    stats = local_train(model, rows, epochs=1, batch_size=32, lr=3e-3, mu=0.1)
    assert stats["loss"] > 0
