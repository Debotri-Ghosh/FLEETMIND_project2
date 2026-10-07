"""Pins the BatchNorm-transport diagnosis behind the federated collapse study.

The finding this file protects: `get_parameters()` / `set_parameters()` move
only `named_parameters()`, so the three BatchNorm layers' running statistics --
nine buffers -- never reach the global model. A global model assembled from
averaged weights therefore evaluated with `running_mean=0, running_var=1`,
i.e. with normalisation effectively switched off, while those weights had been
trained expecting normalised activations.

Aggregating those buffers recovers roughly ten accuracy points and partially
restores InnerRace recall (see the Progress log entry for the real numbers), so
the exact split between parameters and buffers is now load-bearing for a
documented result and must not drift silently.

Every test here asserts a positive count of what it inspected. A test that only
said "the buffers round-trip" would pass just as happily against a model with no
BatchNorm layers at all, which is the failure mode this project keeps finding
in its own suite.
"""

import torch

from fleetmind.federated.client import (
    get_buffers,
    get_parameters,
    recalibrate_bn,
    set_buffers,
)
from fleetmind.federated.run import average_buffers
from fleetmind.ml.model import LABEL_MAP, BearingCNN


def _toy_rows(n=32, label=0):
    import random

    rng = random.Random(0)
    return [([rng.gauss(0, 0.1) for _ in range(2048)], label % len(LABEL_MAP)) for _ in range(n)]


def test_parameters_and_buffers_are_disjoint_and_both_non_empty():
    """The diagnosis itself: BN running stats are buffers, not parameters.

    Asserts real counts rather than a mere inequality, so a model that lost its
    BatchNorm layers would fail here instead of passing vacuously.
    """
    model = BearingCNN(num_classes=len(LABEL_MAP))

    param_names = [name for name, _ in model.named_parameters()]
    buffer_names = [name for name, _ in model.named_buffers()]

    # 3 conv layers + 3 BatchNorm layers + 1 linear, weight and bias each.
    assert len(param_names) == 11, param_names
    # 3 BatchNorm layers x (running_mean, running_var, num_batches_tracked).
    assert len(buffer_names) == 9, buffer_names

    assert set(param_names).isdisjoint(buffer_names)

    # And the buffers really are the BatchNorm statistics, by name.
    for stat in ("running_mean", "running_var", "num_batches_tracked"):
        matching = [n for n in buffer_names if n.endswith(stat)]
        assert len(matching) == 3, (stat, buffer_names)

    # The payload the federated code actually transports carries none of them.
    assert len(get_parameters(model)) == len(param_names)
    assert len(get_buffers(model)) == len(buffer_names)


def test_a_fresh_global_model_has_no_normalisation_statistics():
    """Why the omission mattered: the default buffers are an identity transform.

    This is the state the published federated run evaluated in.
    """
    model = BearingCNN(num_classes=len(LABEL_MAP))
    checked = 0
    for name, buf in model.named_buffers():
        if name.endswith("running_mean"):
            assert torch.allclose(buf, torch.zeros_like(buf))
            checked += 1
        elif name.endswith("running_var"):
            assert torch.allclose(buf, torch.ones_like(buf))
            checked += 1
    assert checked == 6, "expected 3 running_mean + 3 running_var buffers"


def test_get_set_buffers_roundtrip_preserves_values_and_dtypes():
    source = BearingCNN(num_classes=len(LABEL_MAP))
    # Move the statistics away from their defaults so the round-trip is a real
    # test and not a comparison of two identical fresh models.
    seen = recalibrate_bn(source, _toy_rows(64, 0), batch_size=16)
    assert seen == 4, seen

    target = BearingCNN(num_classes=len(LABEL_MAP))
    set_buffers(target, get_buffers(source))

    compared = 0
    for (name_a, a), (name_b, b) in zip(
        source.named_buffers(), target.named_buffers(), strict=True
    ):
        assert name_a == name_b
        assert a.dtype == b.dtype, name_a
        assert torch.allclose(a.float(), b.float()), name_a
        compared += 1
    assert compared == 9, compared


def test_recalibrate_bn_moves_the_statistics_and_reports_batches_seen():
    model = BearingCNN(num_classes=len(LABEL_MAP))
    before = [b.clone() for _, b in model.named_buffers()]

    seen = recalibrate_bn(model, _toy_rows(96, 1), batch_size=32)
    assert seen == 3, "96 rows at batch_size=32 is 3 batches"

    after = list(model.named_buffers())
    moved = sum(
        1
        for (name, b), b0 in zip(after, before, strict=True)
        if not torch.allclose(b.float(), b0.float())
    )
    # All 9 should move: both statistics plus the batch counter on each layer.
    assert moved == 9, [n for (n, _), _ in zip(after, before, strict=True)]

    # It must leave the model ready to evaluate, not stuck in train mode.
    assert not model.training

    # And it must not have trained anything.
    assert all(p.grad is None for p in model.parameters())


def test_recalibrate_bn_respects_max_batches():
    model = BearingCNN(num_classes=len(LABEL_MAP))
    seen = recalibrate_bn(model, _toy_rows(256, 0), batch_size=16, max_batches=2)
    assert seen == 2, seen


def test_average_buffers_is_sample_weighted_and_covers_every_buffer():
    model = BearingCNN(num_classes=len(LABEL_MAP))
    shapes = [tuple(b.shape) for _, b in model.named_buffers()]

    zeros = [torch.zeros(s).tolist() for s in shapes]
    ones = [torch.ones(s).tolist() for s in shapes]

    averaged = average_buffers([zeros, ones], [3, 1])
    assert len(averaged) == len(shapes) == 9

    flat = []
    for layer in averaged:
        t = torch.as_tensor(layer)
        flat.extend(t.flatten().tolist())
    assert len(flat) > 0
    assert all(abs(v - 0.25) < 1e-9 for v in flat), flat[:5]


def test_averaged_buffers_load_back_into_a_model():
    """The whole point: the averaged result must be loadable by set_buffers."""
    a = BearingCNN(num_classes=len(LABEL_MAP))
    b = BearingCNN(num_classes=len(LABEL_MAP))
    assert recalibrate_bn(a, _toy_rows(64, 0), batch_size=32) == 2
    assert recalibrate_bn(b, _toy_rows(64, 2), batch_size=32) == 2

    merged = average_buffers([get_buffers(a), get_buffers(b)], [1, 1])

    target = BearingCNN(num_classes=len(LABEL_MAP))
    set_buffers(target, merged)

    # The merged running_mean must sit between the two sources, not equal either.
    checked = 0
    for (name, t), (_, ba), (_, bb) in zip(
        target.named_buffers(), a.named_buffers(), b.named_buffers(), strict=True
    ):
        if name.endswith("running_mean"):
            assert torch.allclose(t, (ba + bb) / 2, atol=1e-6), name
            checked += 1
    assert checked == 3, checked
