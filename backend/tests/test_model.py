"""Tests for the ML dataset wrapper and the 1D-CNN model."""

import torch

from fleetmind.ml.model import LABEL_MAP, BearingCNN


def test_label_map_has_four_classes():
    assert LABEL_MAP == ["Normal", "InnerRace", "OuterRace", "Ball"]


def test_cnn_forward_shape():
    model = BearingCNN(num_classes=4)
    model.eval()
    x = torch.randn(8, 1, 2048)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (8, 4)


def test_cnn_param_count_is_reasonable():
    n = sum(p.numel() for p in BearingCNN().parameters())
    assert n < 1_000_000


def test_cnn_handles_short_input():
    # model should not crash on a (B,1,2048) batch even if pooling compresses it
    model = BearingCNN(num_classes=4)
    model.eval()
    x = torch.randn(2, 1, 2048)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()
