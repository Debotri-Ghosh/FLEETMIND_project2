"""Tests for Phase 3 augmentation: GAN shapes, sampling, experiment split."""

import torch

from fleetmind.augmentation.exp_split import experiment_split
from fleetmind.augmentation.gan import ConditionalWaveGAN
from fleetmind.augmentation.sampling import denormalize, downsample, normalize, upsample


def test_downsample_upsample_roundtrip():
    x = [0.5 + 0.4 * __import__("math").sin(i / 40.0) for i in range(2048)]
    ds = downsample(x, factor=8)
    assert len(ds) == 256
    up = upsample(ds, target_len=2048)
    assert len(up) == 2048
    assert all(abs(a - b) < 0.1 for a, b in zip(x[::64], up[::64], strict=True))


def test_normalize_denormalize_roundtrip():
    x = [0.5 * i / 256 for i in range(256)]
    scaled, lo, hi = normalize(x)
    assert max(scaled) <= 1.0 and min(scaled) >= -1.0
    back = denormalize(scaled, lo, hi)
    assert all(abs(a - b) < 1e-9 for a, b in zip(x, back, strict=True))


def test_gan_shapes_and_train_step():
    torch.manual_seed(0)
    gan = ConditionalWaveGAN(num_classes=4, latent_dim=8, seq_len=256)
    real = torch.randn(16, 1, 256).clamp(-1, 1)
    labels = torch.randint(0, 4, (16,))
    losses = gan.train_step(real, labels)
    assert set(losses) == {"loss_d", "loss_g"}
    synth = gan.generate([0, 1, 2, 3, 0, 1])
    assert synth.shape == (6, 1, 256)
    assert synth.abs().max() <= 1.0 + 1e-5


def test_experiment_split_deterministic_and_disjoint():
    ids = list(range(1, 17))
    tr1, te1 = experiment_split(ids, frac=0.7, seed=42)
    tr2, te2 = experiment_split(ids, frac=0.7, seed=42)
    assert tr1 == tr2 and te1 == te2
    assert len(tr1) == 11 and len(te1) == 5
    assert set(tr1).isdisjoint(set(te1))
    assert sorted(tr1 + te1) == ids
