"""Window resampling + per-class stats for Phase 3 augmentation."""

from __future__ import annotations

import numpy as np


def downsample(values: list[float], factor: int = 8) -> list[float]:
    """Mean-pool a 2048 window down to 256 (GAN operates at 256)."""
    arr = np.asarray(values, dtype=np.float64)
    n = (len(arr) // factor) * factor
    arr = arr[:n].reshape(-1, factor).mean(axis=1)
    return arr.tolist()


def upsample(values: list[float], target_len: int = 2048) -> list[float]:
    """Linear-interpolate a 256 synthetic window back to 2048."""
    arr = np.asarray(values, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, arr).tolist()


def normalize(values: list[float]) -> tuple[list[float], float, float]:
    """Scale to [-1, 1] (tanh output range); returns (scaled, lo, hi)."""
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return [0.0] * len(arr), lo, hi
    return (((arr - lo) / (hi - lo)) * 2.0 - 1.0).tolist(), lo, hi


def denormalize(scaled: list[float], lo: float, hi: float) -> list[float]:
    arr = np.asarray(scaled, dtype=np.float64)
    if hi - lo < 1e-9:
        return [lo] * len(arr)
    return (((arr + 1.0) / 2.0) * (hi - lo) + lo).tolist()
