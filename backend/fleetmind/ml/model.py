"""1D-CNN bearing fault classifier."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn


class BearingCNN(nn.Module):
    """Reverted to original (64,1,1) kernels (v3) -- v2's widened kernels
    made accuracy WORSE (0.306 vs 0.527), disproving that hypothesis."""

    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


LABEL_MAP: list[str] = ["Normal", "InnerRace", "OuterRace", "Ball"]


def class_weights(rows: list[tuple], num_classes: int = len(LABEL_MAP)) -> torch.Tensor:
    counts = Counter(row[-1] for row in rows)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def to_fft_magnitude(values: list[float], n_bins: int = 512) -> list[float]:
    """FFT magnitude spectrum (first n_bins), log-scaled and z-normalized.
    Bin spacing at 12kHz/2048 samples ~= 5.86 Hz; 512 bins covers 0-3000 Hz,
    well above all CWRU fault characteristic frequencies (~200 Hz)."""
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(arr))[:n_bins]
    if len(spectrum) < n_bins:
        spectrum = np.pad(spectrum, (0, n_bins - len(spectrum)))
    spectrum = np.log1p(spectrum)
    mean, std = spectrum.mean(), spectrum.std()
    if std > 1e-8:
        spectrum = (spectrum - mean) / std
    return spectrum.tolist()
