"""1D-CNN bearing fault classifier.

Architecture (designed to train in <2 min on CPU for an M.Tech demo and to
meet the 99%+ accuracy target reported in the Meta-BOHB CNN-Transformer paper
on CWRU):

    [Conv1d(1->16, k=64, s=2)] -> BatchNorm -> ReLU
    -> [Conv1d(16->32, k=1, s=2)] -> BatchNorm -> ReLU
    -> [Conv1d(32->64, k=1, s=2)] -> BatchNorm -> ReLU
    -> AdaptiveAvgPool1d(1)   # global pooling over the time axis
    -> Linear(64 -> num_classes)

Input shape:  (B, 1, 2048)
Output logits: (B, num_classes)

Total params: ~24k.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BearingCNN(nn.Module):
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
