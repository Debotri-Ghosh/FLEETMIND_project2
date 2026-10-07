"""Full class posteriors for held-out windows, from a trained checkpoint.

`ml/predict.py` keeps only the argmax label and its probability, because that is
all a `ModelPrediction` row stores. Calibration, conformal prediction and
expected-cost decisions all need the *whole* distribution, so this module runs
the same inference and returns every class probability.

It deliberately reuses `to_fft_magnitude()` and the same 2048-sample
pad/truncate as `predict.py`, so a posterior here is the same posterior the
application would have produced. Any divergence would silently invalidate every
calibration number measured downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from fleetmind.ml.model import LABEL_MAP, BearingCNN, to_fft_magnitude

WINDOW_SAMPLES = 2048
MLRUNS = Path("mlruns")


@dataclass(frozen=True)
class WindowPosterior:
    """One held-out window's full predicted distribution and its true class."""

    window_id: int
    experiment_id: int
    window_index: int
    true_label: str
    probabilities: list[float]  # ordered as LABEL_MAP
    #: Pre-softmax outputs, ordered as LABEL_MAP. Temperature scaling divides
    #: these by T and re-softmaxes, so it cannot work from probabilities alone:
    #: the softmax has already discarded the scale that T is meant to adjust.
    logits: list[float] = field(default_factory=list)

    @property
    def predicted_label(self) -> str:
        return LABEL_MAP[max(range(len(self.probabilities)), key=self.probabilities.__getitem__)]

    @property
    def confidence(self) -> float:
        """The probability assigned to the predicted class."""
        return max(self.probabilities)

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.true_label


def checkpoint_path(depot_id: int, mlruns: Path = MLRUNS) -> Path:
    return mlruns / f"depot_{depot_id}" / "cnn_state_dict.pt"


def load_model(depot_id: int, mlruns: Path = MLRUNS) -> BearingCNN:
    model = BearingCNN(num_classes=len(LABEL_MAP))
    model.load_state_dict(torch.load(checkpoint_path(depot_id, mlruns), map_location="cpu"))
    model.eval()
    return model


def _prepare(values: list[float]) -> list[float]:
    """Pad or truncate to 2048 samples, then FFT-magnitude. Mirrors predict.py."""
    vals = list(values)
    if len(vals) < WINDOW_SAMPLES:
        vals += [0.0] * (WINDOW_SAMPLES - len(vals))
    else:
        vals = vals[:WINDOW_SAMPLES]
    return to_fft_magnitude(vals)


def posteriors(
    model: BearingCNN,
    windows: list[tuple[int, int, int, list[float], str]],
    *,
    batch_size: int = 64,
) -> list[WindowPosterior]:
    """Full posteriors for the windows `holdout.holdout_windows()` returned.

    Batched, because a per-window forward pass over ~950 windows per site is
    needlessly slow and the batching cannot change the result: the model is in
    eval() mode, so BatchNorm uses its running statistics rather than the batch.
    """
    out: list[WindowPosterior] = []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        x = torch.tensor([_prepare(vals) for _, _, _, vals, _ in chunk], dtype=torch.float32)
        x = x.unsqueeze(1)  # (batch, 1, bins)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        for (window_id, experiment_id, window_index, _vals, true_label), row, raw in zip(
            chunk, probs, logits, strict=True
        ):
            out.append(
                WindowPosterior(
                    window_id=window_id,
                    experiment_id=experiment_id,
                    window_index=window_index,
                    true_label=true_label,
                    probabilities=[float(v) for v in row],
                    logits=[float(v) for v in raw],
                )
            )
    return out
