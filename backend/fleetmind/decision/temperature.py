"""Temperature scaling: one fitted scalar, and nothing else.

Guo et al. 2017, "On Calibration of Modern Neural Networks". A single parameter
T > 0 divides the logits before the softmax:

    p(y | x) = softmax(z(x) / T)

T is **fitted** by minimising negative log-likelihood on a held-out calibration
set. It is never chosen by hand -- that would be one more invented constant, and
removing those is the point of this package.

Two properties make it the right first calibration step here:

* **It cannot change any prediction.** Dividing every logit by the same positive
  number preserves their order, so the argmax -- and therefore accuracy and every
  per-class recall -- is identical before and after. Only the *confidences* move.
  `test_temperature_never_changes_a_prediction` pins that.
* **It has one degree of freedom**, so it cannot meaningfully overfit a small
  calibration set. That matters a great deal here: the calibration set is four
  recordings per site, and windows inside a recording are strongly
  autocorrelated, so the effective sample size is far below the window count.

T > 1 softens an over-confident model; T < 1 sharpens an under-confident one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fleetmind.ml.model import LABEL_MAP

#: Searched range for T. Wide enough to cover both directions by a large margin;
#: a fit that lands on a bound is reported rather than silently clamped.
T_MIN = 0.05
T_MAX = 20.0


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float
    n_calibration_windows: int
    n_calibration_recordings: int
    iterations: int
    hit_bound: bool

    @property
    def nll_improvement(self) -> float:
        return self.nll_before - self.nll_after

    @property
    def direction(self) -> str:
        if self.temperature > 1.01:
            return "softened (model was over-confident)"
        if self.temperature < 0.99:
            return "sharpened (model was under-confident)"
        return "essentially unchanged"

    def as_dict(self) -> dict:
        return {
            "temperature": round(self.temperature, 6),
            "nll_before": round(self.nll_before, 6),
            "nll_after": round(self.nll_after, 6),
            "nll_improvement": round(self.nll_improvement, 6),
            "direction": self.direction,
            "n_calibration_windows": self.n_calibration_windows,
            "n_calibration_recordings": self.n_calibration_recordings,
            "iterations": self.iterations,
            "hit_bound": self.hit_bound,
        }


def _nll(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> float:
    return float(
        torch.nn.functional.cross_entropy(logits / temperature, targets, reduction="mean").item()
    )


def fit_temperature(
    logits: list[list[float]],
    true_labels: list[str],
    *,
    n_recordings: int = 0,
    max_iter: int = 200,
) -> TemperatureFit:
    """Fit T by minimising NLL on the calibration set.

    Optimised with L-BFGS over log(T), so T stays strictly positive without a
    constraint and the parameterisation is scale-free. The result is clamped to
    [T_MIN, T_MAX] and `hit_bound` records whether the clamp bound, because a T
    pinned at a bound means the fit failed rather than converged and must not be
    reported as a successful calibration.
    """
    if len(logits) != len(true_labels):
        raise ValueError("logits and true_labels must be the same length")
    if not logits:
        raise ValueError("fit_temperature received no calibration data")

    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor([LABEL_MAP.index(lbl) for lbl in true_labels], dtype=torch.long)

    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = 1 initially
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    steps = 0

    def closure() -> torch.Tensor:
        nonlocal steps
        steps += 1
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    optimizer.step(closure)  # type: ignore[arg-type]

    raw = float(torch.exp(log_t.detach()).item())
    if not math.isfinite(raw):
        raw = 1.0
    temperature = min(max(raw, T_MIN), T_MAX)

    return TemperatureFit(
        temperature=temperature,
        nll_before=_nll(z, y, 1.0),
        nll_after=_nll(z, y, temperature),
        n_calibration_windows=len(logits),
        n_calibration_recordings=n_recordings,
        iterations=steps,
        hit_bound=temperature in (T_MIN, T_MAX),
    )


def apply_temperature(logits: list[float], temperature: float) -> list[float]:
    """softmax(logits / T) for one window."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = [v / temperature for v in logits]
    biggest = max(scaled)
    exps = [math.exp(v - biggest) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]


def rescale(posteriors: list, temperature: float) -> list:
    """Return copies of `probs.WindowPosterior` with temperature-scaled probabilities.

    The logits are carried through unchanged, so a later stage can re-scale from
    the original scores rather than compounding two temperatures.
    """
    from dataclasses import replace

    return [replace(p, probabilities=apply_temperature(p.logits, temperature)) for p in posteriors]
