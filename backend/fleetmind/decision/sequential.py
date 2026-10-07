"""Sequential probability ratio test: decide on accumulated evidence, not one window.

Every earlier decision rule in this project flags a machine if **any** single
0.17-second window looks bad. A genuine bearing fault persists across consecutive
windows; one misclassified window does not. Wald's SPRT turns that into a rule
with stated error targets instead of an invented "k of m" count.

The test
--------
Hypotheses: H0 the machine is healthy, H1 it is faulted. Each window contributes
its log-likelihood ratio, and the running sum S is compared with two boundaries:

    A = log((1 - beta) / alpha)     cross upward   -> decide FAULTED (flag)
    B = log(beta / (1 - alpha))     cross downward -> decide HEALTHY

with alpha the false-alarm target and beta the miss target (`OPEN_QUESTIONS.md` Q5).
On deciding HEALTHY the sum resets to zero and monitoring continues: a machine
cleared once is still watched. That makes this a repeated SPRT, the standard way
to run one continuously.

From a classifier posterior to a likelihood ratio
-------------------------------------------------
The CNN outputs P(faulted | x), not P(x | faulted). Bayes gives

    LLR(x) = log[p / (1 - p)] - log[prior / (1 - prior)]

so the training prior must be divided out. **It is derived, not chosen:** the
classifiers were trained with inverse-frequency class weights
(`ml/model.py:class_weights`), which makes the effective training prior uniform
over the four classes, so prior P(faulted) = 3/4.

Two limitations, stated rather than buried
------------------------------------------
* Wald's guarantees assume **independent** observations. Windows cut from one
  continuous recording are strongly autocorrelated, so the achieved error rates
  will not equal alpha and beta. They are measured and reported, never asserted.
* The posterior is clipped to [EPS, 1 - EPS] purely so log() stays finite. That
  bounds any single window's contribution at about 13.8 nats, which is **above**
  A (about 4.55). So one sufficiently over-confident window *can* still flag a
  machine on its own -- the SPRT does not make a single window harmless when the
  model is that sure. `phase4.py` measures how often that happens rather than
  hiding it behind a cap chosen to make the problem disappear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fleetmind.ml.model import LABEL_MAP

#: Defaults from OPEN_QUESTIONS.md Q5: false alarms are made the rarer error.
DEFAULT_ALPHA = 0.01
DEFAULT_BETA = 0.05

#: Effective training prior of "faulted": uniform over 4 classes under
#: inverse-frequency class weighting, 3 of which are faults. Derived, not tuned.
FAULTED_PRIOR = (len(LABEL_MAP) - 1) / len(LABEL_MAP)

#: Numerical guard only -- keeps log() finite. See the module docstring for why it
#: is deliberately NOT set low enough to stop a single window flagging a machine.
EPS = 1e-6


def boundaries(alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> tuple[float, float]:
    """Wald's (A, B). A > 0 > B for any sensible alpha, beta."""
    for name, v in (("alpha", alpha), ("beta", beta)):
        if not 0.0 < v < 0.5:
            raise ValueError(f"{name} must be in (0, 0.5), got {v}")
    return math.log((1.0 - beta) / alpha), math.log(beta / (1.0 - alpha))


def llr(p_faulted: float, prior: float = FAULTED_PRIOR) -> float:
    """Log-likelihood ratio of one window, with the training prior divided out."""
    if not 0.0 < prior < 1.0:
        raise ValueError(f"prior must be in (0, 1), got {prior}")
    p = min(1.0 - EPS, max(EPS, p_faulted))
    return math.log(p / (1.0 - p)) - math.log(prior / (1.0 - prior))


@dataclass(frozen=True)
class SprtResult:
    flagged: bool
    flag_index: int | None  # position in the stream at which A was crossed
    n_examined: int  # windows actually consumed (the test stops at a flag)
    n_healthy_decisions: int  # times B was crossed and the sum reset
    max_sum: float  # highest the evidence ever got, for margin reporting
    single_window_flag: bool  # flagged by one window alone, from a reset sum

    def as_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "flag_index": self.flag_index,
            "n_examined": self.n_examined,
            "n_healthy_decisions": self.n_healthy_decisions,
            "max_sum": round(self.max_sum, 6),
            "single_window_flag": self.single_window_flag,
        }


def run(llrs: list[float], alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> SprtResult:
    """Run the repeated SPRT over one machine's windows, in time order.

    Stops at the first flag: once a machine is flagged, further evidence does not
    un-flag it here (that is a human's call, via the dispatch workflow).
    """
    if not llrs:
        raise ValueError("run() received no evidence; refusing to decide on nothing")
    a, b = boundaries(alpha, beta)
    s = 0.0
    resets = 0
    max_sum = 0.0
    for i, x in enumerate(llrs):
        started_from_zero = s == 0.0
        s += x
        max_sum = max(max_sum, s)
        if s >= a:
            return SprtResult(True, i, i + 1, resets, max_sum, started_from_zero and x >= a)
        if s <= b:
            resets += 1
            s = 0.0
    return SprtResult(False, None, len(llrs), resets, max_sum, False)


def windows_needed_to_flag(
    per_window_llr: float, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA
) -> int | None:
    """Consecutive identical windows needed to flag from a reset sum; None if never."""
    a, _ = boundaries(alpha, beta)
    if per_window_llr <= 0:
        return None
    return max(1, math.ceil(a / per_window_llr - 1e-12))
