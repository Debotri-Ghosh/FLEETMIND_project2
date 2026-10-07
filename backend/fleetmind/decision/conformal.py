"""Split-conformal prediction sets: a coverage guarantee instead of a guess.

At 55-60% accuracy a point prediction is overconfident by construction. A
conformal prediction *set* says instead: "the true class is in {InnerRace, Ball}"
with a guaranteed marginal coverage level. A set of size 1 is an actionable
diagnosis; size 3 is an honest "we do not know", and a maintenance product should
be able to say that.

The method (split / inductive conformal, Vovk et al.; Lei & Wasserman)
---------------------------------------------------------------------
Non-conformity score for a labelled calibration point:

    s_i = 1 - p_hat(y_i | x_i)

i.e. how much probability mass the model withheld from the truth. Then take

    q_hat = the ceil((n+1)(1-alpha))-th smallest of the n calibration scores

and for a new x emit every label whose score is no worse:

    C(x) = { y : 1 - p_hat(y | x) <= q_hat }  =  { y : p_hat(y | x) >= 1 - q_hat }

Under **exchangeability** of calibration and test points this guarantees
P(y in C(x)) >= 1 - alpha, with no assumption about the model being good or
calibrated. The finite-sample correction is the `(n+1)` and the ceiling; using a
plain empirical quantile instead slightly undercovers, which is why
`quantile_index()` is a separate, directly tested function.

When ceil((n+1)(1-alpha)) > n the calibration set is too small to certify the
requested level at all, and the honest answer is the full label set. That case
returns `q_hat = inf` rather than silently falling back to the maximum score.

**Where the guarantee is weak here, stated rather than buried**
---------------------------------------------------------------
Exchangeability is the whole foundation, and this data violates it at the window
level. Calibration windows come from only **four recordings per site**, and
windows cut from one continuous recording are strongly autocorrelated — they are
nearer to four samples than to ~236. So the nominal 90% is optimistic, and any
empirical coverage close to 90% on this test set should be read as "consistent
with the target", not "the guarantee was verified". `ConformalEvaluation` carries
the recording counts so a report cannot quote coverage without them.

Normal is absent from both the calibration and test sets (one Normal recording
per site, kept in training by the stratified split), so coverage for the healthy
class is **unmeasurable**, and is reported as `None` rather than as a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from fleetmind.ml.model import LABEL_MAP


@dataclass(frozen=True)
class ConformalCalibration:
    """A fitted threshold, with the sample it came from."""

    alpha: float
    q_hat: float
    n_calibration_windows: int
    n_calibration_recordings: int
    quantile_index: int
    certifiable: bool  # False when n is too small for this alpha
    classes_present_in_calibration: list[str] = field(default_factory=list)

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def probability_floor(self) -> float:
        """A label enters the set when its probability is at least this."""
        return max(0.0, 1.0 - self.q_hat) if math.isfinite(self.q_hat) else 0.0

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "target_coverage": round(self.target_coverage, 6),
            "q_hat": None if not math.isfinite(self.q_hat) else round(self.q_hat, 6),
            "probability_floor": round(self.probability_floor, 6),
            "quantile_index": self.quantile_index,
            "n_calibration_windows": self.n_calibration_windows,
            "n_calibration_recordings": self.n_calibration_recordings,
            "certifiable_at_this_alpha": self.certifiable,
            "classes_present_in_calibration": self.classes_present_in_calibration,
        }


@dataclass(frozen=True)
class ConformalEvaluation:
    """Empirical behaviour of the sets on a test set none of this was fitted on."""

    calibration: ConformalCalibration
    n_test_windows: int
    n_test_recordings: int
    n_covered: int
    set_sizes: list[int]
    per_class_covered: dict[str, int]
    per_class_total: dict[str, int]
    classes_absent_from_test: list[str]

    @property
    def empirical_coverage(self) -> float:
        return self.n_covered / self.n_test_windows if self.n_test_windows else 0.0

    @property
    def mean_set_size(self) -> float:
        return sum(self.set_sizes) / len(self.set_sizes) if self.set_sizes else 0.0

    @property
    def set_size_distribution(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for size in self.set_sizes:
            out[size] = out.get(size, 0) + 1
        return dict(sorted(out.items()))

    @property
    def coverage_by_true_class(self) -> dict[str, float | None]:
        """None for any class with no test windows -- never 0.0 and never 1.0.

        A class that was never tested has no coverage, and reporting a number
        there would be a claim about data that does not exist.
        """
        out: dict[str, float | None] = {}
        for label in LABEL_MAP:
            total = self.per_class_total.get(label, 0)
            out[label] = None if total == 0 else self.per_class_covered.get(label, 0) / total
        return out

    def as_dict(self) -> dict:
        cov = self.coverage_by_true_class
        return {
            "calibration": self.calibration.as_dict(),
            "n_test_windows": self.n_test_windows,
            "n_test_recordings": self.n_test_recordings,
            "empirical_coverage": round(self.empirical_coverage, 6),
            "target_coverage": round(self.calibration.target_coverage, 6),
            "coverage_minus_target": round(
                self.empirical_coverage - self.calibration.target_coverage, 6
            ),
            "mean_set_size": round(self.mean_set_size, 6),
            "set_size_distribution": {str(k): v for k, v in self.set_size_distribution.items()},
            "coverage_by_true_class": {
                k: (None if v is None else round(v, 6)) for k, v in cov.items()
            },
            "classes_absent_from_test": self.classes_absent_from_test,
            "unmeasurable_coverage": [k for k, v in cov.items() if v is None],
        }


def nonconformity(probabilities: list[float], true_label: str) -> float:
    """1 - p(true class). Larger means the model withheld more from the truth."""
    return 1.0 - probabilities[LABEL_MAP.index(true_label)]


def quantile_index(n: int, alpha: float) -> int:
    """The 1-based rank `ceil((n+1)(1-alpha))` used by split conformal.

    Separated out and tested directly because the finite-sample correction is
    exactly the part that is easy to get subtly wrong: dropping the `+1` or the
    ceiling undercovers, and an off-by-one here would still produce plausible
    numbers.
    """
    if n <= 0:
        raise ValueError("quantile_index needs at least one calibration point")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil((n + 1) * (1.0 - alpha))


def calibrate(
    posteriors: list,
    *,
    alpha: float = 0.10,
    n_recordings: int = 0,
) -> ConformalCalibration:
    """Fit q_hat on calibration data. Never call this with test data."""
    if not posteriors:
        raise ValueError("calibrate() received no calibration data")

    scores = sorted(nonconformity(p.probabilities, p.true_label) for p in posteriors)
    n = len(scores)
    k = quantile_index(n, alpha)

    if k > n:
        # Too few calibration points to certify this level. The honest output is
        # an infinite threshold, i.e. every label is admitted.
        q_hat = math.inf
        certifiable = False
    else:
        q_hat = scores[k - 1]
        certifiable = True

    return ConformalCalibration(
        alpha=alpha,
        q_hat=q_hat,
        n_calibration_windows=n,
        n_calibration_recordings=n_recordings,
        quantile_index=k,
        certifiable=certifiable,
        classes_present_in_calibration=sorted({p.true_label for p in posteriors}),
    )


def prediction_set(probabilities: list[float], calibration: ConformalCalibration) -> list[str]:
    """Every label the threshold admits, in LABEL_MAP order.

    May be empty when the model spreads its mass thinly and q_hat is tight. An
    empty set is a legitimate conformal output -- it means "no label is plausible
    at this level" -- and it is *not* silently widened to the argmax, because
    doing so would quietly break the coverage guarantee.
    """
    if not math.isfinite(calibration.q_hat):
        return list(LABEL_MAP)
    return [
        label
        for index, label in enumerate(LABEL_MAP)
        if (1.0 - probabilities[index]) <= calibration.q_hat
    ]


def evaluate(
    posteriors: list,
    calibration: ConformalCalibration,
    *,
    n_recordings: int = 0,
) -> ConformalEvaluation:
    """Empirical coverage and set sizes on data the calibration never saw."""
    if not posteriors:
        raise ValueError("evaluate() received no test data; refusing to report coverage")

    covered = 0
    sizes: list[int] = []
    per_class_covered: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    for p in posteriors:
        label_set = prediction_set(p.probabilities, calibration)
        sizes.append(len(label_set))
        per_class_total[p.true_label] = per_class_total.get(p.true_label, 0) + 1
        if p.true_label in label_set:
            covered += 1
            per_class_covered[p.true_label] = per_class_covered.get(p.true_label, 0) + 1

    present = set(per_class_total)
    return ConformalEvaluation(
        calibration=calibration,
        n_test_windows=len(posteriors),
        n_test_recordings=n_recordings,
        n_covered=covered,
        set_sizes=sizes,
        per_class_covered=per_class_covered,
        per_class_total=per_class_total,
        classes_absent_from_test=sorted(set(LABEL_MAP) - present),
    )
