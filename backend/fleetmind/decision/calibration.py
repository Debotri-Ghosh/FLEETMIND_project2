"""Measure how badly the model's confidence means what it says.

Phase 0 measures only. Temperature scaling arrives in Phase 1, and it has to be
measured against these numbers to be claimed as an improvement.

Three standard quantities:

* **ECE** (Expected Calibration Error) -- bin predictions by confidence, and
  average |accuracy - mean confidence| over bins weighted by bin population. 0
  is perfect; a model that says "90%" and is right 90% of the time scores 0.
* **Brier score** -- mean squared error between the one-hot truth and the full
  predicted distribution. Unlike ECE it is a proper scoring rule, so it cannot
  be gamed by a degenerate predictor.
* **Reliability table** -- the per-bin numbers ECE averages over, kept because
  a single ECE figure hides whether the model is over- or under-confident.

A note on interpretation that matters for this project: a model that predicts
one class at 99.9% confidence and is right 50% of the time has a *terrible* ECE,
and the already-documented window #2713 case (Ball at 99.91% with the physics
not corroborating) is exactly that shape. ECE puts a number on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from fleetmind.ml.model import LABEL_MAP

DEFAULT_BINS = 10


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed: positive means over-confident (claims more than it delivers)."""
        return self.mean_confidence - self.accuracy

    def as_dict(self) -> dict:
        return {
            "range": [round(self.lower, 4), round(self.upper, 4)],
            "count": self.count,
            "mean_confidence": round(self.mean_confidence, 6),
            "accuracy": round(self.accuracy, 6),
            "gap": round(self.gap, 6),
        }


@dataclass(frozen=True)
class CalibrationReport:
    n_windows: int
    accuracy: float
    mean_confidence: float
    ece: float
    brier: float
    bins: list[ReliabilityBin]
    per_class_recall: dict[str, float]
    predicted_class_shares: dict[str, float]

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy. Positive means over-confident."""
        return self.mean_confidence - self.accuracy

    @property
    def n_bins_populated(self) -> int:
        """How many confidence bins actually held predictions.

        Reported because an ECE averaged over one populated bin is a much weaker
        statement than one averaged over ten, and the figure alone hides that.
        """
        return sum(1 for b in self.bins if b.count > 0)

    def as_dict(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "accuracy": round(self.accuracy, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "overconfidence": round(self.overconfidence, 6),
            "ece": round(self.ece, 6),
            "brier": round(self.brier, 6),
            "n_bins_populated": self.n_bins_populated,
            "per_class_recall": {k: round(v, 6) for k, v in self.per_class_recall.items()},
            "predicted_class_shares": {
                k: round(v, 6) for k, v in self.predicted_class_shares.items()
            },
            "bins": [b.as_dict() for b in self.bins],
        }


def _bin_edges(n_bins: int) -> list[tuple[float, float]]:
    """Equal-width bins over [0, 1]. The last one includes its upper edge."""
    step = 1.0 / n_bins
    return [(i * step, (i + 1) * step) for i in range(n_bins)]


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = DEFAULT_BINS
) -> tuple[float, list[ReliabilityBin]]:
    """ECE and the reliability bins it averages over.

    Equal-width binning (the standard definition), not equal-mass: equal-mass
    bins would hide that this model's confidences cluster near 1.0, which is
    itself the finding.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    total = len(confidences)
    if total == 0:
        return 0.0, []

    bins: list[ReliabilityBin] = []
    ece = 0.0
    for index, (lower, upper) in enumerate(_bin_edges(n_bins)):
        last = index == n_bins - 1

        def in_bin(c: float, lower: float = lower, upper: float = upper, last: bool = last) -> bool:
            return lower <= c <= upper if last else lower <= c < upper

        members = [(c, ok) for c, ok in zip(confidences, correct, strict=True) if in_bin(c)]
        count = len(members)
        if count == 0:
            bins.append(ReliabilityBin(lower, upper, 0, 0.0, 0.0))
            continue
        mean_conf = sum(c for c, _ in members) / count
        acc = sum(1 for _, ok in members if ok) / count
        bins.append(ReliabilityBin(lower, upper, count, mean_conf, acc))
        ece += (count / total) * abs(mean_conf - acc)
    return ece, bins


def brier_score(probabilities: list[list[float]], true_labels: list[str]) -> float:
    """Multiclass Brier score: mean over windows of sum_k (p_k - y_k)^2.

    Ranges 0 (perfect) to 2 (confidently wrong on every window).
    """
    if len(probabilities) != len(true_labels):
        raise ValueError("probabilities and true_labels must be the same length")
    if not probabilities:
        return 0.0
    total = 0.0
    for probs, label in zip(probabilities, true_labels, strict=True):
        target = LABEL_MAP.index(label)
        total += sum((p - (1.0 if k == target else 0.0)) ** 2 for k, p in enumerate(probs))
    return total / len(probabilities)


def measure(posteriors: list, n_bins: int = DEFAULT_BINS) -> CalibrationReport:
    """Calibration of a list of `probs.WindowPosterior`.

    Typed loosely to avoid a circular import; only `.probabilities`,
    `.true_label`, `.predicted_label`, `.confidence` and `.correct` are used.
    """
    if not posteriors:
        return CalibrationReport(0, 0.0, 0.0, 0.0, 0.0, [], {}, {})

    confidences = [p.confidence for p in posteriors]
    correct = [p.correct for p in posteriors]
    ece, bins = expected_calibration_error(confidences, correct, n_bins)

    total = len(posteriors)
    present = sorted({p.true_label for p in posteriors})
    per_class_recall: dict[str, float] = {}
    for label in present:
        of_class = [p for p in posteriors if p.true_label == label]
        per_class_recall[label] = sum(1 for p in of_class if p.correct) / len(of_class)

    shares: dict[str, float] = {}
    for label in LABEL_MAP:
        shares[label] = sum(1 for p in posteriors if p.predicted_label == label) / total

    return CalibrationReport(
        n_windows=total,
        accuracy=sum(correct) / total,
        mean_confidence=sum(confidences) / total,
        ece=ece,
        brier=brier_score(
            [p.probabilities for p in posteriors], [p.true_label for p in posteriors]
        ),
        bins=bins,
        per_class_recall=per_class_recall,
        predicted_class_shares=shares,
    )
