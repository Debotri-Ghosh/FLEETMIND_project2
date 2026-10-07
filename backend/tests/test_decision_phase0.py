"""Phase 0 of the decision layer: pin the measured flaw and the split's limits.

Two things are protected here.

**The flaw.** `risk_agent()` reads `diagnosis_confidence`, which
`diagnostic_agent()` computed from signal RMS -- not the model's `confidence`.
So the classifier's uncertainty never reaches the decision. Phase 1 fixes that,
and `test_the_heuristic_ignores_model_confidence` is the regression test proving
the old behaviour is gone rather than merely assumed gone: it names the exact
confidence values it tried and asserts the number of distinct risk scores they
produced.

**The split's limits.** No Normal recording is ever held out, so the
false-alarm rate is unmeasurable. `quality.py` must report that as `None` with a
reason, never as `0.0`. A rate of zero would read as "we never false-alarm" on
the strength of having never looked, which is the single failure mode this
project keeps finding in its own tests.

Every test asserts a positive count of what it examined.
"""

from __future__ import annotations

import pytest

from fleetmind.decision import baseline, calibration, quality
from fleetmind.ml.model import LABEL_MAP


# ── the measured flaw ──────────────────────────────────────────────────
def _alternating(amplitude: float, n: int = 2048) -> list[float]:
    """A square-ish wave of known RMS, so `diagnostic_agent` sees a fixed signal."""
    return [amplitude * (1 if i % 2 else -1) for i in range(n)]


def test_the_heuristic_ignores_model_confidence():
    """The flaw, pinned: model confidence does not move the heuristic's decision.

    When Phase 1 routes the calibrated posterior into the decision, this test
    must be updated deliberately -- which is the point. It fails loudly instead
    of letting the regression return in silence.
    """
    tried = [0.26, 0.40, 0.55, 0.75, 0.9991]
    signal = _alternating(0.10)

    results = [
        baseline.decide(
            window_id=i,
            experiment_id=1,
            window_index=i,
            true_label="OuterRace",
            predicted_label="OuterRace",
            confidence=c,
            signal_values=signal,
            load_hp=3,
        )
        for i, c in enumerate(tried)
    ]

    assert len(results) == len(tried) == 5, "the test must actually try five confidences"

    distinct_risk = {r.risk_score for r in results}
    distinct_urgency = {r.urgency for r in results}

    # The claim: five very different model confidences, one risk score.
    assert len(distinct_risk) == 1, (
        f"expected the heuristic to ignore model confidence; got {len(distinct_risk)} "
        f"distinct risk scores {sorted(distinct_risk)} across confidences {tried}"
    )
    assert len(distinct_urgency) == 1, sorted(distinct_urgency)

    # And the confidences really were different, so the test is not vacuous.
    assert len({r.model_confidence for r in results}) == 5
    assert min(tried) < 0.3 < 0.9 < max(tried), "the span tried must straddle unsure/certain"


def test_signal_amplitude_does_move_the_heuristic():
    """The other half of the flaw: an invented RMS formula is what drives it.

    Without this, the test above could be explained by the risk score being
    saturated at 1.0 for every possible input, which would be a different
    (and less interesting) defect.
    """
    scores = {}
    for amplitude in (0.005, 0.02, 0.10):
        d = baseline.decide(
            window_id=1,
            experiment_id=1,
            window_index=0,
            true_label="OuterRace",
            predicted_label="OuterRace",
            confidence=0.9991,
            signal_values=_alternating(amplitude),
            load_hp=3,
        )
        scores[amplitude] = d.risk_score

    assert len(scores) == 3
    assert len(set(scores.values())) > 1, (
        f"signal amplitude should move the heuristic's risk score, but all three "
        f"amplitudes gave {scores}"
    )


def test_baseline_passes_load_hp_and_not_a_site_id():
    """Guards the resolved keying bug: the factor must follow motor load."""
    seen = {}
    for load_hp in (0, 3):
        seen[load_hp] = baseline.decide(
            window_id=1,
            experiment_id=1,
            window_index=0,
            true_label="InnerRace",
            predicted_label="InnerRace",
            confidence=0.8,
            signal_values=_alternating(0.02),
            load_hp=load_hp,
        ).risk_score
    assert len(seen) == 2
    assert seen[3] > seen[0], f"3 HP must score above 0 HP, got {seen}"


# ── the split's limits ─────────────────────────────────────────────────
class _Decision:
    """Minimal stand-in exposing exactly what quality.score() reads.

    Hand-rolled rather than a MagicMock: an auto-attribute mock would satisfy
    `.actionable` whatever the scorer asked for, and would pass even if the
    scorer read the wrong field.
    """

    def __init__(self, experiment_id, true_label, window_index, actionable):
        self.experiment_id = experiment_id
        self.true_label = true_label
        self.window_index = window_index
        self.actionable = actionable


def test_false_alarm_rate_is_none_not_zero_when_no_healthy_recording_is_held_out():
    """The critical honesty property of the whole Phase 0 report."""
    decisions = [_Decision(10, "OuterRace", i, i >= 2) for i in range(5)]
    report = quality.score(decisions, label="test")

    assert report.n_recordings_examined == 1
    assert report.n_windows_examined == 5
    assert report.n_healthy_recordings == 0

    assert report.false_alarm_rate is None, "must be None, never 0.0"
    assert "false_alarm_rate" not in report.measurable
    assert "false_alarm_rate" in report.not_measurable
    assert "Normal" in report.not_measurable["false_alarm_rate"]

    # Serialised form must carry the same distinction.
    payload = report.as_dict()
    assert payload["false_alarm_rate"] is None
    assert payload["n_healthy_recordings"] == 0
    assert "false_alarm_rate" in payload["not_measurable"]


def test_false_alarm_rate_is_computed_when_a_healthy_recording_is_present():
    """The companion guard: the None above is a real branch, not the only branch.

    Without this, `false_alarm_rate` could be hardcoded to None and every test
    above would still pass.
    """
    decisions = (
        [_Decision(1, "Normal", i, i == 0) for i in range(4)]  # one false alarm
        + [_Decision(2, "Normal", i, False) for i in range(4)]  # clean
        + [_Decision(3, "OuterRace", i, i >= 1) for i in range(4)]  # detected at 1
    )
    report = quality.score(decisions, label="test")

    assert report.n_recordings_examined == 3
    assert report.n_windows_examined == 12
    assert report.n_healthy_recordings == 2
    assert report.n_faulted_recordings == 1

    assert report.false_alarm_rate == pytest.approx(0.5)
    assert report.miss_rate == pytest.approx(0.0)
    assert report.mean_detection_delay == pytest.approx(1.0)
    assert set(report.measurable) == {
        "false_alarm_rate",
        "miss_rate",
        "mean_detection_delay",
    }
    assert report.not_measurable == {}


def test_a_miss_is_a_faulted_recording_never_flagged():
    decisions = [_Decision(7, "Ball", i, False) for i in range(6)]
    report = quality.score(decisions, label="test")
    assert report.n_windows_examined == 6
    assert report.n_faulted_recordings == 1
    assert report.n_misses == 1
    assert report.miss_rate == pytest.approx(1.0)
    assert report.mean_detection_delay is None
    assert "mean_detection_delay" in report.not_measurable


def test_scoring_nothing_raises_instead_of_reporting_zeros():
    """A report that examined nothing must not look like a clean result."""
    with pytest.raises(ValueError, match="examined no decisions"):
        quality.score([], label="test")


def test_a_recording_with_two_true_labels_is_rejected():
    decisions = [_Decision(1, "Ball", 0, True), _Decision(1, "OuterRace", 1, True)]
    with pytest.raises(ValueError, match="more than one true label"):
        quality.score(decisions, label="test")


# ── calibration arithmetic ─────────────────────────────────────────────
def test_perfect_calibration_scores_zero_ece():
    """Ten windows at 100% confidence, all correct: nothing to correct."""
    conf = [1.0] * 10
    ece, bins = calibration.expected_calibration_error(conf, [True] * 10)
    assert ece == pytest.approx(0.0)
    assert sum(b.count for b in bins) == 10, "every window must land in a bin"


def test_confident_and_wrong_scores_the_worst_possible_ece():
    conf = [1.0] * 10
    ece, bins = calibration.expected_calibration_error(conf, [False] * 10)
    assert ece == pytest.approx(1.0)
    assert sum(b.count for b in bins) == 10
    populated = [b for b in bins if b.count]
    assert len(populated) == 1
    assert populated[0].gap == pytest.approx(1.0), "gap must be positive = over-confident"


def test_ece_bins_account_for_every_prediction():
    """The guard against a binning bug silently dropping windows.

    A dropped window would lower ECE, i.e. flatter the model, which is exactly
    the direction of error that must not pass unnoticed.
    """
    conf = [i / 100 for i in range(101)]  # includes both 0.0 and 1.0
    correct = [i % 2 == 0 for i in range(101)]
    ece, bins = calibration.expected_calibration_error(conf, correct, n_bins=10)
    assert sum(b.count for b in bins) == 101, [b.count for b in bins]
    assert 0.0 <= ece <= 1.0
    assert len([b for b in bins if b.count]) == 10


def test_brier_score_endpoints():
    n = len(LABEL_MAP)
    onehot = [0.0] * n
    onehot[LABEL_MAP.index("Ball")] = 1.0
    assert calibration.brier_score([onehot], ["Ball"]) == pytest.approx(0.0)

    wrong = [0.0] * n
    wrong[LABEL_MAP.index("Normal")] = 1.0
    assert calibration.brier_score([wrong], ["Ball"]) == pytest.approx(2.0)

    uniform = [1.0 / n] * n
    expected = (1 - 1 / n) ** 2 + (n - 1) * (1 / n) ** 2
    assert calibration.brier_score([uniform], ["Ball"]) == pytest.approx(expected)


def test_measure_rejects_nothing_gracefully_but_reports_zero_windows():
    report = calibration.measure([])
    assert report.n_windows == 0
    assert report.n_bins_populated == 0


class _Post:
    def __init__(self, probabilities, true_label):
        self.probabilities = probabilities
        self.true_label = true_label

    @property
    def predicted_label(self):
        return LABEL_MAP[max(range(len(self.probabilities)), key=self.probabilities.__getitem__)]

    @property
    def confidence(self):
        return max(self.probabilities)

    @property
    def correct(self):
        return self.predicted_label == self.true_label


def test_measure_detects_a_majority_class_predictor():
    """The shape the federated model took: one class always, regardless of truth.

    Accuracy alone would read 0.5 and look mediocre; the predicted-class share
    makes the degeneracy explicit.
    """
    outer = LABEL_MAP.index("OuterRace")
    probs = [0.01, 0.01, 0.97, 0.01]
    posteriors = [_Post(probs, "OuterRace") for _ in range(10)] + [
        _Post(probs, "Ball") for _ in range(10)
    ]
    report = calibration.measure(posteriors)

    assert report.n_windows == 20
    assert report.accuracy == pytest.approx(0.5)
    assert report.predicted_class_shares["OuterRace"] == pytest.approx(1.0)
    assert report.predicted_class_shares["Ball"] == pytest.approx(0.0)
    assert report.per_class_recall == {"Ball": 0.0, "OuterRace": 1.0}
    assert report.overconfidence > 0.4, report.overconfidence
    assert probs[outer] == 0.97
