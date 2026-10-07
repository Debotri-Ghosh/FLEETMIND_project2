"""Recording-level decisions: one decision per machine, not per window.

The property that matters most is that **scoring nothing cannot look like a
result**. A recording-level accuracy is a ratio over as few as four recordings,
so an aggregation bug that silently dropped them all would compute 0/0 or return
an empty list, and an unguarded caller would print something plausible. Both
`aggregate([])` and `RecordingScore.accuracy` on an empty score therefore raise,
and `test_scoring_zero_recordings_raises_rather_than_reporting_an_accuracy` pins
that.

Every test asserts a positive count of what it examined.
"""

from __future__ import annotations

import math

import pytest

from fleetmind.decision import recording
from fleetmind.decision.probs import WindowPosterior
from fleetmind.ml.model import LABEL_MAP


def _softmax(logits):
    biggest = max(logits)
    exps = [math.exp(v - biggest) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def _win(experiment_id, window_index, true_label, logits):
    return WindowPosterior(
        window_id=experiment_id * 1000 + window_index,
        experiment_id=experiment_id,
        window_index=window_index,
        true_label=true_label,
        probabilities=_softmax(logits),
        logits=list(logits),
    )


def _onehot(label, magnitude=6.0):
    row = [0.0] * len(LABEL_MAP)
    row[LABEL_MAP.index(label)] = magnitude
    return row


# ── the guard against scoring nothing ──────────────────────────────────
def test_scoring_zero_recordings_raises_rather_than_reporting_an_accuracy():
    with pytest.raises(ValueError, match="refusing to score nothing"):
        recording.aggregate([])

    # And the score object refuses to divide by zero even if built directly.
    empty = recording.RecordingScore(
        vote="logits",
        n_recordings_scored=0,
        n_correct=0,
        n_windows_examined=0,
        decisions=[],
        classes_absent=[],
    )
    with pytest.raises(ValueError, match="no recordings were scored"):
        _ = empty.accuracy


def test_an_unknown_aggregation_rule_is_refused():
    windows = [_win(1, i, "Ball", _onehot("Ball")) for i in range(3)]
    with pytest.raises(ValueError, match="vote must be one of"):
        recording.aggregate(windows, vote="mean")


def test_a_recording_with_two_true_labels_is_refused():
    windows = [_win(1, 0, "Ball", _onehot("Ball")), _win(1, 1, "OuterRace", _onehot("Ball"))]
    with pytest.raises(ValueError, match="more than one label"):
        recording.aggregate(windows)


def test_logit_averaging_refuses_posteriors_with_no_logits():
    """Guards the Phase 2 field addition: a stale posterior must not score as 0."""
    stale = [
        WindowPosterior(
            window_id=i,
            experiment_id=1,
            window_index=i,
            true_label="Ball",
            probabilities=_softmax(_onehot("Ball")),
            logits=[],  # as an older probs.posteriors() would have produced
        )
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="needs logits"):
        recording.aggregate(stale, vote="logits")
    # The probability rule does not need them, so it still works.
    assert len(recording.aggregate(stale, vote="probs")) == 1


# ── the aggregation itself ─────────────────────────────────────────────
def test_one_decision_per_recording_regardless_of_window_count():
    windows = (
        [_win(10, i, "Ball", _onehot("Ball")) for i in range(59)]
        + [_win(11, i, "InnerRace", _onehot("InnerRace")) for i in range(59)]
        + [_win(12, i, "OuterRace", _onehot("OuterRace")) for i in range(118)]
    )
    assert len(windows) == 236
    decisions = recording.aggregate(windows)
    assert len(decisions) == 3, "three recordings in, three decisions out"
    assert [d.n_windows for d in decisions] == [59, 59, 118]
    assert sum(d.n_windows for d in decisions) == 236, "every window must be accounted for"
    assert all(d.correct for d in decisions)


def test_averaging_outvotes_a_minority_of_wrong_windows():
    """The point of aggregating: a few bad windows should not flip a machine."""
    windows = [_win(1, i, "Ball", _onehot("Ball", 4.0)) for i in range(40)] + [
        _win(1, 40 + i, "Ball", _onehot("OuterRace", 4.0)) for i in range(19)
    ]
    assert len(windows) == 59
    # Per-window, 19 of 59 are wrong.
    assert sum(1 for w in windows if w.predicted_label != "Ball") == 19
    decision = recording.aggregate(windows, vote="logits")[0]
    assert decision.n_windows == 59
    assert decision.predicted_label == "Ball"
    assert decision.correct


def test_averaging_can_also_be_outvoted_so_it_is_not_a_free_win():
    """Honest counter-case: aggregation flips a recording the wrong way too.

    Without this, the suite would only ever demonstrate aggregation helping,
    which would be evidence of a biased test set rather than of the method.
    """
    windows = [_win(1, i, "Ball", _onehot("Ball", 4.0)) for i in range(20)] + [
        _win(1, 20 + i, "OuterRace", 4.0 and _onehot("OuterRace", 4.0)) for i in range(39)
    ]
    # All 59 windows belong to a Ball recording; 39 of them look like OuterRace.
    windows = [_win(1, w.window_index, "Ball", w.logits) for w in windows]
    decision = recording.aggregate(windows, vote="logits")[0]
    assert decision.n_windows == 59
    assert decision.predicted_label == "OuterRace"
    assert not decision.correct, "aggregation must be able to be wrong"


def test_one_extreme_logit_hijacks_logit_averaging_but_not_the_other_rules():
    """The weakness of the rule this task specified, pinned rather than argued.

    Logits are unbounded, so one window at logit 40 outweighs nine at logit 1.0
    in the mean. Probabilities are bounded, so that window contributes at most
    1/n and cannot do the same. This is the reason all three rules are reported
    instead of one being chosen silently -- and it corrects the opposite claim
    that was originally written into recording.py's docstring.
    """
    windows = [_win(1, 0, "Ball", _onehot("OuterRace", 40.0))] + [
        _win(1, 1 + i, "Ball", _onehot("Ball", 1.0)) for i in range(9)
    ]
    assert len(windows) == 10
    assert sum(1 for w in windows if w.predicted_label == "Ball") == 9

    by_logits = recording.aggregate(windows, vote="logits")[0].predicted_label
    by_probs = recording.aggregate(windows, vote="probs")[0].predicted_label
    by_majority = recording.aggregate(windows, vote="majority")[0].predicted_label

    assert by_logits == "OuterRace", "one unbounded logit dominates the mean"
    assert by_probs == "Ball", "a bounded probability contributes at most 1/n"
    assert by_majority == "Ball", "9 of 10 windows argmax to Ball"
    assert len({by_logits, by_probs}) == 2, "the rules must be able to disagree"


def test_majority_vote_breaks_ties_deterministically():
    windows = [_win(1, 0, "Ball", _onehot("Ball")), _win(1, 1, "Ball", _onehot("InnerRace"))]
    first = recording.aggregate(windows, vote="majority")[0].predicted_label
    for _ in range(5):
        assert recording.aggregate(windows, vote="majority")[0].predicted_label == first


# ── the score object ───────────────────────────────────────────────────
def test_score_reports_counts_not_just_a_ratio():
    windows = (
        [_win(1, i, "Ball", _onehot("Ball")) for i in range(10)]
        + [_win(2, i, "InnerRace", _onehot("OuterRace")) for i in range(10)]
        + [_win(3, i, "OuterRace", _onehot("OuterRace")) for i in range(10)]
        + [_win(4, i, "Ball", _onehot("Ball")) for i in range(10)]
    )
    s = recording.score(windows, "logits", classes_absent=["Normal"])

    assert s.n_recordings_scored == 4, "the real split has 4 recordings per site"
    assert s.n_windows_examined == 40
    assert s.n_correct == 3
    assert s.accuracy == pytest.approx(0.75)
    assert s.per_class == {"Ball": "2/2", "InnerRace": "0/1", "OuterRace": "1/1"}
    assert s.classes_absent == ["Normal"]

    payload = s.as_dict()
    assert payload["n_recordings_scored"] == 4
    assert payload["n_windows_examined"] == 40
    assert payload["per_class_correct_over_total"]["InnerRace"] == "0/1"
    assert payload["classes_absent_from_holdout"] == ["Normal"]
    assert len(payload["decisions"]) == 4


def test_one_flipped_recording_moves_a_four_recording_site_by_25_points():
    """Quantifies why the report prints n: the estimate is extremely coarse."""
    base = [_win(e, i, "Ball", _onehot("Ball")) for e in (1, 2, 3, 4) for i in range(5)]
    assert recording.score(base, "logits").accuracy == pytest.approx(1.00)

    flipped = [
        _win(e, i, "Ball", _onehot("OuterRace") if e == 4 else _onehot("Ball"))
        for e in (1, 2, 3, 4)
        for i in range(5)
    ]
    s = recording.score(flipped, "logits")
    assert s.n_recordings_scored == 4
    assert s.accuracy == pytest.approx(0.75)
    assert 1.00 - s.accuracy == pytest.approx(0.25)


def test_the_report_reads_the_deployed_checkpoint_path():
    """It must score mlruns/depot_N, not the v2 directory."""
    from pathlib import Path

    for depot_id in (1, 2, 3, 4):
        path = recording.probs.checkpoint_path(depot_id, Path("mlruns"))
        assert path.parent.name == f"depot_{depot_id}", path
        assert "_v2" not in str(path), path
        assert path.name == "cnn_state_dict.pt"
