"""Phase 4: the repeated sequential probability ratio test.

Pinned here: Wald's boundaries match their formulas; the training prior is divided
out (so a window at exactly the prior carries zero evidence); a stream of healthy
evidence of any length never flags; strong evidence flags at the right window; the
sum resets on a healthy decision so a machine cleared earlier is still watched; and
the honest limitation -- one sufficiently confident window CAN still flag alone --
is asserted rather than hidden.

Every test asserts a positive count of what it examined.
"""

from __future__ import annotations

import math

import pytest

from fleetmind.decision import sequential


def test_boundaries_match_walds_formulas():
    checked = 0
    for alpha, beta in ((0.01, 0.05), (0.05, 0.05), (0.001, 0.1)):
        a, b = sequential.boundaries(alpha, beta)
        assert a == pytest.approx(math.log((1 - beta) / alpha))
        assert b == pytest.approx(math.log(beta / (1 - alpha)))
        assert a > 0 > b
        checked += 1
    assert checked == 3
    # The defaults, by hand: log(0.95/0.01) = log 95, log(0.05/0.99).
    a, b = sequential.boundaries()
    assert a == pytest.approx(4.5539, abs=1e-4)
    assert b == pytest.approx(-2.9857, abs=1e-4)


def test_impossible_error_targets_are_refused():
    for bad in (0.0, 0.5, 0.9, -0.1):
        with pytest.raises(ValueError, match="must be in"):
            sequential.boundaries(alpha=bad)
        with pytest.raises(ValueError, match="must be in"):
            sequential.boundaries(beta=bad)


def test_the_prior_is_derived_from_class_weighting_not_chosen():
    # Inverse-frequency class weights -> uniform effective prior over 4 classes,
    # three of which are faults.
    assert pytest.approx(0.75) == sequential.FAULTED_PRIOR


def test_a_window_at_exactly_the_prior_carries_no_evidence():
    assert sequential.llr(sequential.FAULTED_PRIOR) == pytest.approx(0.0, abs=1e-12)
    # Above the prior is evidence FOR a fault, below it AGAINST -- even if p > 0.5.
    assert sequential.llr(0.9) > 0
    assert sequential.llr(0.6) < 0, "0.6 is BELOW the 0.75 prior, so it is evidence of health"


def test_the_clip_bounds_a_single_window_but_not_below_the_flag_boundary():
    """The limitation, asserted: one over-confident window can still flag alone."""
    a, _ = sequential.boundaries()
    biggest = sequential.llr(1.0)
    assert math.isfinite(biggest)
    assert biggest == pytest.approx(math.log((1 - 1e-6) / 1e-6) - math.log(3), abs=1e-6)
    assert biggest > a, "a single maximal window exceeds A -- this is NOT prevented"
    assert sequential.run([biggest]).single_window_flag is True


def test_a_long_stream_of_healthy_evidence_never_flags():
    stream = [sequential.llr(0.01)] * 500
    r = sequential.run(stream)
    assert r.n_examined == 500, "every window must be examined when nothing flags"
    assert not r.flagged and r.flag_index is None
    assert r.n_healthy_decisions > 100, "the sum should keep resetting at B"


def test_sustained_fault_evidence_flags_at_the_first_window_crossing_a():
    per = sequential.llr(0.95)  # log(19) - log(3) ~= 1.846 per window
    needed = sequential.windows_needed_to_flag(per)
    assert needed == 3  # 1.846 * 3 = 5.54 >= 4.554 > 1.846 * 2
    r = sequential.run([per] * 10)
    assert r.flagged and r.flag_index == needed - 1
    assert r.n_examined == needed, "the test must stop at the flag"
    assert not r.single_window_flag


def test_one_typical_misclassified_window_does_not_flag_a_healthy_stream():
    """The robustness the any-window rule lacks, for a moderate wrong window."""
    healthy = [sequential.llr(0.01)] * 30
    wrong = sequential.llr(0.95)
    r = sequential.run(healthy[:15] + [wrong] + healthy[15:])
    assert r.n_examined == 31
    assert not r.flagged


def test_the_sum_resets_so_a_machine_cleared_earlier_is_still_watched():
    healthy = [sequential.llr(0.01)] * 20
    fault = [sequential.llr(0.99)] * 5
    r = sequential.run(healthy + fault)
    assert r.n_healthy_decisions >= 1, "it must have cleared the machine first"
    assert r.flagged and r.flag_index >= 20, "and still catch the later fault"


def test_deciding_on_no_evidence_raises():
    with pytest.raises(ValueError, match="no evidence"):
        sequential.run([])


def test_windows_needed_handles_evidence_that_can_never_flag():
    assert sequential.windows_needed_to_flag(0.0) is None
    assert sequential.windows_needed_to_flag(-1.0) is None
    assert sequential.windows_needed_to_flag(10.0) == 1
