"""Phase 3: the expected-cost policy, the cost matrix, and cross-site false alarms.

The properties that carry the weight:

* **The model's probability reaches the decision.** The heuristic ignored it
  (`test_the_heuristic_ignores_model_confidence` pins that flaw); the policy must
  not. `test_the_policy_responds_to_model_confidence` is the regression test for
  the fix.
* **Only cost ratios matter.** `costs.py` claims that scaling the whole matrix
  changes no decision, which is what justifies relative units and no invented
  currency. That claim is tested, not asserted.
* **"Unseen" healthy recordings really are unseen.** A site's own Normal recording
  is in its training set; scoring it as a false-alarm test would measure the model
  on its training data. `crosssite.assert_unseen` refuses that, and its test proves
  the guard actually fires.

Every test asserts a positive count of what it examined.
"""

from __future__ import annotations

import pytest

from fleetmind.agents.sync_equipment_status import _URGENCY_RE, URGENCY_ORDER
from fleetmind.decision import costs, crosssite, policy, quality
from fleetmind.ml.model import LABEL_MAP


def _probs(p_normal: float) -> list[float]:
    """A 4-class distribution with the given P(Normal), rest on OuterRace."""
    out = [0.0] * len(LABEL_MAP)
    out[LABEL_MAP.index("Normal")] = p_normal
    out[LABEL_MAP.index("OuterRace")] = 1.0 - p_normal
    return out


# ── the cost matrix ────────────────────────────────────────────────────
def test_every_matrix_entry_is_derived_from_a_named_primitive():
    p = costs.CostPrimitives(failure=100, repair=20, visit=2, stop=15)
    m = costs.matrix(p)
    assert set(m) == set(costs.ACTIONS) and len(m) == 4
    expected = {
        "monitor": (0, 100),
        "inspect": (2, 22),
        "schedule_maintenance": (20, 20),
        "stop_machine": (35, 35),
    }
    checked = 0
    for action, (healthy, faulted) in expected.items():
        assert m[action][costs.HEALTHY] == healthy, action
        assert m[action][costs.FAULTED] == faulted, action
        checked += 2
    assert checked == 8, "all eight entries must be checked"


def test_impossible_cost_assumptions_are_refused():
    with pytest.raises(ValueError, match="must be non-negative"):
        costs.CostPrimitives(visit=-1)
    # Repair at or above failure makes maintenance pointless; refuse it rather than
    # silently produce a policy that only ever monitors.
    for repair in (100.0, 150.0):
        with pytest.raises(ValueError, match="repair must cost less than failure"):
            costs.CostPrimitives(repair=repair)


def test_scaling_the_whole_matrix_changes_no_decision():
    """The justification for relative units, tested rather than asserted."""
    base = costs.matrix()
    checked = 0
    for factor in (0.01, 3.7, 1000.0):
        scaled = {a: {s: v * factor for s, v in row.items()} for a, row in base.items()}
        for i in range(101):
            p = i / 100
            assert policy.choose(p, base)[0] == policy.choose(p, scaled)[0], (factor, p)
            checked += 1
    assert checked == 3 * 101


# ── the policy ─────────────────────────────────────────────────────────
def test_the_policy_responds_to_model_confidence():
    """The fix for the flaw Phase 0 measured: probability must move the decision.

    The heuristic returned one identical risk score for confidences 0.26 to 0.9991.
    Here the same span must produce different actions.
    """
    tried = [0.001, 0.01, 0.10, 0.50, 0.95, 0.999]
    actions = [policy.choose(p)[0] for p in tried]
    assert len(actions) == 6
    assert len(set(actions)) >= 3, f"expected the action to change with P(faulted): {actions}"
    assert actions[0] == "monitor"
    assert actions[-1] == "schedule_maintenance"
    # And it is monotone: more evidence of a fault never chooses a milder action.
    ranks = [costs.ACTIONS.index(a) for a in actions]
    assert ranks == sorted(ranks), actions


def test_default_boundaries_follow_from_the_costs():
    """Derived by hand: monitor=100p, inspect=2+20p, schedule=20, stop=35.

    monitor=inspect at p = 2/80 = 0.025;  inspect=schedule at p = 18/20 = 0.9.
    """
    regions = policy.boundaries()
    assert [r["action"] for r in regions] == ["monitor", "inspect", "schedule_maintenance"]
    assert regions[0]["to_p"] == pytest.approx(0.025, abs=1e-4)
    assert regions[1]["to_p"] == pytest.approx(0.9, abs=1e-4)
    assert policy.flag_threshold() == pytest.approx(0.9, abs=1e-4)


def test_stop_machine_is_never_optimal_and_the_report_says_so():
    """Without degradation-rate data, urgency cannot be priced; it must not be faked."""
    assert policy.never_optimal() == ["stop_machine"]
    # It is dominated in BOTH states, which is why no p can select it.
    m = costs.matrix()
    for state in costs.STATES:
        assert m["stop_machine"][state] > m["schedule_maintenance"][state], state


def test_exact_ties_go_to_the_less_disruptive_action():
    action, table = policy.choose(0.025)  # monitor and inspect cost exactly 2.5
    assert table["monitor"] == pytest.approx(table["inspect"])
    assert action == "monitor"


def test_p_faulted_is_one_minus_p_normal_and_needs_no_severity_table():
    checked = 0
    for p_normal in (0.0, 0.2, 0.9, 1.0):
        assert policy.p_faulted(_probs(p_normal)) == pytest.approx(1.0 - p_normal)
        checked += 1
    assert checked == 4
    with pytest.raises(ValueError, match="class probabilities"):
        policy.p_faulted([0.5, 0.5])


def test_a_probability_outside_zero_to_one_is_refused():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="must be a probability"):
            policy.expected_costs(bad)


def test_decision_text_is_readable_by_the_existing_sync_bridge():
    """sync_equipment_status parses the leading [URGENCY] tag; it must still work."""
    seen = 0
    for p, expected_action in (
        (0.001, "monitor"),
        (0.5, "inspect"),
        (0.99, "schedule_maintenance"),
    ):
        action, table = policy.choose(p)
        assert action == expected_action
        text = policy.decision_text(action, p, table)
        tag = _URGENCY_RE.match(text)
        assert tag is not None, text
        assert tag.group(1) == costs.ACTION_TO_URGENCY[action]
        assert tag.group(1) in URGENCY_ORDER
        seen += 1
    assert seen == 3
    # Only maintenance reaches the level at which sync flags a machine.
    flagging = {
        a for a, u in costs.ACTION_TO_URGENCY.items() if URGENCY_ORDER[u] >= URGENCY_ORDER["HIGH"]
    }
    assert flagging == set(policy.ACTIONABLE)


def test_policy_decisions_score_with_the_same_code_as_the_baseline():
    """quality.score must accept PolicyDecision exactly as it accepts the heuristic's."""

    class _P:
        def __init__(self, exp, idx, label, p_normal):
            self.window_id = exp * 100 + idx
            self.experiment_id = exp
            self.window_index = idx
            self.true_label = label
            self.probabilities = _probs(p_normal)

    posts = [_P(1, i, "Normal", 0.999) for i in range(5)] + [
        _P(2, i, "Ball", 0.001) for i in range(5)
    ]
    decisions = policy.decide_all(posts)
    assert len(decisions) == 10
    report = quality.score(decisions, label="policy")
    assert report.n_recordings_examined == 2
    assert report.n_false_alarms == 0
    assert report.n_misses == 0
    assert report.false_alarm_rate == 0.0, "a healthy recording WAS examined, so 0.0 is a real rate"


# ── cross-site healthy recordings ──────────────────────────────────────
def test_each_model_gets_the_other_three_sites_healthy_recordings(archive_session):
    checked = 0
    for depot_id in (1, 2, 3, 4):
        groups = crosssite.unseen_healthy_windows(archive_session, depot_id)
        sources = [src for src, _ in groups]
        assert depot_id not in sources
        assert sorted(sources) == sorted({1, 2, 3, 4} - {depot_id})
        for _src, windows in groups:
            assert windows, "every source site has a Normal recording"
            assert {w[4] for w in windows} == {"Normal"}
            checked += len(windows)
    assert checked > 0


def test_assert_unseen_refuses_a_sites_own_healthy_recording(archive_session):
    """The guard is real: feeding a model its own training data must raise."""
    own = crosssite.normal_experiment_ids(archive_session, 1)
    assert len(own) == 1, "CWRU has exactly one Normal recording per site"
    with pytest.raises(AssertionError, match="not unseen"):
        crosssite.assert_unseen(archive_session, 1, own)
    # ...and accepts another site's.
    other = crosssite.normal_experiment_ids(archive_session, 2)
    crosssite.assert_unseen(archive_session, 1, other)
    with pytest.raises(ValueError, match="no healthy recordings"):
        crosssite.assert_unseen(archive_session, 1, [])
