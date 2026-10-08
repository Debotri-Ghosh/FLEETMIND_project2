"""Phase 5: the expected-cost policy and SPRT wired into the live product.

Covers the sync bridge's SPRT flagging rule, the shared `policy.assess()`
summary, the LLM tool that exposes it, and the evidence API that shows the
policy's decision beside the heuristic baseline's.

Every test that asserts "nothing was flagged" also asserts how much evidence it
examined, so it cannot pass by looking at nothing (CLAUDE.md constraint 9).
"""

import json

import pytest

from fleetmind.agents.sync_equipment_status import (
    policy_decision_rows,
    sprt_trigger,
    sync_equipment_status,
)
from fleetmind.agents.tools import EvidenceContext, get_decision_policy, run_tool
from fleetmind.db.models import (
    AgentDecision,
    Equipment,
    ModelPrediction,
    ServiceRequest,
    SignalWindow,
)
from fleetmind.decision import policy, sequential
from fleetmind.ml.model import LABEL_MAP

EXP_ID = 1  # the one experiment product_session seeds, linked to equipment 1


def _probs(p_faulted: float) -> list[float]:
    """A 4-class distribution with the given P(faulted), spread over the faults."""
    out = [p_faulted / (len(LABEL_MAP) - 1)] * len(LABEL_MAP)
    out[policy.NORMAL_INDEX] = 1.0 - p_faulted
    return out


def _add_policy_windows(session, p_values: list[float | None], start_index: int = 1) -> list[int]:
    """Append windows to the seeded recording, each with a policy decision."""
    decision_ids = []
    for offset, p in enumerate(p_values):
        w = SignalWindow(
            experiment_id=EXP_ID,
            window_index=start_index + offset,
            start_sample=0,
            end_sample=8,
            raw_values=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            label="InnerRace",
        )
        session.add(w)
        session.flush()
        pred = ModelPrediction(
            window_id=w.id,
            model_version="v1",
            predicted_label="InnerRace",
            confidence=0.9,
            class_probabilities=_probs(p) if p is not None else None,
        )
        session.add(pred)
        session.flush()
        d = AgentDecision(
            prediction_id=pred.id,
            agent_name=policy.AGENT_NAME,
            decision_text=f"[HIGH] p={p}",
            risk_score=p,
        )
        session.add(d)
        session.flush()
        decision_ids.append(d.id)
    session.commit()
    return decision_ids


# ── the sync bridge's SPRT rule ─────────────────────────────────────────


def test_policy_mode_ignores_heuristic_decisions(product_session):
    """The seeded [HIGH] heuristic decision flags under the heuristic rule but is
    not evidence for the policy, so it must not flag the machine there."""
    heuristic = product_session.query(AgentDecision).filter_by(agent_name="risk").count()
    assert heuristic == 1  # there really is a HIGH decision to ignore

    changes = sync_equipment_status(product_session, policy="expected_cost")

    assert changes == []
    assert product_session.get(Equipment, 1).status == "active"
    assert policy_decision_rows(product_session, EXP_ID) == []


def test_sustained_fault_evidence_flags_and_points_at_the_crossing_window(product_session):
    ids = _add_policy_windows(product_session, [0.99, 0.99, 0.99, 0.99])
    per_window = sequential.llr(0.99)
    needed = sequential.windows_needed_to_flag(per_window)
    assert needed is not None and 1 < needed <= 4  # accumulation is actually exercised

    sync_equipment_status(product_session, policy="expected_cost")
    product_session.commit()

    assert product_session.get(Equipment, 1).status == "maintenance"
    req = product_session.query(ServiceRequest).one()
    assert req.agent_decision_id == ids[needed - 1]
    assert req.priority == "high"
    assert "sequential probability ratio test" in req.description
    assert f"after {needed} window(s)" in req.description


def test_one_isolated_alarming_window_does_not_flag_under_the_sprt(product_session):
    """The heuristic flags on ANY high window; the SPRT needs accumulated evidence."""
    p_alarm = 0.97  # llr ~ 2.38, below A ~ 4.55
    assert 0 < sequential.llr(p_alarm) < sequential.boundaries()[0]
    _add_policy_windows(product_session, [0.2, 0.2, p_alarm, 0.2, 0.2])

    changes = sync_equipment_status(product_session, policy="expected_cost")

    assert changes == []
    assert product_session.get(Equipment, 1).status == "active"
    _, result = sprt_trigger(policy_decision_rows(product_session, EXP_ID))
    assert result is not None and result.n_examined == 5
    assert result.flagged is False


def test_a_row_without_p_faulted_is_skipped_not_read_as_zero(product_session):
    _add_policy_windows(product_session, [None, None])
    rows = policy_decision_rows(product_session, EXP_ID)
    assert len(rows) == 2
    trigger, result = sprt_trigger(rows)
    assert trigger is None and result is None  # no evidence -> no decision at all


def test_policy_flag_clears_when_the_evidence_changes(product_session):
    _add_policy_windows(product_session, [0.999, 0.999])
    sync_equipment_status(product_session, policy="expected_cost")
    product_session.commit()
    assert product_session.get(Equipment, 1).status == "maintenance"

    for d in product_session.query(AgentDecision).filter_by(agent_name=policy.AGENT_NAME):
        d.risk_score = 0.01
    product_session.commit()
    changes = sync_equipment_status(product_session, policy="expected_cost")

    assert [(c[1], c[2]) for c in changes] == [("maintenance", "active")]


def test_policy_sync_is_idempotent(product_session):
    _add_policy_windows(product_session, [0.999, 0.999])
    sync_equipment_status(product_session, policy="expected_cost")
    product_session.commit()
    log: list = []
    second = sync_equipment_status(product_session, policy="expected_cost", filed_requests_log=log)
    assert second == [] and log == []
    assert product_session.query(ServiceRequest).count() == 1


def test_unknown_policy_is_refused(product_session):
    with pytest.raises(ValueError):
        sync_equipment_status(product_session, policy="vibes")


# ── policy.assess and the LLM tool ──────────────────────────────────────


def test_assess_agrees_with_choose_and_names_every_class():
    probs = _probs(0.95)
    a = policy.assess(probs)
    action, table = policy.choose(policy.p_faulted(probs))
    assert a["action"] == action == "schedule_maintenance"
    assert a["urgency"] == "HIGH"
    assert list(a["class_probabilities"]) == list(LABEL_MAP)
    assert a["expected_costs"] == {k: round(v, 4) for k, v in table.items()}
    assert a["flag_threshold_p_faulted"] == pytest.approx(0.9, abs=1e-4)
    assert a["never_optimal_actions"] == ["stop_machine"]


def _ctx(**over) -> EvidenceContext:
    base = dict(
        service_request_id=1,
        window_id=1,
        equipment_name="M",
        site_name="Plant A",
        depot_id=1,
        load_hp=0,
        nominal_rpm=1797.0,
        sensor_position="DE",
        sampling_rate_hz=12000,
        predicted_label="InnerRace",
        model_confidence=0.9,
        model_version="v1",
        raw_values=[0.0] * 16,
        stored_risk_score=0.95,
        stored_decision_text="[HIGH] x",
    )
    base.update(over)
    return EvidenceContext(**base)


def test_decision_policy_tool_reports_the_policy_as_authoritative_when_it_decided():
    out = get_decision_policy(
        _ctx(stored_agent_name=policy.AGENT_NAME, class_probabilities=_probs(0.95))
    )
    assert out["available"] is True
    assert out["policy_is_authoritative_here"] is True
    assert out["p_faulted"] == pytest.approx(0.95)
    # dispatched by name, exactly as the LLM loop calls it
    assert (
        json.loads(run_tool(_ctx(class_probabilities=_probs(0.5)), "get_decision_policy", {}))[
            "action"
        ]
        == policy.choose(0.5)[0]
    )


def test_decision_policy_tool_says_so_when_no_distribution_is_stored():
    out = get_decision_policy(_ctx(stored_agent_name="risk"))
    assert out["available"] is False
    assert "0007" in out["reason"]


# ── the evidence API ────────────────────────────────────────────────────


def test_evidence_shows_the_policy_decision_beside_the_baseline(product_client, product_session):
    _add_policy_windows(product_session, [0.999, 0.999, 0.999])
    sync_equipment_status(product_session, policy="expected_cost")
    product_session.commit()
    rid = product_session.query(ServiceRequest).one().id

    body = product_client.get(f"/service-requests/{rid}/evidence").json()

    assert body["decided_by"] == policy.AGENT_NAME
    pa = body["policy_assessment"]
    assert pa["action"] == "schedule_maintenance"
    assert pa["p_faulted"] == pytest.approx(0.999)
    assert set(pa["expected_costs"]) == set(policy.costs.ACTIONS)
    sprt = body["sprt"]
    assert sprt["flagged"] is True
    assert sprt["windows_available"] == 3
    assert 1 <= sprt["windows_examined"] <= 3
    assert sprt["alpha"] == sequential.DEFAULT_ALPHA
    # the heuristic baseline is still shown, recomputed, for comparison
    assert 0.0 <= body["baseline_risk_score"] <= 1.0
    assert body["baseline_urgency"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_evidence_says_when_the_recording_was_training_data(product_client, product_session):
    """The fixture site has one recording; a single-recording class always goes to
    train under stratified_experiment_split, so this one is in-sample -- and the
    package must say so rather than imply the model never saw it."""
    from fleetmind.decision.holdout import split_for_depot

    sync_equipment_status(product_session)
    product_session.commit()
    rid = product_session.query(ServiceRequest).one().id
    split = split_for_depot(product_session, 1)
    assert split.train_experiment_ids == [EXP_ID]  # the split really was computed

    body = product_client.get(f"/service-requests/{rid}/evidence").json()

    assert body["recording_in_training_split"] is True


def test_evidence_for_a_heuristic_request_carries_no_policy_block(product_client, product_session):
    sync_equipment_status(product_session)  # fixture pins heuristic
    product_session.commit()
    rid = product_session.query(ServiceRequest).one().id

    body = product_client.get(f"/service-requests/{rid}/evidence").json()

    assert body["decided_by"] == "risk"
    assert body["policy_assessment"] is None  # fixture prediction has no distribution
    assert body["sprt"] is None
    assert body["baseline_urgency"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
