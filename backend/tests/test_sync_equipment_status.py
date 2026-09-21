"""Tests for fleetmind.agents.sync_equipment_status: the worst-urgency
aggregation, the retired-equipment exclusion, and idempotent re-runs.

Uses product_session (see conftest.py): Equipment 1 is linked to an
experiment with one HIGH agent_decision; Equipment 2 is unlinked
(linked_experiment_id is None, so it's excluded from the query entirely);
Equipment 3 is retired but linked to the same HIGH-decision experiment.
"""

import pytest
from sqlalchemy.exc import IntegrityError

from fleetmind.agents import sync_equipment_status as sync_mod
from fleetmind.agents.sync_equipment_status import _worst_urgency, sync_equipment_status
from fleetmind.db.models import (
    AgentDecision,
    Equipment,
    Experiment,
    ModelPrediction,
    ServiceRequest,
    SignalWindow,
)


def test_worst_urgency_picks_highest_rank():
    assert (
        _worst_urgency(["[LOW] fine", "[CRITICAL] failing now", "[MEDIUM] watch it"]) == "CRITICAL"
    )


def test_worst_urgency_ignores_untagged_text():
    assert _worst_urgency(["no tag here", "also no tag"]) is None


def test_worst_urgency_empty_list():
    assert _worst_urgency([]) is None


def test_worst_urgency_skips_unrecognized_tag_but_keeps_known_ones():
    assert _worst_urgency(["[WEIRD] unknown tag", "[HIGH] known tag"]) == "HIGH"


def test_sync_flips_active_to_maintenance_on_high_decision(product_session):
    changes = sync_equipment_status(product_session)

    assert ("Bearing Motor A1", "active", "maintenance", "HIGH") in changes
    equip = product_session.get(Equipment, 1)
    assert equip.status == "maintenance"


def test_sync_excludes_retired_equipment_even_with_high_decision(product_session):
    changes = sync_equipment_status(product_session)

    assert not any(name == "Retired Motor" for name, *_ in changes)
    equip = product_session.get(Equipment, 3)
    assert equip.status == "retired"


def test_sync_leaves_unlinked_equipment_untouched(product_session):
    changes = sync_equipment_status(product_session)

    assert not any(name == "Bearing Motor A2" for name, *_ in changes)
    equip = product_session.get(Equipment, 2)
    assert equip.status == "active"


def test_sync_dry_run_reports_but_does_not_commit(product_session):
    changes = sync_equipment_status(product_session, dry_run=True)
    assert len(changes) == 1

    product_session.rollback()
    equip = product_session.get(Equipment, 1)
    assert equip.status == "active"


def test_sync_is_idempotent_on_second_run(product_session):
    first = sync_equipment_status(product_session)
    assert len(first) == 1

    second = sync_equipment_status(product_session)
    assert second == []


def test_sync_clears_flag_when_decisions_no_longer_warrant_it(product_session):
    sync_equipment_status(product_session)
    assert product_session.get(Equipment, 1).status == "maintenance"

    # Downgrade the only decision for this equipment's experiment to LOW.
    window = product_session.query(SignalWindow).filter_by(experiment_id=1).one()
    prediction = product_session.query(ModelPrediction).filter_by(window_id=window.id).one()
    decision = product_session.query(AgentDecision).filter_by(prediction_id=prediction.id).one()
    decision.decision_text = "[LOW] within normal range"
    product_session.commit()

    changes = sync_equipment_status(product_session)
    assert ("Bearing Motor A1", "maintenance", "active", "LOW") in changes
    assert product_session.get(Equipment, 1).status == "active"


def _ai_requests_for(session, equipment_id):
    return (
        session.query(ServiceRequest)
        .filter_by(equipment_id=equipment_id)
        .filter(ServiceRequest.agent_decision_id.is_not(None))
        .all()
    )


def test_sync_files_ai_service_request_on_maintenance_flag(product_session):
    sync_equipment_status(product_session)

    requests = _ai_requests_for(product_session, 1)
    assert len(requests) == 1
    req = requests[0]
    assert req.status == "reported"
    assert req.priority == "high"
    assert req.customer_id == 1
    assert req.depot_id == 1
    assert "High" in req.title
    assert req.agent_decision_id is not None
    assert req.prediction_id is not None


def test_sync_does_not_duplicate_open_ai_request_on_rerun(product_session):
    sync_equipment_status(product_session)
    sync_equipment_status(product_session)
    sync_equipment_status(product_session)

    assert len(_ai_requests_for(product_session, 1)) == 1


def test_sync_dry_run_does_not_create_service_request(product_session):
    sync_equipment_status(product_session, dry_run=True)
    product_session.rollback()

    assert _ai_requests_for(product_session, 1) == []


def test_sync_refiles_after_previous_ai_request_resolved(product_session):
    sync_equipment_status(product_session)
    first = _ai_requests_for(product_session, 1)[0]
    first.status = "resolved"
    product_session.commit()

    sync_equipment_status(product_session)

    open_requests = [r for r in _ai_requests_for(product_session, 1) if r.status != "resolved"]
    assert len(open_requests) == 1
    assert open_requests[0].id != first.id


def test_sync_does_not_file_request_for_unlinked_or_retired_equipment(product_session):
    sync_equipment_status(product_session)

    assert _ai_requests_for(product_session, 2) == []  # unlinked
    assert _ai_requests_for(product_session, 3) == []  # retired


def test_sync_filed_requests_log_records_equipment_and_urgency(product_session):
    log: list[tuple[str, str]] = []
    sync_equipment_status(product_session, filed_requests_log=log)

    assert log == [("Bearing Motor A1", "HIGH")]


# --- Hardening: malformed/missing decision data ------------------------------


def test_sync_leaves_equipment_active_when_only_untagged_decisions_exist(product_session):
    """Equipment 1's experiment has one HIGH decision (from the fixture) --
    retag it as untagged text so _worst_urgency/_worst_decision have zero
    usable decisions to work with (distinct from the "zero decisions at all"
    case already covered by test_sync_leaves_unlinked_equipment_untouched).
    Equipment must stay 'active' and no AI service request should be filed."""
    decision = product_session.query(AgentDecision).one()
    decision.decision_text = "elevated vibration, no urgency tag assigned by the risk agent"
    product_session.commit()

    changes = sync_equipment_status(product_session)

    assert changes == []
    assert product_session.get(Equipment, 1).status == "active"
    assert _ai_requests_for(product_session, 1) == []


# --- Hardening: race condition on the AI-filed-request unique index ---------


def test_partial_unique_index_rejects_duplicate_open_ai_request_at_db_level(product_session):
    """Proves the partial unique index itself (declared on ServiceRequest in
    db/models.py, mirrored for Postgres in alembic/versions/0003_...) is a
    real, enforced DB constraint under sqlite too -- not just documented.
    Bypasses the ORM-level "already open" check entirely: a second 'open'
    AI-filed request for the same equipment must fail at the DB with
    IntegrityError."""
    decision_id = product_session.query(AgentDecision).one().id

    product_session.add(
        ServiceRequest(
            customer_id=1,
            depot_id=1,
            equipment_id=1,
            title="first AI request",
            status="reported",
            agent_decision_id=decision_id,
        )
    )
    product_session.commit()

    with pytest.raises(IntegrityError), product_session.begin_nested():
        product_session.execute(
            sync_mod.insert(ServiceRequest).values(
                customer_id=1,
                depot_id=1,
                equipment_id=1,
                title="second AI request -- should be rejected",
                status="reported",
                agent_decision_id=decision_id,
            )
        )

    # The savepoint rollback (not a full session rollback) should leave the
    # session usable and the first request untouched.
    assert product_session.query(ServiceRequest).filter_by(equipment_id=1).count() == 1


def test_ensure_ai_service_request_treats_integrity_error_as_noop(product_session, monkeypatch):
    """Mocked-duplicate-insert simulation of the real race: the "already
    open" read-check (correctly) finds nothing yet, but the DB itself then
    rejects the insert because a concurrent sync run already won -- confirms
    _ensure_ai_service_request swallows the IntegrityError as a no-op
    (returns None) instead of propagating it and crashing the whole sync run,
    and leaves the session usable for anything after it."""
    decision = product_session.query(AgentDecision).one()
    decision_rows = [(decision.id, decision.decision_text, decision.prediction_id)]

    real_execute = product_session.execute

    def fake_execute(stmt, *args, **kwargs):
        if getattr(stmt, "is_insert", False):
            raise IntegrityError("mocked duplicate insert", {}, Exception("unique violation"))
        return real_execute(stmt, *args, **kwargs)

    monkeypatch.setattr(product_session, "execute", fake_execute)

    result = sync_mod._ensure_ai_service_request(
        product_session,
        equip_id=1,
        customer_id=1,
        depot_id=1,
        decision_rows=decision_rows,
        dry_run=False,
    )
    assert result is None

    monkeypatch.undo()
    assert _ai_requests_for(product_session, 1) == []
    # Session is still usable -- not left in an aborted state by the caught error.
    assert product_session.query(Equipment).count() == 3


# --- Hardening: mid-loop failure rolls back cleanly --------------------------


def test_sync_rolls_back_cleanly_on_error_mid_loop(product_session, monkeypatch):
    """Simulates Postgres going away (or any unexpected error) partway
    through the equipment loop: equipment 1 (id=1) is processed and updated
    first (id order), then a second linked+HIGH equipment (id=4, added here)
    triggers a forced failure. Mirrors main()'s try/except around the call to
    sync_equipment_status, which rolls back before re-raising -- confirms
    that rollback actually undoes equipment 1's uncommitted update too, so a
    mid-loop failure leaves the DB exactly as it was before the run (no
    partial commits), not half-updated."""
    exp2 = Experiment(
        depot_id=1,
        fault_type="OuterRace",
        fault_diameter_in=0.007,
        sensor_position="DE",
        sampling_rate_hz=12000,
        source_file="12k_DE/Load_0/OuterRace/OR007_0.mat",
        source_variable="X098_DE_time",
        n_samples=4096,
    )
    product_session.add(exp2)
    product_session.flush()
    window2 = SignalWindow(
        experiment_id=exp2.id,
        window_index=0,
        start_sample=0,
        end_sample=8,
        raw_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        label="OuterRace",
    )
    product_session.add(window2)
    product_session.flush()
    prediction2 = ModelPrediction(
        window_id=window2.id, model_version="v1", predicted_label="OuterRace", confidence=0.9
    )
    product_session.add(prediction2)
    product_session.flush()
    product_session.add(
        AgentDecision(
            prediction_id=prediction2.id,
            agent_name="risk",
            decision_text="[CRITICAL] failing now",
            risk_score=0.95,
        )
    )
    product_session.add(
        Equipment(
            id=4,
            depot_id=1,
            customer_id=1,
            name="Bearing Motor A4",
            equipment_type="Motor",
            status="active",
            linked_experiment_id=exp2.id,
        )
    )
    product_session.commit()

    original = sync_mod._ensure_ai_service_request

    def boom(*args, **kwargs):
        if kwargs.get("equip_id") == 4:
            raise RuntimeError("simulated DB failure mid-loop")
        return original(*args, **kwargs)

    monkeypatch.setattr(sync_mod, "_ensure_ai_service_request", boom)

    with pytest.raises(RuntimeError, match="simulated DB failure mid-loop"):
        sync_equipment_status(product_session)

    # Mirror main()'s error handling: roll back before doing anything else.
    product_session.rollback()

    assert product_session.get(Equipment, 1).status == "active"
    assert product_session.get(Equipment, 4).status == "active"
    assert _ai_requests_for(product_session, 1) == []
    assert _ai_requests_for(product_session, 4) == []
