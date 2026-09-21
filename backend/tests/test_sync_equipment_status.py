"""Tests for fleetmind.agents.sync_equipment_status: the worst-urgency
aggregation, the retired-equipment exclusion, and idempotent re-runs.

Uses product_session (see conftest.py): Equipment 1 is linked to an
experiment with one HIGH agent_decision; Equipment 2 is unlinked
(linked_experiment_id is None, so it's excluded from the query entirely);
Equipment 3 is retired but linked to the same HIGH-decision experiment.
"""

from fleetmind.agents.sync_equipment_status import _worst_urgency, sync_equipment_status
from fleetmind.db.models import (
    AgentDecision,
    Equipment,
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
