"""Tests for the replay engine and the one-call pipeline.

The replay engine simulates arrival over a static archive by toggling
signal_windows.released_at; the pipeline turns arrivals into predictions,
agent decisions, and (via the sync bridge) equipment flags + AI-filed
service requests.
"""

from datetime import UTC, datetime

import pytest
import torch
from sqlalchemy.orm import sessionmaker

from fleetmind.db.models import (
    AgentDecision,
    Customer,
    Depot,
    Equipment,
    Experiment,
    ModelPrediction,
    ServiceRequest,
    SignalWindow,
    Technician,
    WorkOrder,
    seed_depots_payload,
)
from fleetmind.ml.model import LABEL_MAP, BearingCNN
from fleetmind.pipeline import process_arrived_windows
from fleetmind.replay.engine import release_next_batch, replay_status, reset_replay


@pytest.fixture()
def model_dir(tmp_path):
    """A depot_1 checkpoint so the pipeline has something to run."""
    depot_dir = tmp_path / "depot_1"
    depot_dir.mkdir()
    torch.save(BearingCNN(num_classes=len(LABEL_MAP)).state_dict(), depot_dir / "cnn_state_dict.pt")
    return tmp_path


@pytest.fixture()
def replay_session(db_engine):
    """Depot 1, one experiment, 10 unreleased windows, one linked equipment."""
    session = sessionmaker(bind=db_engine, expire_on_commit=False)()
    for row in seed_depots_payload():
        session.add(Depot(**row))
    session.add(Customer(id=1, name="Acme Logistics", email="ops@acme.com", company="Acme Ltd"))
    session.add(Technician(id=1, name="Alice Chen", phone="+1-555-0101", status="available"))
    session.flush()

    exp = Experiment(
        depot_id=1,
        fault_type="InnerRace",
        fault_diameter_in=0.007,
        sensor_position="DE",
        sampling_rate_hz=12000,
        source_file="12k_DE/Load_0/InnerRace/IR007_0.mat",
        source_variable="X097_DE_time",
        n_samples=4096,
    )
    session.add(exp)
    session.flush()
    for i in range(10):
        session.add(
            SignalWindow(
                experiment_id=exp.id,
                window_index=i,
                start_sample=i * 2048,
                end_sample=(i + 1) * 2048,
                raw_values=[0.05 * (i + 1)] * 2048,
                label="InnerRace",
                released_at=None,
            )
        )
    session.add(
        Equipment(
            id=1,
            depot_id=1,
            customer_id=1,
            name="Bearing Motor A1",
            equipment_type="Motor",
            status="active",
            linked_experiment_id=exp.id,
        )
    )
    session.commit()
    yield session
    session.close()


# --- engine: status -----------------------------------------------------


def test_status_starts_at_zero_released(replay_session):
    status = replay_status(replay_session)
    assert (status.total_windows, status.released, status.processed, status.pending) == (
        10,
        0,
        0,
        0,
    )
    assert status.percent_released == 0.0
    assert status.complete is False


def test_status_tracks_release_progress(replay_session):
    release_next_batch(replay_session, batch_size=4)
    replay_session.commit()

    status = replay_status(replay_session)
    assert status.released == 4
    assert status.pending == 4  # released but not yet predicted
    assert status.percent_released == 40.0


# --- engine: release ----------------------------------------------------


def test_release_walks_the_archive_oldest_first(replay_session):
    release_next_batch(replay_session, batch_size=3)
    replay_session.commit()

    released = (
        replay_session.query(SignalWindow.window_index)
        .filter(SignalWindow.released_at.is_not(None))
        .order_by(SignalWindow.window_index)
        .all()
    )
    assert [r[0] for r in released] == [0, 1, 2]


def test_release_stops_at_the_end_of_the_archive(replay_session):
    assert release_next_batch(replay_session, batch_size=8) == 8
    replay_session.commit()
    assert release_next_batch(replay_session, batch_size=8) == 2  # only 2 left
    replay_session.commit()
    assert release_next_batch(replay_session, batch_size=8) == 0  # exhausted
    replay_session.commit()

    assert replay_status(replay_session).released == 10


def test_release_rejects_nonsense_batch_size(replay_session):
    with pytest.raises(ValueError):
        release_next_batch(replay_session, batch_size=0)


# --- pipeline -----------------------------------------------------------


def test_pipeline_processes_only_what_has_arrived(replay_session, model_dir):
    release_next_batch(replay_session, batch_size=3)
    replay_session.commit()

    result = process_arrived_windows(replay_session, model_dir=model_dir)

    assert result["predictions_written"] == 3
    assert result["decisions_written"] == 3
    assert replay_session.query(ModelPrediction).count() == 3
    assert replay_session.query(AgentDecision).count() == 3


def test_pipeline_tick_with_no_arrivals_is_a_noop(replay_session, model_dir):
    release_next_batch(replay_session, batch_size=2)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)

    result = process_arrived_windows(replay_session, model_dir=model_dir)

    assert result["predictions_written"] == 0
    assert result["decisions_written"] == 0
    assert replay_session.query(ModelPrediction).count() == 2


def test_pipeline_accumulates_across_successive_arrivals(replay_session, model_dir):
    for _ in range(3):
        release_next_batch(replay_session, batch_size=2)
        replay_session.commit()
        process_arrived_windows(replay_session, model_dir=model_dir)

    status = replay_status(replay_session)
    assert status.released == 6
    assert status.processed == 6
    assert status.pending == 0
    assert replay_session.query(AgentDecision).count() == 6


def test_pipeline_feeds_the_model_prediction_to_the_agents(replay_session, model_dir):
    """The decision must hang off the model's prediction row -- that link is
    what the operator UI reads to show fault type and risk score."""
    release_next_batch(replay_session, batch_size=2)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)

    prediction_ids = {p.id for p in replay_session.query(ModelPrediction).all()}
    decision_prediction_ids = {d.prediction_id for d in replay_session.query(AgentDecision).all()}
    assert decision_prediction_ids <= prediction_ids
    assert len(decision_prediction_ids) == 2
    # and the predictions came from the real model, not a placeholder
    versions = {p.model_version for p in replay_session.query(ModelPrediction).all()}
    assert versions == {"cnn_depot_1"}


# --- engine: reset ------------------------------------------------------


def test_reset_rewinds_everything_the_pipeline_produced(replay_session, model_dir):
    release_next_batch(replay_session, batch_size=10)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)

    summary = reset_replay(replay_session)
    replay_session.commit()

    assert summary["windows_unreleased"] == 10
    assert summary["predictions_deleted"] == 10
    assert summary["decisions_deleted"] == 10
    status = replay_status(replay_session)
    assert (status.released, status.processed, status.pending) == (0, 0, 0)


def test_reset_clears_ai_requests_and_unflags_equipment(replay_session, model_dir):
    """Force the equipment to be flagged, then confirm reset puts it back."""
    release_next_batch(replay_session, batch_size=10)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)

    # Make the decision unambiguously HIGH so the sync bridge flags equipment
    # and files an AI request, whatever the untrained model happened to predict.
    # Both fields are forced so this holds under either decision rule.
    for decision in replay_session.query(AgentDecision).all():
        decision.decision_text = "[HIGH] forced for test"
        decision.risk_score = 0.999999
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)
    assert replay_session.get(Equipment, 1).status == "maintenance"
    assert replay_session.query(ServiceRequest).count() == 1

    summary = reset_replay(replay_session)
    replay_session.commit()

    assert summary["ai_service_requests_deleted"] == 1
    assert summary["equipment_reset_to_active"] == 1
    assert replay_session.get(Equipment, 1).status == "active"
    assert replay_session.query(ServiceRequest).count() == 0


def test_reset_preserves_customer_filed_requests(replay_session, model_dir):
    """A human wrote these; rewinding the simulation must not erase them."""
    replay_session.add(
        ServiceRequest(
            customer_id=1,
            depot_id=1,
            equipment_id=1,
            title="Customer heard a noise",
            status="reported",
            priority="medium",
            agent_decision_id=None,
        )
    )
    replay_session.commit()

    release_next_batch(replay_session, batch_size=5)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)
    reset_replay(replay_session)
    replay_session.commit()

    remaining = replay_session.query(ServiceRequest).all()
    assert len(remaining) == 1
    assert remaining[0].title == "Customer heard a noise"


def test_reset_preserves_retired_equipment(replay_session):
    retired = Equipment(
        id=2,
        depot_id=1,
        customer_id=1,
        name="Retired Motor",
        equipment_type="Motor",
        status="retired",
        linked_experiment_id=None,
    )
    replay_session.add(retired)
    replay_session.commit()

    reset_replay(replay_session)
    replay_session.commit()

    assert replay_session.get(Equipment, 2).status == "retired"


def test_reset_deletes_work_orders_dispatched_against_ai_requests(replay_session, model_dir):
    release_next_batch(replay_session, batch_size=10)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)
    for decision in replay_session.query(AgentDecision).all():
        # Force a flag under either decision rule: the heuristic reads the
        # [URGENCY] tag, the expected-cost/SPRT path reads P(faulted).
        decision.decision_text = "[CRITICAL] forced for test"
        decision.risk_score = 0.999999
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)

    ai_request = replay_session.query(ServiceRequest).one()
    replay_session.add(
        WorkOrder(service_request_id=ai_request.id, technician_id=1, status="assigned")
    )
    replay_session.commit()

    summary = reset_replay(replay_session)
    replay_session.commit()

    assert summary["work_orders_deleted"] == 1
    assert replay_session.query(WorkOrder).count() == 0


def test_replay_is_repeatable_end_to_end(replay_session, model_dir):
    """Run the whole archive, reset, run it again -- same end state. This is
    what makes the demo re-runnable in front of an audience."""
    release_next_batch(replay_session, batch_size=10)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)
    first = replay_status(replay_session)

    reset_replay(replay_session)
    replay_session.commit()

    release_next_batch(replay_session, batch_size=10)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)
    second = replay_status(replay_session)

    assert (first.released, first.processed) == (second.released, second.processed) == (10, 10)
    assert second.complete is True


def test_reset_timestamp_is_timezone_aware(replay_session):
    """released_at is compared against now() in the worker; a naive timestamp
    here would blow up on the first comparison."""
    release_next_batch(replay_session, batch_size=1)
    replay_session.commit()

    window = replay_session.query(SignalWindow).filter(SignalWindow.released_at.is_not(None)).one()
    assert window.released_at is not None
    # sqlite drops tzinfo on round-trip; the value must still be sane either way
    assert (
        abs(
            (
                datetime.now(UTC).replace(tzinfo=None) - window.released_at.replace(tzinfo=None)
            ).total_seconds()
        )
        < 60
    )


# --- decision policy selection (OPEN_QUESTIONS.md Q6) ---------------------


def _set_policy(monkeypatch, value):
    from fleetmind.core.config import get_settings

    monkeypatch.setattr(get_settings(), "decision_policy", value)


def test_pipeline_expected_cost_decides_from_the_stored_distribution(
    replay_session, model_dir, monkeypatch
):
    from fleetmind.decision import policy

    _set_policy(monkeypatch, "expected_cost")
    release_next_batch(replay_session, batch_size=4)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)

    preds = {p.id: p for p in replay_session.query(ModelPrediction).all()}
    decisions = replay_session.query(AgentDecision).all()
    assert len(decisions) == 4
    for d in decisions:
        probs = preds[d.prediction_id].class_probabilities
        assert probs is not None and len(probs) == len(LABEL_MAP)
        assert sum(probs) == pytest.approx(1.0, abs=1e-5)
        assert d.agent_name == policy.AGENT_NAME
        assert d.risk_score == pytest.approx(policy.p_faulted(probs))
        assert d.decision_text.startswith("[")


def test_pipeline_heuristic_switch_restores_the_agent_chain(replay_session, model_dir, monkeypatch):
    from fleetmind.decision import policy

    _set_policy(monkeypatch, "heuristic")
    release_next_batch(replay_session, batch_size=3)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)

    names = {d.agent_name for d in replay_session.query(AgentDecision).all()}
    assert replay_session.query(AgentDecision).count() == 3
    assert policy.AGENT_NAME not in names


def test_policy_skips_predictions_without_a_distribution(replay_session, model_dir, monkeypatch):
    """Pre-0007 rows cannot be decided by the policy; they are left undecided,
    never decided by some other rule under the policy's name."""
    _set_policy(monkeypatch, "heuristic")
    release_next_batch(replay_session, batch_size=2)
    replay_session.commit()
    process_arrived_windows(replay_session, model_dir=model_dir)
    replay_session.query(AgentDecision).delete()
    for p in replay_session.query(ModelPrediction).all():
        p.class_probabilities = None
    replay_session.commit()

    _set_policy(monkeypatch, "expected_cost")
    result = process_arrived_windows(replay_session, model_dir=model_dir)

    assert replay_session.query(ModelPrediction).count() == 2
    assert result["decisions_written"] == 0
    assert replay_session.query(AgentDecision).count() == 0
