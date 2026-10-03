"""Tests for the /replay control endpoints.

These cover the API contract and the guards around it (can't step while the
worker runs, reset stops the worker first). The worker's own loop is not
started against the real model here — tick_sync is patched — because these
tests are about the endpoint behaviour, not about torch inference, which
tests/test_replay.py already covers end to end.
"""

from datetime import UTC, datetime

import pytest

from fleetmind.replay.worker import TickResult, worker


@pytest.fixture(autouse=True)
def _reset_worker_state():
    """The worker is a module-level singleton shared across tests."""
    worker.running = False
    worker._task = None
    worker.ticks = 0
    worker.last_tick = None
    worker.last_error = None
    yield
    worker.running = False
    worker._task = None


def _quiet_tick(**overrides):
    defaults = dict(
        at=datetime.now(UTC),
        released=5,
        predictions_written=5,
        decisions_written=5,
        equipment_status_changes=1,
        service_requests_filed=1,
    )
    defaults.update(overrides)
    return TickResult(**defaults)


def test_status_reports_archive_and_worker_state(product_client):
    resp = product_client.get("/replay/status")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_windows"] == 1  # product_session seeds one window
    assert body["released"] == 0  # fixture leaves released_at NULL
    assert body["percent_released"] == 0.0
    assert body["complete"] is False
    assert body["worker"]["running"] is False


def test_step_runs_one_tick_and_reports_counts(product_client, monkeypatch):
    monkeypatch.setattr(worker, "tick_sync", lambda *a, **k: _quiet_tick())

    resp = product_client.post("/replay/step")

    assert resp.status_code == 200
    assert resp.json() == {
        "released": 5,
        "predictions_written": 5,
        "decisions_written": 5,
        "equipment_status_changes": 1,
        "service_requests_filed": 1,
        "error": None,
    }


def test_step_surfaces_a_failed_tick_instead_of_500ing(product_client, monkeypatch):
    """A tick that blows up must come back as a reported error, not an
    unhandled exception — the console needs to show what went wrong."""
    monkeypatch.setattr(
        worker,
        "tick_sync",
        lambda *a, **k: _quiet_tick(
            released=0,
            predictions_written=0,
            decisions_written=0,
            equipment_status_changes=0,
            service_requests_filed=0,
            error="OperationalError: database is gone",
        ),
    )

    resp = product_client.post("/replay/step")

    assert resp.status_code == 200
    assert resp.json()["error"] == "OperationalError: database is gone"


def test_step_is_refused_while_the_worker_is_running(product_client):
    worker.running = True

    resp = product_client.post("/replay/step")

    assert resp.status_code == 409
    assert "stop it before stepping" in resp.json()["detail"]


def test_start_sets_worker_parameters(product_client, monkeypatch):
    started = {}

    async def fake_start(interval_seconds=None, batch_size=None):
        started["interval_seconds"] = interval_seconds
        started["batch_size"] = batch_size
        worker.interval_seconds = interval_seconds or worker.interval_seconds
        worker.batch_size = batch_size or worker.batch_size
        worker.running = True
        return True

    monkeypatch.setattr(worker, "start", fake_start)

    resp = product_client.post("/replay/start", json={"interval_seconds": 1.5, "batch_size": 10})

    assert resp.status_code == 200
    assert started == {"interval_seconds": 1.5, "batch_size": 10}
    assert resp.json()["worker"]["running"] is True


def test_start_validates_out_of_range_parameters(product_client):
    assert product_client.post("/replay/start", json={"interval_seconds": 0.01}).status_code == 422
    assert product_client.post("/replay/start", json={"batch_size": 0}).status_code == 422
    assert product_client.post("/replay/start", json={"batch_size": 9999}).status_code == 422


def test_stop_halts_the_worker(product_client, monkeypatch):
    worker.running = True

    async def fake_stop():
        worker.running = False
        return True

    monkeypatch.setattr(worker, "stop", fake_stop)

    resp = product_client.post("/replay/stop")

    assert resp.status_code == 200
    assert resp.json()["worker"]["running"] is False


def test_reset_stops_the_worker_before_wiping(product_client, monkeypatch):
    """Resetting underneath a running loop would let it re-release windows
    mid-wipe, so reset must stop it first."""
    calls = []
    worker.running = True

    async def fake_stop():
        calls.append("stop")
        worker.running = False
        return True

    monkeypatch.setattr(worker, "stop", fake_stop)

    resp = product_client.post("/replay/reset")

    assert resp.status_code == 200
    assert calls == ["stop"]
    assert worker.running is False


def test_reset_reports_what_it_cleared(product_client, product_session):
    from fleetmind.agents.sync_equipment_status import sync_equipment_status

    # product_session seeds a HIGH decision; run the bridge so there is an
    # AI-filed request and a flagged equipment row for reset to clear.
    sync_equipment_status(product_session)
    product_session.commit()

    resp = product_client.post("/replay/reset")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ai_service_requests_deleted"] == 1
    assert body["equipment_reset_to_active"] == 1
    assert body["decisions_deleted"] == 1


def test_reset_zeroes_the_worker_counters(product_client):
    """A rewound simulation must read as rewound.

    start() already zeroes ticks, so this gap only showed between a reset and
    the next start -- which is exactly the state a demo is left in, where the
    dashboard would show a stale tick count beside 0 arrivals.
    """
    from fleetmind.replay.worker import worker

    worker.ticks = 10
    worker.last_error = "something old"

    resp = product_client.post("/replay/reset")

    assert resp.status_code == 200
    assert worker.ticks == 0
    assert worker.last_tick is None
    assert worker.last_error is None
    assert product_client.get("/replay/status").json()["worker"]["ticks"] == 0


def test_reset_preserves_the_configured_interval_and_batch_size(product_client):
    """Those are user settings, not per-run counters."""
    from fleetmind.replay.worker import worker

    worker.interval_seconds = 3.0
    worker.batch_size = 123

    product_client.post("/replay/reset")

    assert worker.interval_seconds == 3.0
    assert worker.batch_size == 123
