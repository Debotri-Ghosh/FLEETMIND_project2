"""Tests for the replay worker's own lifecycle (start/stop/loop control).

These exercise the *real* start/stop path rather than mocking it, which is
what tests/test_replay_api.py does -- and is exactly why the CancelledError
escape below went unnoticed until a browser hit it: asyncio.CancelledError
inherits from BaseException, so a `suppress(Exception)` around `await task`
lets it propagate out of the stop handler as a 500.

tick_sync is stubbed so the loop spins fast and never touches torch or the
DB; what is under test here is the control flow around it.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from fleetmind.replay.worker import ReplayWorker, TickResult


def _tick(released: int = 5) -> TickResult:
    return TickResult(
        at=datetime.now(UTC),
        released=released,
        predictions_written=released,
        decisions_written=released,
        equipment_status_changes=0,
        service_requests_filed=0,
    )


@pytest.fixture()
def worker():
    """A fresh worker per test -- never the module-level singleton, so these
    tests can't leak a running loop into another test."""
    w = ReplayWorker(interval_seconds=0.25, batch_size=5)
    w.tick_sync = lambda *a, **k: _tick()  # type: ignore[method-assign]
    yield w
    if w.running:
        asyncio.run(w.stop())


def test_stop_while_running_does_not_raise(worker):
    """The regression: stopping a *running* worker cancels its task, and the
    CancelledError from awaiting it must not escape stop()."""

    async def scenario():
        await worker.start()
        await asyncio.sleep(0.4)  # let at least one tick land
        assert worker.running is True
        await worker.stop()  # must not raise
        return worker.running

    assert asyncio.run(scenario()) is False


def test_stop_when_idle_is_a_safe_noop(worker):
    async def scenario():
        return await worker.stop()

    assert asyncio.run(scenario()) is False  # nothing to stop, reported honestly


def test_start_twice_does_not_spawn_a_second_loop(worker):
    """Double-clicking Start must not leave two loops racing over the same
    windows."""

    async def scenario():
        first = await worker.start()
        second = await worker.start()
        tasks_running = worker._task
        await worker.stop()
        return first, second, tasks_running

    first, second, task = asyncio.run(scenario())
    assert first is True  # started
    assert second is False  # already running -> no-op, not an error
    assert task is not None


def test_start_clamps_out_of_range_settings(worker):
    async def scenario():
        await worker.start(interval_seconds=999.0, batch_size=99999)
        await worker.stop()

    asyncio.run(scenario())
    assert worker.interval_seconds == 60.0  # MAX_INTERVAL_SECONDS
    assert worker.batch_size == 500  # MAX_BATCH_SIZE


def test_worker_auto_stops_when_the_archive_is_exhausted(worker):
    """A tick that releases nothing and predicts nothing means the archive is
    finished; the loop should end itself rather than spin forever."""
    worker.tick_sync = lambda *a, **k: _tick(released=0)  # type: ignore[method-assign]

    async def scenario():
        await worker.start()
        for _ in range(20):  # give the loop a moment to notice and exit
            if not worker.running:
                break
            await asyncio.sleep(0.1)
        return worker.running

    assert asyncio.run(scenario()) is False


def test_tick_counter_and_last_tick_are_reported(worker):
    async def scenario():
        await worker.start()
        await asyncio.sleep(0.7)
        await worker.stop()

    asyncio.run(scenario())
    assert worker.ticks >= 1
    assert worker.last_tick is not None
    assert worker.last_tick.released == 5
    assert worker.as_dict()["last_tick"]["predictions_written"] == 5


def test_a_failing_tick_is_recorded_not_fatal(worker):
    """A tick that errors must be surfaced in status, and must not kill the
    loop silently."""
    worker.tick_sync = lambda *a, **k: TickResult(  # type: ignore[method-assign]
        at=datetime.now(UTC),
        released=1,
        predictions_written=0,
        decisions_written=0,
        equipment_status_changes=0,
        service_requests_filed=0,
        error="OperationalError: boom",
    )

    async def scenario():
        await worker.start()
        await asyncio.sleep(0.4)
        running = worker.running
        await worker.stop()
        return running

    still_running = asyncio.run(scenario())
    assert still_running is True  # kept going
    assert worker.last_error == "OperationalError: boom"
