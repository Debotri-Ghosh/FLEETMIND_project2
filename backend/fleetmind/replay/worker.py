"""Background replay worker: ticks the simulated sensor feed.

Runs as an asyncio task inside the API process, so a demo needs one backend
terminal rather than a separate worker process. Each tick:

    release_next_batch(batch_size)  ->  process_arrived_windows()

The pipeline step is CPU-bound (torch inference), so it is offloaded to a
worker thread — running it directly on the event loop would freeze every
other API request for the duration of the batch, which would make the very
dashboard we are trying to keep live stop responding.

State lives in one module-level singleton because there is exactly one replay
per backend process; start/stop take a lock so double-clicking "Start" can't
spawn two loops racing over the same windows.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from anyio import to_thread

from fleetmind.core.config import get_settings
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.pipeline import process_arrived_windows
from fleetmind.replay.engine import release_next_batch

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_BATCH_SIZE = 25
MIN_INTERVAL_SECONDS = 0.25
MAX_INTERVAL_SECONDS = 60.0
MAX_BATCH_SIZE = 500


@dataclass
class TickResult:
    at: datetime
    released: int
    predictions_written: int
    decisions_written: int
    equipment_status_changes: int
    service_requests_filed: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "released": self.released,
            "predictions_written": self.predictions_written,
            "decisions_written": self.decisions_written,
            "equipment_status_changes": self.equipment_status_changes,
            "service_requests_filed": self.service_requests_filed,
            "error": self.error,
        }


@dataclass
class ReplayWorker:
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE
    running: bool = False
    ticks: int = 0
    last_tick: TickResult | None = None
    last_error: str | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # -- session handling -------------------------------------------------

    def _new_session(self):
        """The worker owns its own session: request-scoped ones are closed
        the moment the triggering HTTP call returns."""
        engine = create_db_engine(get_settings().database_url)
        return make_session_factory(engine)()

    # -- one tick ---------------------------------------------------------

    def tick_sync(self, release: bool = True) -> TickResult:
        """Release a batch (optionally) and process everything that has arrived.

        Synchronous and self-contained so it can run either in the worker
        thread or directly from a request handler (the manual "step" button).
        """
        session = self._new_session()
        try:
            released = release_next_batch(session, self.batch_size) if release else 0
            if released:
                session.commit()
            counts = process_arrived_windows(session)
            return TickResult(
                at=datetime.now(UTC),
                released=released,
                predictions_written=counts["predictions_written"],
                decisions_written=counts["decisions_written"],
                equipment_status_changes=counts["equipment_status_changes"],
                service_requests_filed=counts["service_requests_filed"],
            )
        except Exception as exc:  # keep the loop alive; surface it in status
            session.rollback()
            logger.exception("Replay tick failed")
            return TickResult(
                at=datetime.now(UTC),
                released=0,
                predictions_written=0,
                decisions_written=0,
                equipment_status_changes=0,
                service_requests_filed=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            session.close()

    # -- the loop ---------------------------------------------------------

    async def _run(self) -> None:
        try:
            while self.running:
                result = await to_thread.run_sync(self.tick_sync)
                self.ticks += 1
                self.last_tick = result
                self.last_error = result.error
                # Nothing released and nothing processed => archive exhausted.
                if (
                    result.error is None
                    and result.released == 0
                    and result.predictions_written == 0
                ):
                    logger.info("Replay archive exhausted after %s ticks; stopping.", self.ticks)
                    self.running = False
                    break
                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            raise
        finally:
            self.running = False

    # -- controls ---------------------------------------------------------

    async def start(self, interval_seconds: float | None = None, batch_size: int | None = None):
        async with self._lock:
            if interval_seconds is not None:
                self.interval_seconds = max(
                    MIN_INTERVAL_SECONDS, min(float(interval_seconds), MAX_INTERVAL_SECONDS)
                )
            if batch_size is not None:
                self.batch_size = max(1, min(int(batch_size), MAX_BATCH_SIZE))
            if self.running:
                return False  # already running; not an error, just a no-op
            self.running = True
            self.ticks = 0
            self.last_error = None
            self._task = asyncio.create_task(self._run())
            return True

    async def stop(self) -> bool:
        async with self._lock:
            if not self.running and self._task is None:
                return False
            self.running = False
            task = self._task
            self._task = None
        if task is not None:
            task.cancel()
            # We are shutting the loop down on purpose; whatever it raises on
            # the way out (cancellation, or a tick that was mid-failure) is
            # not actionable here.
            with contextlib.suppress(Exception):
                await task
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "interval_seconds": self.interval_seconds,
            "batch_size": self.batch_size,
            "ticks": self.ticks,
            "last_tick": self.last_tick.as_dict() if self.last_tick else None,
            "last_error": self.last_error,
        }


worker = ReplayWorker()
