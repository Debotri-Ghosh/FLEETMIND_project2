"""Release/reset/status primitives for the replay engine.

The whole simulation is one column: ``signal_windows.released_at``.

* ``reset_replay``   -> set it back to NULL everywhere and clear everything
                        derived from it, so a demo can be re-run from zero.
* ``release_next_batch`` -> mark the next N windows as arrived, oldest first.
* ``replay_status``  -> how far through the archive the replay has got.

Nothing here runs the model; see fleetmind/pipeline.py for that. Keeping
release and processing separate is what lets the API expose "release a batch
and process it" (step) and "process whatever has arrived" independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update

from fleetmind.db.models import (
    AgentDecision,
    Equipment,
    ModelPrediction,
    ServiceRequest,
    SignalWindow,
    WorkOrder,
)


@dataclass
class ReplayStatus:
    total_windows: int
    released: int
    processed: int
    pending: int  # released but not yet predicted

    @property
    def complete(self) -> bool:
        return self.released >= self.total_windows and self.pending == 0

    @property
    def percent_released(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return round(100.0 * self.released / self.total_windows, 1)


def replay_status(session) -> ReplayStatus:
    total = session.execute(select(func.count()).select_from(SignalWindow)).scalar_one()
    released = session.execute(
        select(func.count()).select_from(SignalWindow).where(SignalWindow.released_at.is_not(None))
    ).scalar_one()
    processed = session.execute(
        select(func.count(func.distinct(ModelPrediction.window_id)))
        .select_from(ModelPrediction)
        .join(SignalWindow, SignalWindow.id == ModelPrediction.window_id)
        .where(SignalWindow.released_at.is_not(None))
    ).scalar_one()
    return ReplayStatus(
        total_windows=total,
        released=released,
        processed=processed,
        pending=max(released - processed, 0),
    )


def release_next_batch(session, batch_size: int = 25) -> int:
    """Mark the next batch of windows as arrived. Returns how many were released.

    Oldest-first by id, so a replay walks the archive in a stable order and a
    given batch size always produces the same sequence of arrivals.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    next_ids = (
        session.execute(
            select(SignalWindow.id)
            .where(SignalWindow.released_at.is_(None))
            .order_by(SignalWindow.id)
            .limit(batch_size)
        )
        .scalars()
        .all()
    )
    if not next_ids:
        return 0

    session.execute(
        update(SignalWindow)
        .where(SignalWindow.id.in_(next_ids))
        .values(released_at=datetime.now(UTC))
    )
    return len(next_ids)


def reset_replay(session) -> dict[str, int]:
    """Rewind to "nothing has arrived yet" so the demo can be run again.

    Deletes only data the pipeline itself produces and can regenerate:
    predictions, agent decisions, AI-filed service requests (and any work
    orders dispatched against them), and the equipment flags those caused.
    Customer-filed service requests are left untouched — a human wrote those,
    the replay didn't, so rewinding the simulation must not erase them.
    """
    ai_request_ids = (
        session.execute(
            select(ServiceRequest.id).where(ServiceRequest.agent_decision_id.is_not(None))
        )
        .scalars()
        .all()
    )

    work_orders_deleted = 0
    if ai_request_ids:
        work_orders_deleted = session.execute(
            delete(WorkOrder).where(WorkOrder.service_request_id.in_(ai_request_ids))
        ).rowcount
        session.execute(delete(ServiceRequest).where(ServiceRequest.id.in_(ai_request_ids)))

    decisions_deleted = session.execute(delete(AgentDecision)).rowcount
    predictions_deleted = session.execute(delete(ModelPrediction)).rowcount

    # Equipment flags are derived state: anything the pipeline flagged goes
    # back to active. 'retired' is a manual, permanent state and is preserved.
    equipment_reset = session.execute(
        update(Equipment).where(Equipment.status == "maintenance").values(status="active")
    ).rowcount

    windows_unreleased = session.execute(
        update(SignalWindow).where(SignalWindow.released_at.is_not(None)).values(released_at=None)
    ).rowcount

    return {
        "windows_unreleased": windows_unreleased,
        "predictions_deleted": predictions_deleted,
        "decisions_deleted": decisions_deleted,
        "ai_service_requests_deleted": len(ai_request_ids),
        "work_orders_deleted": work_orders_deleted,
        "equipment_reset_to_active": equipment_reset,
    }
