"""Replay controls: drive the simulated sensor feed from the operator console.

This is demo/simulation tooling, not a production ingest API — the underlying
data is an archived CWRU dataset being replayed, and the UI says so.

`start`/`stop` drive the background worker; `step` runs exactly one tick
synchronously so a presenter can advance the simulation one batch at a time
and talk over it; `reset` rewinds so the whole demo can be run again.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from fleetmind.api.schemas import (
    ReplayResetResponse,
    ReplayStartRequest,
    ReplayStatusResponse,
    ReplayTickResponse,
)
from fleetmind.auth.deps import require_admin
from fleetmind.db import session as db_session
from fleetmind.replay.engine import replay_status, reset_replay
from fleetmind.replay.worker import worker

# Replay mutates the whole dataset -- predictions, decisions, equipment
# status. Admin only; an operator must not be able to reset the archive.
router = APIRouter(prefix="/replay", tags=["replay"], dependencies=[Depends(require_admin)])


@router.get("/status", response_model=ReplayStatusResponse)
def get_status(db: Session = Depends(db_session.get_session)):
    status = replay_status(db)
    return ReplayStatusResponse(
        total_windows=status.total_windows,
        released=status.released,
        processed=status.processed,
        pending=status.pending,
        percent_released=status.percent_released,
        complete=status.complete,
        worker=worker.as_dict(),
    )


@router.post("/start", response_model=ReplayStatusResponse)
async def start_replay(
    payload: ReplayStartRequest | None = None,
    db: Session = Depends(db_session.get_session),
):
    payload = payload or ReplayStartRequest()
    await worker.start(
        interval_seconds=payload.interval_seconds,
        batch_size=payload.batch_size,
    )
    status = replay_status(db)
    return ReplayStatusResponse(
        total_windows=status.total_windows,
        released=status.released,
        processed=status.processed,
        pending=status.pending,
        percent_released=status.percent_released,
        complete=status.complete,
        worker=worker.as_dict(),
    )


@router.post("/stop", response_model=ReplayStatusResponse)
async def stop_replay(db: Session = Depends(db_session.get_session)):
    await worker.stop()
    status = replay_status(db)
    return ReplayStatusResponse(
        total_windows=status.total_windows,
        released=status.released,
        processed=status.processed,
        pending=status.pending,
        percent_released=status.percent_released,
        complete=status.complete,
        worker=worker.as_dict(),
    )


@router.post("/step", response_model=ReplayTickResponse)
async def step_replay():
    """Advance exactly one batch, synchronously, and report what happened.

    Refused while the worker is running: otherwise two writers would race over
    the same windows and the counts reported here would be meaningless.
    """
    if worker.running:
        raise HTTPException(
            status_code=409,
            detail="Replay worker is running; stop it before stepping manually.",
        )
    result = worker.tick_sync()
    return ReplayTickResponse(
        released=result.released,
        predictions_written=result.predictions_written,
        decisions_written=result.decisions_written,
        equipment_status_changes=result.equipment_status_changes,
        service_requests_filed=result.service_requests_filed,
        error=result.error,
    )


@router.post("/reset", response_model=ReplayResetResponse)
async def reset(db: Session = Depends(db_session.get_session)):
    """Rewind the simulation to "nothing has arrived yet".

    Stops the worker first — resetting underneath a running loop would let it
    immediately re-release windows mid-wipe. Deletes only pipeline-produced
    data; customer-filed requests and retired equipment are preserved.
    """
    await worker.stop()
    summary = reset_replay(db)
    db.commit()
    return ReplayResetResponse(**summary)
