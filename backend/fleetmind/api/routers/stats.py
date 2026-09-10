"""Aggregate statistics endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import FaultDistributionEntry
from fleetmind.db import session as db_session
from fleetmind.db.models import SignalWindow

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/fault-distribution", response_model=list[FaultDistributionEntry])
def fault_distribution(db: Session = Depends(db_session.get_session)):
    rows = db.execute(
        select(SignalWindow.label, func.count().label("n_windows"))
        .group_by(SignalWindow.label)
        .order_by(func.count().desc())
    )
    return [FaultDistributionEntry(label=label, n_windows=n) for label, n in rows]
