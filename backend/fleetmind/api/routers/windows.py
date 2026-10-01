"""Signal-window endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import WindowDetail, WindowSummary
from fleetmind.auth.deps import require_any_role
from fleetmind.db import session as db_session
from fleetmind.db.models import SignalWindow

# Any signed-in role may read the research/ML layer; nobody anonymous may.
router = APIRouter(tags=["windows"], dependencies=[Depends(require_any_role)])


@router.get("/experiments/{experiment_id}/windows", response_model=list[WindowSummary])
def list_windows_for_experiment(
    experiment_id: int,
    limit: int = Query(default=10, le=100),
    db: Session = Depends(db_session.get_session),
):
    query = (
        select(
            SignalWindow.id,
            SignalWindow.experiment_id,
            SignalWindow.window_index,
            SignalWindow.start_sample,
            SignalWindow.end_sample,
            SignalWindow.label,
        )
        .where(SignalWindow.experiment_id == experiment_id)
        .order_by(SignalWindow.window_index)
        .limit(limit)
    )
    return [WindowSummary(**row._mapping) for row in db.execute(query)]


@router.get("/signal-window/{window_id}", response_model=WindowDetail)
def get_signal_window(window_id: int, db: Session = Depends(db_session.get_session)):
    window = db.get(SignalWindow, window_id)
    if window is None:
        raise HTTPException(status_code=404, detail="window not found")
    return WindowDetail(
        id=window.id,
        experiment_id=window.experiment_id,
        window_index=window.window_index,
        start_sample=window.start_sample,
        end_sample=window.end_sample,
        label=window.label,
        raw_values=window.raw_values,
    )
