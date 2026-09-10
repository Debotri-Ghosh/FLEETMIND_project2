"""Experiment endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import Experiment as ExperimentSchema
from fleetmind.db import session as db_session
from fleetmind.db.models import Experiment as ExperimentModel

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentSchema])
def list_experiments(
    depot_id: int | None = None,
    fault_type: str | None = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(db_session.get_session),
):
    query = select(ExperimentModel).order_by(ExperimentModel.id).limit(limit)
    if depot_id is not None:
        query = query.where(ExperimentModel.depot_id == depot_id)
    if fault_type is not None:
        query = query.where(ExperimentModel.fault_type == fault_type)
    return list(db.scalars(query))


@router.get("/{experiment_id}", response_model=ExperimentSchema)
def get_experiment(experiment_id: int, db: Session = Depends(db_session.get_session)):
    experiment = db.get(ExperimentModel, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return experiment
