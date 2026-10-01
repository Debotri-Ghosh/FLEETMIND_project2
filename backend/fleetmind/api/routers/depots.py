"""Depot endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import Depot as DepotSchema
from fleetmind.auth.deps import require_any_role
from fleetmind.db import session as db_session
from fleetmind.db.models import Depot as DepotModel
from fleetmind.db.models import Experiment

# Any signed-in role may read the research/ML layer; nobody anonymous may.
router = APIRouter(prefix="/depots", tags=["depots"], dependencies=[Depends(require_any_role)])


@router.get("", response_model=list[DepotSchema])
def list_depots(db: Session = Depends(db_session.get_session)):
    n_exp = (
        select(func.count(Experiment.id))
        .where(Experiment.depot_id == DepotModel.id)
        .correlate(DepotModel)
        .scalar_subquery()
    )
    rows = db.execute(
        select(DepotModel, n_exp.label("n_experiments")).order_by(DepotModel.load_hp)
    ).all()
    return [
        DepotSchema(
            id=d.id,
            name=d.name,
            load_hp=d.load_hp,
            nominal_rpm=d.nominal_rpm,
            n_experiments=count or 0,
        )
        for d, count in rows
    ]
