"""Equipment endpoints. Read-only — rows are seeded and their status is kept
in sync by fleetmind.agents.sync_equipment_status, not written via the API."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import Equipment as EquipmentSchema
from fleetmind.db import session as db_session
from fleetmind.db.models import Equipment as EquipmentModel

router = APIRouter(prefix="/equipment", tags=["equipment"])


@router.get("", response_model=list[EquipmentSchema])
def list_equipment(
    depot_id: int | None = None,
    customer_id: int | None = None,
    status: str | None = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(db_session.get_session),
):
    query = select(EquipmentModel).order_by(EquipmentModel.id).limit(limit)
    if depot_id is not None:
        query = query.where(EquipmentModel.depot_id == depot_id)
    if customer_id is not None:
        query = query.where(EquipmentModel.customer_id == customer_id)
    if status is not None:
        query = query.where(EquipmentModel.status == status)
    return list(db.scalars(query))


@router.get("/{equipment_id}", response_model=EquipmentSchema)
def get_equipment(equipment_id: int, db: Session = Depends(db_session.get_session)):
    equipment = db.get(EquipmentModel, equipment_id)
    if equipment is None:
        raise HTTPException(status_code=404, detail="equipment not found")
    return equipment
