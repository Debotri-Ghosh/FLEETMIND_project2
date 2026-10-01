"""Equipment endpoints. Read-only — rows are seeded and their status is kept
in sync by fleetmind.agents.sync_equipment_status, not written via the API."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import Equipment as EquipmentSchema
from fleetmind.auth.deps import assert_may_read, get_current_user, scope_by_customer, scope_by_depot
from fleetmind.db import session as db_session
from fleetmind.db.models import Equipment as EquipmentModel
from fleetmind.db.models import User

router = APIRouter(prefix="/equipment", tags=["equipment"])


@router.get("", response_model=list[EquipmentSchema])
def list_equipment(
    depot_id: int | None = None,
    customer_id: int | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    query = select(EquipmentModel).order_by(EquipmentModel.id).limit(limit)

    # The caller's own scope is applied FIRST and comes from their user row,
    # never from the query parameters below. A customer passing
    # ?customer_id=2 therefore narrows their own result set and can never
    # widen it; same for an operator and ?depot_id.
    for condition in (
        scope_by_customer(user, EquipmentModel.customer_id),
        scope_by_depot(user, EquipmentModel.depot_id),
    ):
        if condition is not None:
            query = query.where(condition)

    if depot_id is not None:
        query = query.where(EquipmentModel.depot_id == depot_id)
    if customer_id is not None:
        query = query.where(EquipmentModel.customer_id == customer_id)
    if status is not None:
        query = query.where(EquipmentModel.status == status)
    return list(db.scalars(query))


@router.get("/{equipment_id}", response_model=EquipmentSchema)
def get_equipment(
    equipment_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    equipment = db.get(EquipmentModel, equipment_id)
    if equipment is None:
        raise HTTPException(status_code=404, detail="equipment not found")
    assert_may_read(user, customer_id=equipment.customer_id, depot_id=equipment.depot_id)
    return equipment
