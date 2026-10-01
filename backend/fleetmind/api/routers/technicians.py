"""Technician endpoints. Read-only — seeded via db/seed_product_tables.py."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import Technician as TechnicianSchema
from fleetmind.auth.deps import require_operator
from fleetmind.db import session as db_session
from fleetmind.db.models import Technician as TechnicianModel

# Technician roster is dispatch data: operators and admins only.
router = APIRouter(
    prefix="/technicians", tags=["technicians"], dependencies=[Depends(require_operator)]
)


@router.get("", response_model=list[TechnicianSchema])
def list_technicians(status: str | None = None, db: Session = Depends(db_session.get_session)):
    query = select(TechnicianModel).order_by(TechnicianModel.id)
    if status is not None:
        query = query.where(TechnicianModel.status == status)
    return list(db.scalars(query))


@router.get("/{technician_id}", response_model=TechnicianSchema)
def get_technician(technician_id: int, db: Session = Depends(db_session.get_session)):
    technician = db.get(TechnicianModel, technician_id)
    if technician is None:
        raise HTTPException(status_code=404, detail="technician not found")
    return technician
