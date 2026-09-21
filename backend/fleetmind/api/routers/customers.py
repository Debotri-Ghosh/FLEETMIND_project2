"""Customer endpoints. Read-only — seeded via db/seed_product_tables.py."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import Customer as CustomerSchema
from fleetmind.db import session as db_session
from fleetmind.db.models import Customer as CustomerModel

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("", response_model=list[CustomerSchema])
def list_customers(db: Session = Depends(db_session.get_session)):
    return list(db.scalars(select(CustomerModel).order_by(CustomerModel.id)))


@router.get("/{customer_id}", response_model=CustomerSchema)
def get_customer(customer_id: int, db: Session = Depends(db_session.get_session)):
    customer = db.get(CustomerModel, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="customer not found")
    return customer
