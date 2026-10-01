"""Customer endpoints. Read-only — seeded via db/seed_product_tables.py."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import Customer as CustomerSchema
from fleetmind.auth.deps import ROLE_CUSTOMER, get_current_user, may_read_customer
from fleetmind.db import session as db_session
from fleetmind.db.models import Customer as CustomerModel
from fleetmind.db.models import User

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("", response_model=list[CustomerSchema])
def list_customers(
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    """Operators and admins see the roster; a customer sees only itself.

    Returning a one-element list rather than 403 keeps the frontend's
    "viewing as" control working unchanged for every role, while still making
    it impossible for a customer to learn who the other customers are.
    """
    query = select(CustomerModel).order_by(CustomerModel.id)
    if user.role == ROLE_CUSTOMER:
        query = query.where(CustomerModel.id == user.customer_id)
    return list(db.scalars(query))


@router.get("/{customer_id}", response_model=CustomerSchema)
def get_customer(
    customer_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    customer = db.get(CustomerModel, customer_id)
    # 404 rather than 403 for someone else's customer: a 403 would confirm the
    # id exists and let a customer enumerate the others.
    if customer is None or not may_read_customer(user, customer_id):
        raise HTTPException(status_code=404, detail="customer not found")
    return customer
