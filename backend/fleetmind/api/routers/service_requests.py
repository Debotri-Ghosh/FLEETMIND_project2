"""Service-request endpoints: customers file requests, depot operators triage them."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import ServiceRequest as ServiceRequestSchema
from fleetmind.api.schemas import ServiceRequestCreate
from fleetmind.db import session as db_session
from fleetmind.db.models import ServiceRequest as ServiceRequestModel

router = APIRouter(prefix="/service-requests", tags=["service-requests"])


@router.get("", response_model=list[ServiceRequestSchema])
def list_service_requests(
    customer_id: int | None = None,
    depot_id: int | None = None,
    equipment_id: int | None = None,
    status: str | None = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(db_session.get_session),
):
    query = select(ServiceRequestModel).order_by(ServiceRequestModel.id.desc()).limit(limit)
    if customer_id is not None:
        query = query.where(ServiceRequestModel.customer_id == customer_id)
    if depot_id is not None:
        query = query.where(ServiceRequestModel.depot_id == depot_id)
    if equipment_id is not None:
        query = query.where(ServiceRequestModel.equipment_id == equipment_id)
    if status is not None:
        query = query.where(ServiceRequestModel.status == status)
    return list(db.scalars(query))


@router.get("/{request_id}", response_model=ServiceRequestSchema)
def get_service_request(request_id: int, db: Session = Depends(db_session.get_session)):
    request = db.get(ServiceRequestModel, request_id)
    if request is None:
        raise HTTPException(status_code=404, detail="service request not found")
    return request


@router.post("", response_model=ServiceRequestSchema, status_code=201)
def create_service_request(
    payload: ServiceRequestCreate, db: Session = Depends(db_session.get_session)
):
    result = db.execute(insert(ServiceRequestModel).values(**payload.model_dump()))
    db.commit()
    request = db.get(ServiceRequestModel, result.inserted_primary_key[0])
    return request
