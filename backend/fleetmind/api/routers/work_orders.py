"""Work-order endpoints: technician dispatch tracking for a service request.

location_lat/lng are simulated — CWRU has no real GPS telemetry (see CLAUDE.md).
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import insert, select, update
from sqlalchemy.orm import Session

from fleetmind.api.schemas import WorkOrder as WorkOrderSchema
from fleetmind.api.schemas import WorkOrderCreate, WorkOrderStatusUpdate
from fleetmind.db import session as db_session
from fleetmind.db.models import ServiceRequest, WorkOrder

router = APIRouter(prefix="/work-orders", tags=["work-orders"])


@router.get("", response_model=list[WorkOrderSchema])
def list_work_orders(
    technician_id: int | None = None,
    status: str | None = None,
    service_request_id: int | None = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(db_session.get_session),
):
    query = select(WorkOrder).order_by(WorkOrder.id.desc()).limit(limit)
    if technician_id is not None:
        query = query.where(WorkOrder.technician_id == technician_id)
    if status is not None:
        query = query.where(WorkOrder.status == status)
    if service_request_id is not None:
        query = query.where(WorkOrder.service_request_id == service_request_id)
    return list(db.scalars(query))


@router.get("/{work_order_id}", response_model=WorkOrderSchema)
def get_work_order(work_order_id: int, db: Session = Depends(db_session.get_session)):
    work_order = db.get(WorkOrder, work_order_id)
    if work_order is None:
        raise HTTPException(status_code=404, detail="work order not found")
    return work_order


@router.post("", response_model=WorkOrderSchema, status_code=201)
def create_work_order(payload: WorkOrderCreate, db: Session = Depends(db_session.get_session)):
    service_request = db.get(ServiceRequest, payload.service_request_id)
    if service_request is None:
        raise HTTPException(status_code=404, detail="service request not found")

    result = db.execute(insert(WorkOrder).values(**payload.model_dump()))
    db.execute(
        update(ServiceRequest)
        .where(ServiceRequest.id == payload.service_request_id)
        .values(status="dispatched")
    )
    db.commit()
    return db.get(WorkOrder, result.inserted_primary_key[0])


@router.patch("/{work_order_id}/status", response_model=WorkOrderSchema)
def update_work_order_status(
    work_order_id: int,
    payload: WorkOrderStatusUpdate,
    db: Session = Depends(db_session.get_session),
):
    work_order = db.get(WorkOrder, work_order_id)
    if work_order is None:
        raise HTTPException(status_code=404, detail="work order not found")

    values: dict = {"status": payload.status}
    now = datetime.now(UTC)
    if payload.status == "in_progress" and work_order.started_at is None:
        values["started_at"] = now
    if payload.status == "completed" and work_order.completed_at is None:
        values["completed_at"] = now

    db.execute(update(WorkOrder).where(WorkOrder.id == work_order_id).values(**values))

    request_status = {
        "traveling": "dispatched",
        "in_progress": "in_progress",
        "completed": "resolved",
        "cancelled": "cancelled",
    }.get(payload.status)
    if request_status is not None:
        db.execute(
            update(ServiceRequest)
            .where(ServiceRequest.id == work_order.service_request_id)
            .values(status=request_status)
        )

    db.commit()
    return db.get(WorkOrder, work_order_id)
