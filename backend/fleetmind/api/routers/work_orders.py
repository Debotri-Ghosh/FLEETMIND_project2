"""Work-order endpoints: technician dispatch tracking for a service request.

location_lat/lng are simulated — CWRU has no real GPS telemetry (see CLAUDE.md).
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import insert, select, update
from sqlalchemy.orm import Session

from fleetmind.api.schemas import WorkOrder as WorkOrderSchema
from fleetmind.api.schemas import WorkOrderCreate, WorkOrderStatusUpdate
from fleetmind.auth.deps import get_current_user, require_operator, scope_by_depot
from fleetmind.db import session as db_session
from fleetmind.db.models import ServiceRequest, Technician, User, WorkOrder

# Dispatch is an operator function end to end. A customer may see that their
# request was dispatched (via /service-requests) but has no business reading
# or writing the technician assignment behind it.
router = APIRouter(
    prefix="/work-orders",
    tags=["work-orders"],
    dependencies=[Depends(require_operator)],
)


def _scoped_to_callers_site(query, user: User):
    """Restrict work orders to those whose service request is at the caller's site.

    Work orders carry no depot_id of their own, so the scope has to be reached
    through the service request -- which is also the only place it is true.
    """
    condition = scope_by_depot(user, ServiceRequest.depot_id)
    if condition is None:
        return query
    return query.join(ServiceRequest, ServiceRequest.id == WorkOrder.service_request_id).where(
        condition
    )


@router.get("", response_model=list[WorkOrderSchema])
def list_work_orders(
    technician_id: int | None = None,
    status: str | None = None,
    service_request_id: int | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    query = _scoped_to_callers_site(
        select(WorkOrder).order_by(WorkOrder.id.desc()).limit(limit), user
    )
    if technician_id is not None:
        query = query.where(WorkOrder.technician_id == technician_id)
    if status is not None:
        query = query.where(WorkOrder.status == status)
    if service_request_id is not None:
        query = query.where(WorkOrder.service_request_id == service_request_id)
    return list(db.scalars(query))


@router.get("/{work_order_id}", response_model=WorkOrderSchema)
def get_work_order(
    work_order_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    work_order = db.get(WorkOrder, work_order_id)
    if work_order is None:
        raise HTTPException(status_code=404, detail="work order not found")
    _assert_same_site(db, user, work_order.service_request_id)
    return work_order


def _assert_same_site(db: Session, user: User, service_request_id: int) -> None:
    """404 for a work order at another site -- see auth/deps.assert_may_read."""
    request = db.get(ServiceRequest, service_request_id)
    if request is None:
        raise HTTPException(status_code=404, detail="work order not found")
    if scope_by_depot(user, ServiceRequest.depot_id) is not None and (
        request.depot_id != user.depot_id
    ):
        raise HTTPException(status_code=404, detail="work order not found")


@router.post("", response_model=WorkOrderSchema, status_code=201)
def create_work_order(
    payload: WorkOrderCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    service_request = db.get(ServiceRequest, payload.service_request_id)
    if service_request is None:
        raise HTTPException(status_code=404, detail="service request not found")
    # An operator may only dispatch against their own site's queue.
    _assert_same_site(db, user, payload.service_request_id)
    if db.get(Technician, payload.technician_id) is None:
        raise HTTPException(status_code=404, detail=f"technician {payload.technician_id} not found")

    # service_request_id has a unique constraint (one work order per request).
    # Check for an existing work order before the insert so a double-dispatch
    # surfaces as a clean 409 instead of an unhandled IntegrityError -> 500.
    existing = db.scalar(
        select(WorkOrder).where(WorkOrder.service_request_id == payload.service_request_id)
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"service request {payload.service_request_id} already has work order {existing.id}"
            ),
        )

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
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    # No transition-graph validation here: any status -> any other status is
    # allowed, including "backwards" moves (completed -> assigned) and setting
    # the same status twice. Reasoned default: field ops need to correct a
    # mis-click or a bad dispatch without filing a support ticket, and there's
    # no safety-critical invariant a backwards transition could violate here.
    # This is a conscious choice, not an oversight -- see
    # test_work_order_status_allows_any_transition for the behavior this locks in.
    work_order = db.get(WorkOrder, work_order_id)
    if work_order is None:
        raise HTTPException(status_code=404, detail="work order not found")
    _assert_same_site(db, user, work_order.service_request_id)

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
