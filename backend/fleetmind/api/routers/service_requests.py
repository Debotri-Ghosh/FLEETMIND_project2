"""Service-request endpoints: customers file requests, depot operators triage them."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import ServiceRequest as ServiceRequestSchema
from fleetmind.api.schemas import ServiceRequestCreate
from fleetmind.db import session as db_session
from fleetmind.db.models import AgentDecision, ModelPrediction
from fleetmind.db.models import ServiceRequest as ServiceRequestModel

router = APIRouter(prefix="/service-requests", tags=["service-requests"])


def _with_agent_context(row) -> ServiceRequestSchema:
    """row is a (ServiceRequestModel, decision_text, risk_score, predicted_label)
    Row from a query outer-joined through agent_decision_id -> prediction_id.
    The join columns are None for customer-filed requests."""
    sr, decision_text, risk_score, predicted_label = row
    return ServiceRequestSchema.model_validate(sr).model_copy(
        update={
            "agent_decision_text": decision_text,
            "agent_risk_score": risk_score,
            "predicted_fault": predicted_label,
        }
    )


def _select_with_agent_context():
    return (
        select(
            ServiceRequestModel,
            AgentDecision.decision_text,
            AgentDecision.risk_score,
            ModelPrediction.predicted_label,
        )
        .outerjoin(AgentDecision, AgentDecision.id == ServiceRequestModel.agent_decision_id)
        .outerjoin(ModelPrediction, ModelPrediction.id == AgentDecision.prediction_id)
    )


@router.get("", response_model=list[ServiceRequestSchema])
def list_service_requests(
    customer_id: int | None = None,
    depot_id: int | None = None,
    equipment_id: int | None = None,
    status: str | None = None,
    ai_flagged: bool | None = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(db_session.get_session),
):
    query = _select_with_agent_context().order_by(ServiceRequestModel.id.desc()).limit(limit)
    if customer_id is not None:
        query = query.where(ServiceRequestModel.customer_id == customer_id)
    if depot_id is not None:
        query = query.where(ServiceRequestModel.depot_id == depot_id)
    if equipment_id is not None:
        query = query.where(ServiceRequestModel.equipment_id == equipment_id)
    if status is not None:
        query = query.where(ServiceRequestModel.status == status)
    if ai_flagged is True:
        query = query.where(ServiceRequestModel.agent_decision_id.is_not(None))
    elif ai_flagged is False:
        query = query.where(ServiceRequestModel.agent_decision_id.is_(None))
    return [_with_agent_context(row) for row in db.execute(query).all()]


@router.get("/{request_id}", response_model=ServiceRequestSchema)
def get_service_request(request_id: int, db: Session = Depends(db_session.get_session)):
    row = db.execute(
        _select_with_agent_context().where(ServiceRequestModel.id == request_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="service request not found")
    return _with_agent_context(row)


@router.post("", response_model=ServiceRequestSchema, status_code=201)
def create_service_request(
    payload: ServiceRequestCreate, db: Session = Depends(db_session.get_session)
):
    result = db.execute(insert(ServiceRequestModel).values(**payload.model_dump()))
    db.commit()
    request = db.get(ServiceRequestModel, result.inserted_primary_key[0])
    return request
