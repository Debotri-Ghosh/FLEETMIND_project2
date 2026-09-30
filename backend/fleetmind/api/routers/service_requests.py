"""Service-request endpoints: customers file requests, depot operators triage them."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from fleetmind.agents.agents import diagnostic_agent
from fleetmind.api.schemas import EvidencePackage, MatchedFrequency, ServiceRequestCreate
from fleetmind.api.schemas import ServiceRequest as ServiceRequestSchema
from fleetmind.db import session as db_session
from fleetmind.db.models import (
    AgentDecision,
    Customer,
    Depot,
    Equipment,
    Experiment,
    ModelPrediction,
    SignalWindow,
)
from fleetmind.db.models import ServiceRequest as ServiceRequestModel
from fleetmind.ml.bearing_freq import (
    characteristic_frequencies,
    matched_frequency_for_fault,
    spectrum,
)

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
    limit: int = Query(default=100, ge=1, le=500),
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


@router.get("/{request_id}/evidence", response_model=EvidencePackage)
def get_evidence(request_id: int, db: Session = Depends(db_session.get_session)):
    """Inspectable evidence behind an AI-filed request.

    Walks the chain the decision was actually made through:
      ServiceRequest -> AgentDecision -> ModelPrediction -> SignalWindow
                     -> Experiment -> Depot (for shaft speed)

    The diagnostic stats are *recomputed* here with the same
    `agents.agents._signal_stats` / `diagnostic_agent` functions that produced
    the decision, because only `decision_text` and `risk_score` are persisted.
    Same function, same input, same numbers -- but it is a recomputation, not
    a read-back, and is labelled as such in the UI.
    """
    row = db.execute(
        select(
            ServiceRequestModel,
            AgentDecision,
            ModelPrediction,
            SignalWindow,
            Experiment,
            Depot,
        )
        .join(AgentDecision, AgentDecision.id == ServiceRequestModel.agent_decision_id)
        .join(ModelPrediction, ModelPrediction.id == AgentDecision.prediction_id)
        .join(SignalWindow, SignalWindow.id == ModelPrediction.window_id)
        .join(Experiment, Experiment.id == SignalWindow.experiment_id)
        .join(Depot, Depot.id == Experiment.depot_id)
        .where(ServiceRequestModel.id == request_id)
    ).first()

    if row is None:
        # Distinguish "no such request" from "request has no AI evidence", so
        # the UI can say which -- a customer-filed request legitimately has
        # nothing to show here and that is not an error condition.
        exists = db.get(ServiceRequestModel, request_id)
        if exists is None:
            raise HTTPException(status_code=404, detail=f"service request {request_id} not found")
        raise HTTPException(
            status_code=404,
            detail=(
                f"service request {request_id} has no agent decision attached "
                "(customer-filed requests carry no model evidence)"
            ),
        )

    service_request, decision, prediction, window, experiment, depot = row
    values = [float(v) for v in (window.raw_values or [])]

    diagnostic = diagnostic_agent(
        {
            "predicted_label": prediction.predicted_label,
            "confidence": prediction.confidence,
            "signal_values": values,
        }
    )
    freqs, mags = spectrum(values, experiment.sampling_rate_hz)
    matched = matched_frequency_for_fault(
        prediction.predicted_label,
        depot.nominal_rpm or 1797,
        experiment.sensor_position,
    )

    return EvidencePackage(
        service_request_id=service_request.id,
        window_id=window.id,
        equipment_id=service_request.equipment_id,
        site_name=depot.name,
        predicted_label=prediction.predicted_label,
        model_confidence=prediction.confidence,
        model_version=prediction.model_version,
        raw_values=values,
        sampling_rate_hz=experiment.sampling_rate_hz,
        spectrum_freqs_hz=freqs,
        spectrum_magnitudes=mags,
        matched_frequency=MatchedFrequency(**vars(matched)) if matched else None,
        all_characteristic_frequencies=characteristic_frequencies(
            depot.nominal_rpm or 1797, experiment.sensor_position
        ),
        signal_rms=diagnostic["signal_rms"],
        signal_peak=diagnostic["signal_peak"],
        diagnosis=diagnostic["diagnosis"],
        diagnosis_confidence=diagnostic["diagnosis_confidence"],
        risk_score=decision.risk_score,
        decision_text=decision.decision_text,
    )


@router.post("", response_model=ServiceRequestSchema, status_code=201)
def create_service_request(
    payload: ServiceRequestCreate, db: Session = Depends(db_session.get_session)
):
    if db.get(Customer, payload.customer_id) is None:
        raise HTTPException(status_code=404, detail=f"customer {payload.customer_id} not found")
    if db.get(Depot, payload.depot_id) is None:
        raise HTTPException(status_code=404, detail=f"depot {payload.depot_id} not found")
    if payload.equipment_id is not None:
        equipment = db.get(Equipment, payload.equipment_id)
        if equipment is None:
            raise HTTPException(
                status_code=404, detail=f"equipment {payload.equipment_id} not found"
            )
        # Every piece of equipment belongs to exactly one depot. A request that
        # names an equipment_id and a depot_id that disagree is inconsistent
        # client data, not merely "not found" -- we reject it (422) rather than
        # silently trusting whichever field or silently overwriting depot_id
        # with the equipment's real depot, since either would hide a client bug.
        if equipment.depot_id != payload.depot_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"equipment {payload.equipment_id} belongs to depot "
                    f"{equipment.depot_id}, not depot {payload.depot_id}"
                ),
            )

    result = db.execute(insert(ServiceRequestModel).values(**payload.model_dump()))
    db.commit()
    request = db.get(ServiceRequestModel, result.inserted_primary_key[0])
    return request
