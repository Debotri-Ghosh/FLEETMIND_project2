"""Service-request endpoints: customers file requests, depot operators triage them."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from fleetmind.agents.agents import diagnostic_agent
from fleetmind.agents.llm import LLMNotConfigured, generate_narrative
from fleetmind.agents.tools import EvidenceContext
from fleetmind.api.schemas import (
    EvidencePackage,
    MatchedFrequency,
    ServiceRequestCreate,
    ServiceRequestNarrative,
)
from fleetmind.api.schemas import ServiceRequest as ServiceRequestSchema
from fleetmind.db import session as db_session
from fleetmind.db.models import (
    AgentDecision,
    Customer,
    Depot,
    Equipment,
    Experiment,
    LLMNarrative,
    ModelPrediction,
    SignalWindow,
)
from fleetmind.db.models import ServiceRequest as ServiceRequestModel
from fleetmind.ml.bearing_freq import (
    characteristic_frequencies,
    matched_frequency_for_fault,
    spectrum,
)

logger = logging.getLogger(__name__)

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


def _evidence_chain(db: Session, request_id: int):
    """Walk ServiceRequest -> AgentDecision -> Prediction -> Window -> Experiment -> Depot.

    Shared by the evidence package and the LLM narrative so both are
    guaranteed to describe the same window; a mismatch between the plots a
    reviewer looks at and the numbers the narrative quotes would be worse
    than having neither.
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
    return row


def _evidence_context(db: Session, request_id: int) -> EvidenceContext:
    """The same chain, packaged for the LLM tools."""
    service_request, decision, prediction, window, experiment, depot = _evidence_chain(
        db, request_id
    )
    equipment = (
        db.get(Equipment, service_request.equipment_id)
        if service_request.equipment_id is not None
        else None
    )
    return EvidenceContext(
        service_request_id=service_request.id,
        window_id=window.id,
        equipment_name=equipment.name if equipment else None,
        site_name=depot.name,
        depot_id=depot.id,
        load_hp=depot.load_hp,
        nominal_rpm=float(depot.nominal_rpm or 1797),
        sensor_position=experiment.sensor_position,
        sampling_rate_hz=experiment.sampling_rate_hz,
        predicted_label=prediction.predicted_label,
        model_confidence=prediction.confidence,
        model_version=prediction.model_version,
        raw_values=[float(v) for v in (window.raw_values or [])],
        stored_risk_score=decision.risk_score,
        stored_decision_text=decision.decision_text,
    )


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
    service_request, decision, prediction, window, experiment, depot = _evidence_chain(
        db, request_id
    )
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


def _narrative_response(row: LLMNarrative, cached: bool) -> ServiceRequestNarrative:
    return ServiceRequestNarrative.model_validate(row).model_copy(update={"cached": cached})


@router.get("/{request_id}/narrative", response_model=ServiceRequestNarrative)
def get_narrative(request_id: int, db: Session = Depends(db_session.get_session)):
    """Return the cached narrative, if one has been generated.

    Read-only and free: a 404 here means "not generated yet", not an error.
    Generation costs a real API call and is therefore an explicit POST.
    """
    row = db.execute(
        select(LLMNarrative).where(LLMNarrative.service_request_id == request_id)
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"no narrative generated yet for service request {request_id}",
        )
    return _narrative_response(row, cached=True)


@router.post("/{request_id}/narrative", response_model=ServiceRequestNarrative)
def create_narrative(
    request_id: int,
    force: bool = Query(
        default=False,
        description="Regenerate even if a narrative is cached. Costs another API call.",
    ),
    db: Session = Depends(db_session.get_session),
):
    """Generate (or return the cached) Claude narrative for an AI-filed request.

    POST rather than GET because the uncached path spends money and is not
    idempotent in cost, even though it is idempotent in effect. The cached
    path is returned without calling the API at all unless `force=true`.
    """
    existing = db.execute(
        select(LLMNarrative).where(LLMNarrative.service_request_id == request_id)
    ).scalar_one_or_none()
    if existing is not None and not force:
        return _narrative_response(existing, cached=True)

    context = _evidence_context(db, request_id)
    try:
        result = generate_narrative(context)
    except LLMNotConfigured as exc:
        # 503, not 500: the service is correctly built and simply unconfigured.
        # The rule-based decision for this request already exists regardless.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface the provider's own message
        logger.exception("narrative generation failed for service request %s", request_id)
        raise HTTPException(status_code=502, detail=f"narrative generation failed: {exc}") from exc

    if existing is not None:
        db.delete(existing)
        db.flush()

    row = LLMNarrative(
        service_request_id=request_id,
        model=result.model,
        diagnosis_narrative=result.diagnosis_narrative,
        maintenance_recommendation=result.maintenance_recommendation,
        corroboration=result.corroboration,
        evidence_cited=result.evidence_cited,
        tool_calls=result.tool_calls,
        ungrounded_numbers=result.ungrounded_numbers,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _narrative_response(row, cached=False)


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
