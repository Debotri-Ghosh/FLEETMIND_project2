"""Pydantic response schemas for the FleetMind API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    status: str
    database: str
    version: str


class Depot(BaseModel):
    id: int
    name: str
    load_hp: int
    nominal_rpm: int | None
    n_experiments: int = 0

    model_config = ConfigDict(from_attributes=True)


class Experiment(BaseModel):
    id: int
    depot_id: int
    fault_type: str
    fault_diameter_in: float | None
    sensor_position: str
    sampling_rate_hz: int
    source_file: str
    source_variable: str
    n_samples: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WindowSummary(BaseModel):
    id: int
    experiment_id: int
    window_index: int
    start_sample: int
    end_sample: int
    label: str

    model_config = ConfigDict(from_attributes=True)


class WindowDetail(WindowSummary):
    raw_values: list[float] = Field(description="Raw vibration samples for this window")


class FaultDistributionEntry(BaseModel):
    label: str
    n_windows: int


class DepotModelMetrics(BaseModel):
    depot_id: int
    depot_name: str
    best_val_acc: float
    train_windows: int
    val_windows: int


class FederatedSummary(BaseModel):
    strategy: str
    rounds: int
    mean_baseline_accuracy: float
    mean_final_accuracy: float
    gain_vs_baseline: float
    per_depot_final_accuracy: dict[int, float] = Field(
        description="Federated (post-training) accuracy per depot, for comparison against "
        "each depot's own local-baseline best_val_acc in per_depot."
    )


class ModelPerformance(BaseModel):
    per_depot: list[DepotModelMetrics] = Field(
        description="Honest, experiment-level-split per-depot classifier results from mlruns/."
    )
    federated: FederatedSummary | None = Field(
        default=None,
        description="FedProx federated-learning summary from mlruns/federated/fedprox/, "
        "or null if that run hasn't been produced.",
    )


class ProductSummary(BaseModel):
    equipment_by_status: dict[str, int]
    open_requests_by_priority: dict[str, int] = Field(
        description="Requests not in ('resolved', 'cancelled'), grouped by priority."
    )
    ai_flagged_open: int = Field(description="Open requests with agent_decision_id set.")
    active_work_orders: int = Field(description="Work orders not in ('completed', 'cancelled').")


class MatchedFrequency(BaseModel):
    """The bearing characteristic frequency that corroborates the predicted fault.

    Null for Normal (no defect frequency exists) — the UI must show "no band"
    rather than drawing one."""

    name: str
    hz: float
    band_low_hz: float
    band_high_hz: float
    harmonic: int


class EvidencePackage(BaseModel):
    """Inspectable evidence behind one AI-filed service request.

    Everything here is derived from the one signal window that triggered the
    decision, so a reviewer can check the finding rather than trust it."""

    service_request_id: int
    window_id: int
    equipment_id: int | None
    site_name: str | None = Field(default=None, description="Display name of the site.")

    predicted_label: str
    model_confidence: float | None = Field(
        default=None, description="Softmax confidence of the CNN's predicted class."
    )
    model_version: str | None = None

    raw_values: list[float] = Field(description="The window's raw vibration samples.")
    sampling_rate_hz: int
    spectrum_freqs_hz: list[float] = Field(
        description="Frequency axis (Hz) of the plain magnitude spectrum."
    )
    spectrum_magnitudes: list[float] = Field(
        description="Raw FFT magnitudes — NOT the log-scaled, z-normalized tensor the "
        "CNN consumes; this axis is meant to be readable by a person."
    )
    matched_frequency: MatchedFrequency | None = None
    all_characteristic_frequencies: dict[str, float] = Field(
        description="BPFI/BPFO/BSF/FTF for this shaft speed and bearing."
    )

    signal_rms: float = Field(description="Recomputed by the diagnostic agent's own function.")
    signal_peak: float
    diagnosis: str
    diagnosis_confidence: float
    risk_score: float | None = None
    decision_text: str | None = None


class ReplayStatusResponse(BaseModel):
    """Progress of the simulated sensor feed over the static CWRU archive."""

    total_windows: int
    released: int = Field(description="Windows that have 'arrived' so far.")
    processed: int
    pending: int = Field(description="Arrived but not yet predicted.")
    percent_released: float
    complete: bool
    worker: dict = Field(description="Background worker state: running, interval, last tick.")


class ReplayStartRequest(BaseModel):
    interval_seconds: float | None = Field(
        default=None, ge=0.25, le=60.0, description="Seconds between ticks."
    )
    batch_size: int | None = Field(
        default=None, ge=1, le=500, description="Windows released per tick."
    )


class ReplayTickResponse(BaseModel):
    released: int
    predictions_written: int
    decisions_written: int
    equipment_status_changes: int
    service_requests_filed: int
    error: str | None = None


class ReplayResetResponse(BaseModel):
    windows_unreleased: int
    predictions_deleted: int
    decisions_deleted: int
    ai_service_requests_deleted: int
    work_orders_deleted: int
    equipment_reset_to_active: int


class Customer(BaseModel):
    id: int
    name: str
    email: str | None
    company: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Technician(BaseModel):
    id: int
    name: str
    phone: str | None
    status: str

    model_config = ConfigDict(from_attributes=True)


class Equipment(BaseModel):
    id: int
    depot_id: int
    customer_id: int
    name: str
    equipment_type: str
    installed_at: datetime | None
    status: str
    linked_experiment_id: int | None = Field(
        default=None,
        description="Simulated link to a CWRU experiment — CWRU has no real per-motor identity.",
    )
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ServiceRequest(BaseModel):
    id: int
    customer_id: int
    depot_id: int
    equipment_id: int | None
    title: str
    description: str | None
    priority: str
    status: str
    agent_decision_id: int | None
    prediction_id: int | None
    technician_id: int | None
    created_at: datetime
    updated_at: datetime
    agent_decision_text: str | None = Field(
        default=None, description="Set only for AI-filed requests (agent_decision_id is not null)."
    )
    agent_risk_score: float | None = Field(
        default=None, description="Set only for AI-filed requests (agent_decision_id is not null)."
    )
    predicted_fault: str | None = Field(
        default=None, description="Set only for AI-filed requests (agent_decision_id is not null)."
    )

    model_config = ConfigDict(from_attributes=True)


class ServiceRequestCreate(BaseModel):
    customer_id: int
    depot_id: int
    equipment_id: int | None = None
    title: str
    description: str | None = None
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class WorkOrder(BaseModel):
    id: int
    service_request_id: int
    technician_id: int
    status: str
    location_lat: float | None = Field(
        default=None, description="Simulated GPS — CWRU has no location telemetry."
    )
    location_lng: float | None = Field(
        default=None, description="Simulated GPS — CWRU has no location telemetry."
    )
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class WorkOrderCreate(BaseModel):
    service_request_id: int
    technician_id: int
    location_lat: float | None = None
    location_lng: float | None = None


class WorkOrderStatusUpdate(BaseModel):
    status: Literal["assigned", "traveling", "in_progress", "completed", "cancelled"]


class NarrativeToolCall(BaseModel):
    """One tool the model called while writing a narrative, and what it got back.

    Persisted verbatim so any sentence in the narrative can be checked against
    the figures the model was actually handed."""

    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    result: str


class ServiceRequestNarrative(BaseModel):
    """Claude's write-up of one AI-filed service request.

    Prose only. Every figure in it came from a deterministic tool; the model
    is not permitted to originate a number, and `ungrounded_numbers` lists
    anything the post-hoc grounding check could not trace back to a tool
    result (normally empty)."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    service_request_id: int
    model: str
    diagnosis_narrative: str
    maintenance_recommendation: str
    corroboration: Literal["strong", "partial", "weak", "none"]
    evidence_cited: list[str] = Field(default_factory=list)
    tool_calls: list[NarrativeToolCall] = Field(default_factory=list)
    ungrounded_numbers: list[str] = Field(
        default_factory=list,
        description="Numbers in the prose not found in any tool result. Should be empty.",
    )
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: datetime
    cached: bool = Field(
        default=True, description="False when this response generated a fresh API call."
    )
