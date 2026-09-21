"""Pydantic response schemas for the FleetMind API."""

from datetime import datetime
from typing import Literal

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
