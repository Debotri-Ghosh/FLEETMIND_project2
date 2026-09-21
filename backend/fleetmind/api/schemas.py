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
