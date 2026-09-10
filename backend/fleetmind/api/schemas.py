"""Pydantic response schemas for the FleetMind API."""

from datetime import datetime

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
