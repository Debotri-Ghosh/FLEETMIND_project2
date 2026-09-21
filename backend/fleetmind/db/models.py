"""SQLAlchemy ORM models mirroring the FleetMind schema."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from fleetmind.db.base import Base

RawValuesType = JSON().with_variant(JSONB(), "postgresql")


class Depot(Base):
    __tablename__ = "depots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    load_hp: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    nominal_rpm: Mapped[int | None] = mapped_column(SmallInteger)


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        CheckConstraint(
            "fault_type IN ('Normal', 'InnerRace', 'OuterRace', 'Ball')",
            name="ck_experiments_fault_type",
        ),
        CheckConstraint("sensor_position IN ('DE', 'FE', 'BA')", name="ck_experiments_sensor"),
        Index("idx_experiments_depot", "depot_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    depot_id: Mapped[int] = mapped_column(ForeignKey("depots.id"), nullable=False)
    fault_type: Mapped[str] = mapped_column(Text, nullable=False)
    fault_diameter_in: Mapped[float | None] = mapped_column(Numeric(5, 3))
    sensor_position: Mapped[str] = mapped_column(Text, nullable=False)
    sampling_rate_hz: Mapped[int] = mapped_column(Integer, nullable=False, default=12000)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    source_variable: Mapped[str] = mapped_column(Text, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SignalWindow(Base):
    __tablename__ = "signal_windows"
    __table_args__ = (
        UniqueConstraint("experiment_id", "window_index", name="uq_windows_exp_idx"),
        Index("idx_windows_experiment", "experiment_id"),
        Index("idx_windows_label", "label"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    window_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_sample: Mapped[int] = mapped_column(Integer, nullable=False)
    end_sample: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_values: Mapped[list[float]] = mapped_column(RawValuesType, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    __table_args__ = (Index("idx_predictions_window", "window_id"),)

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    window_id: Mapped[int] = mapped_column(ForeignKey("signal_windows.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_label: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("model_predictions.id"), nullable=False)
    agent_name: Mapped[str] = mapped_column(Text, nullable=False)
    decision_text: Mapped[str] = mapped_column(Text, nullable=False)
    risk_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Customer(Base):
    """Fleet manager / customer who owns equipment at depots."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str | None] = mapped_column(Text, unique=True)
    company: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Equipment(Base):
    """A physical asset (motor / bearing) at a depot owned by a customer."""

    __tablename__ = "equipment"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'maintenance', 'retired')",
            name="ck_equipment_status",
        ),
        Index("idx_equipment_depot", "depot_id"),
        Index("idx_equipment_customer", "customer_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    depot_id: Mapped[int] = mapped_column(ForeignKey("depots.id"), nullable=False)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    equipment_type: Mapped[str] = mapped_column(Text, nullable=False, default="Motor")
    installed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    # Simulated link: CWRU has no real per-motor identity, so each equipment row
    # is deterministically assigned one CWRU experiment (see agents/sync_equipment_status.py).
    linked_experiment_id: Mapped[int | None] = mapped_column(ForeignKey("experiments.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Technician(Base):
    """Service technician available for dispatch."""

    __tablename__ = "technicians"
    __table_args__ = (
        CheckConstraint(
            "status IN ('available', 'busy', 'offline')",
            name="ck_technician_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    phone: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="available")


class ServiceRequest(Base):
    """Customer-reported issue."""

    __tablename__ = "service_requests"
    __table_args__ = (
        CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'critical')",
            name="ck_service_requests_priority",
        ),
        CheckConstraint(
            "status IN ('reported', 'diagnosed', 'dispatched', 'in_progress', 'resolved', 'cancelled')",
            name="ck_service_requests_status",
        ),
        Index("idx_service_requests_customer", "customer_id"),
        Index("idx_service_requests_depot", "depot_id"),
        Index("idx_service_requests_status", "status"),
        # Partial unique index: at most one *open*, AI-filed service request per
        # equipment. Prevents a race between two concurrent
        # sync_equipment_status runs (see agents/sync_equipment_status.py:
        # _ensure_ai_service_request) from both passing the "already has an
        # open AI-filed request" read-check before either commits, which would
        # otherwise file two AI requests for the same equipment. Declared here
        # (not only in alembic/versions/0003_...) so the sqlite test DB, which
        # is built from Base.metadata.create_all rather than migrations, also
        # enforces it -- see tests/test_sync_equipment_status.py for the race
        # test exercising this for real against sqlite.
        Index(
            "uq_service_requests_open_ai_request",
            "equipment_id",
            unique=True,
            postgresql_where=text(
                "agent_decision_id IS NOT NULL AND status NOT IN ('resolved', 'cancelled')"
            ),
            sqlite_where=text(
                "agent_decision_id IS NOT NULL AND status NOT IN ('resolved', 'cancelled')"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    depot_id: Mapped[int] = mapped_column(ForeignKey("depots.id"), nullable=False)
    equipment_id: Mapped[int | None] = mapped_column(ForeignKey("equipment.id"))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(Text, nullable=False, default="medium")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="reported")
    agent_decision_id: Mapped[int | None] = mapped_column(ForeignKey("agent_decisions.id"))
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("model_predictions.id"))
    technician_id: Mapped[int | None] = mapped_column(ForeignKey("technicians.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class WorkOrder(Base):
    """Technician dispatch tracking for a service request."""

    __tablename__ = "work_orders"
    __table_args__ = (
        CheckConstraint(
            "status IN ('assigned', 'traveling', 'in_progress', 'completed', 'cancelled')",
            name="ck_work_orders_status",
        ),
        UniqueConstraint("service_request_id", name="uq_work_orders_request"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    service_request_id: Mapped[int] = mapped_column(
        ForeignKey("service_requests.id"), nullable=False
    )
    technician_id: Mapped[int] = mapped_column(ForeignKey("technicians.id"), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="assigned")
    location_lat: Mapped[float | None] = mapped_column(Float)
    location_lng: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def seed_customers_payload() -> list[dict[str, Any]]:
    return [
        {"name": "Acme Logistics", "email": "ops@acme.com", "company": "Acme Logistics Ltd"},
        {
            "name": "Global Freight",
            "email": "ops@globalfreight.com",
            "company": "Global Freight Inc",
        },
        {
            "name": "Rapid Transport",
            "email": "ops@rapidtransport.com",
            "company": "Rapid Transport Co",
        },
    ]


def seed_equipment_payload() -> list[dict[str, Any]]:
    return [
        {"depot_id": 1, "customer_id": 1, "name": "Bearing Motor A1", "equipment_type": "Motor"},
        {"depot_id": 1, "customer_id": 1, "name": "Bearing Motor A2", "equipment_type": "Motor"},
        {"depot_id": 2, "customer_id": 1, "name": "Conveyor Motor B1", "equipment_type": "Motor"},
        {"depot_id": 2, "customer_id": 2, "name": "Pump Motor B2", "equipment_type": "Pump"},
        {"depot_id": 3, "customer_id": 2, "name": "Fan Motor C1", "equipment_type": "Fan"},
        {"depot_id": 3, "customer_id": 3, "name": "Compressor C2", "equipment_type": "Compressor"},
        {
            "depot_id": 4,
            "customer_id": 3,
            "name": "Generator Motor D1",
            "equipment_type": "Generator",
        },
        {"depot_id": 4, "customer_id": 1, "name": "Bearing Motor D2", "equipment_type": "Motor"},
    ]


def seed_technicians_payload() -> list[dict[str, Any]]:
    return [
        {"name": "Alice Chen", "phone": "+1-555-0101", "status": "available"},
        {"name": "Bob Martinez", "phone": "+1-555-0102", "status": "available"},
    ]


def seed_depots_payload() -> list[dict[str, Any]]:
    return [
        {"name": "depot_0hp", "load_hp": 0, "nominal_rpm": 1797},
        {"name": "depot_1hp", "load_hp": 1, "nominal_rpm": 1772},
        {"name": "depot_2hp", "load_hp": 2, "nominal_rpm": 1750},
        {"name": "depot_3hp", "load_hp": 3, "nominal_rpm": 1730},
    ]
