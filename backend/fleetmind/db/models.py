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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from fleetmind.db.base import Base

# JSONB on Postgres, plain JSON elsewhere (tests use sqlite).
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


def seed_depots_payload() -> list[dict[str, Any]]:
    """The 4 real CWRU load levels -> 4 simulated depots."""
    return [
        {"name": "depot_0hp", "load_hp": 0, "nominal_rpm": 1797},
        {"name": "depot_1hp", "load_hp": 1, "nominal_rpm": 1772},
        {"name": "depot_2hp", "load_hp": 2, "nominal_rpm": 1750},
        {"name": "depot_3hp", "load_hp": 3, "nominal_rpm": 1730},
    ]
