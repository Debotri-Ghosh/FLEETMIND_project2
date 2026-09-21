"""Shared pytest fixtures: sqlite-backed app + session override."""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_fleetmind.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from fleetmind.api.main import create_app  # noqa: E402
from fleetmind.db import session as db_session  # noqa: E402
from fleetmind.db.base import Base  # noqa: E402
from fleetmind.db.models import (  # noqa: E402
    AgentDecision,
    Customer,
    Depot,
    Equipment,
    Experiment,
    ModelPrediction,
    SignalWindow,
    Technician,
    seed_depots_payload,
)


@pytest.fixture()
def db_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def seeded_session(db_engine):
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = factory()
    for row in seed_depots_payload():
        session.add(Depot(**row))
    session.flush()
    exp = Experiment(
        depot_id=1,
        fault_type="InnerRace",
        fault_diameter_in=0.007,
        sensor_position="DE",
        sampling_rate_hz=12000,
        source_file="12k_DE/Load_0/InnerRace/IR007_0.mat",
        source_variable="X097_DE_time",
        n_samples=4096,
    )
    session.add(exp)
    session.flush()
    session.add(
        SignalWindow(
            experiment_id=exp.id,
            window_index=0,
            start_sample=0,
            end_sample=8,
            raw_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            label="InnerRace",
        )
    )
    session.commit()
    yield session
    session.close()


@pytest.fixture()
def client(db_engine, seeded_session):
    app = create_app()

    def override_get_session():
        yield seeded_session

    app.dependency_overrides[db_session.get_session] = override_get_session
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def product_session(db_engine):
    """Seeds depots + customers/technicians/equipment + one experiment ->
    window -> prediction -> agent_decision chain, for product-layer router
    and sync_equipment_status tests."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = factory()

    for row in seed_depots_payload():
        session.add(Depot(**row))
    session.flush()

    session.add(Customer(id=1, name="Acme Logistics", email="ops@acme.com", company="Acme Logistics Ltd"))
    session.add(Technician(id=1, name="Alice Chen", phone="+1-555-0101", status="available"))
    session.flush()

    exp = Experiment(
        depot_id=1,
        fault_type="InnerRace",
        fault_diameter_in=0.007,
        sensor_position="DE",
        sampling_rate_hz=12000,
        source_file="12k_DE/Load_0/InnerRace/IR007_0.mat",
        source_variable="X097_DE_time",
        n_samples=4096,
    )
    session.add(exp)
    session.flush()

    window = SignalWindow(
        experiment_id=exp.id,
        window_index=0,
        start_sample=0,
        end_sample=8,
        raw_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        label="InnerRace",
    )
    session.add(window)
    session.flush()

    prediction = ModelPrediction(
        window_id=window.id, model_version="v1", predicted_label="InnerRace", confidence=0.9
    )
    session.add(prediction)
    session.flush()

    session.add(
        AgentDecision(
            prediction_id=prediction.id,
            agent_name="risk",
            decision_text="[HIGH] elevated fault repetition rate",
            risk_score=0.8,
        )
    )

    session.add(
        Equipment(
            id=1,
            depot_id=1,
            customer_id=1,
            name="Bearing Motor A1",
            equipment_type="Motor",
            status="active",
            linked_experiment_id=exp.id,
        )
    )
    session.add(
        Equipment(
            id=2,
            depot_id=1,
            customer_id=1,
            name="Bearing Motor A2",
            equipment_type="Motor",
            status="active",
            linked_experiment_id=None,
        )
    )
    session.add(
        Equipment(
            id=3,
            depot_id=1,
            customer_id=1,
            name="Retired Motor",
            equipment_type="Motor",
            status="retired",
            linked_experiment_id=exp.id,
        )
    )
    session.commit()
    yield session
    session.close()


@pytest.fixture()
def product_client(db_engine, product_session):
    app = create_app()

    def override_get_session():
        yield product_session

    app.dependency_overrides[db_session.get_session] = override_get_session
    with TestClient(app) as c:
        yield c
