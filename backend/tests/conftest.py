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
    Depot,
    Experiment,
    SignalWindow,
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
