"""Shared pytest fixtures: sqlite-backed app + session override."""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_fleetmind.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from fleetmind.api.main import create_app  # noqa: E402
from fleetmind.auth.passwords import hash_password  # noqa: E402
from fleetmind.auth.tokens import create_access_token  # noqa: E402
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
    User,
    seed_depots_payload,
)

# One bcrypt hash reused across every fixture user. Hashing is deliberately
# slow, and paying that cost once per *session* rather than once per fixture
# keeps the suite fast without weakening anything -- the password-handling
# tests hash for real.
TEST_PASSWORD = "test-password"


@pytest.fixture(scope="session")
def test_password_hash() -> str:
    return hash_password(TEST_PASSWORD)


def make_user(session, test_password_hash, **kwargs) -> User:
    user = User(hashed_password=test_password_hash, **kwargs)
    session.add(user)
    session.commit()
    return user


def authorize(test_client, user: User) -> None:
    """Attach a real signed token for `user` to every request the client makes.

    A real token, not a dependency override: overriding get_current_user would
    skip the very code that decodes, verifies and re-checks the account, which
    is the part most worth testing.
    """
    token = create_access_token(
        user_id=user.id,
        role=user.role,
        depot_id=user.depot_id,
        customer_id=user.customer_id,
    )
    test_client.headers["Authorization"] = f"Bearer {token}"


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
def client(db_engine, seeded_session, test_password_hash):
    """Admin-authenticated client.

    Every endpoint now requires a token, so the pre-auth tests authenticate as
    admin: they are about endpoint behaviour, not about the access rules,
    which tests/test_auth.py covers separately with each role.
    """
    app = create_app()

    def override_get_session():
        yield seeded_session

    app.dependency_overrides[db_session.get_session] = override_get_session
    admin = make_user(
        seeded_session, test_password_hash, email="admin@fleetmind.example.com", role="admin"
    )
    with TestClient(app) as c:
        authorize(c, admin)
        yield c


@pytest.fixture()
def anon_client(db_engine, seeded_session):
    """Deliberately unauthenticated, for proving routes are actually closed."""
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

    session.add(
        Customer(id=1, name="Acme Logistics", email="ops@acme.com", company="Acme Logistics Ltd")
    )
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


def _product_app(product_session):
    app = create_app()

    def override_get_session():
        yield product_session

    app.dependency_overrides[db_session.get_session] = override_get_session
    return app


@pytest.fixture()
def product_client(db_engine, product_session, test_password_hash):
    """Admin-authenticated product client (pre-auth tests keep their meaning)."""
    admin = make_user(
        product_session, test_password_hash, email="admin@fleetmind.example.com", role="admin"
    )
    with TestClient(_product_app(product_session)) as c:
        authorize(c, admin)
        yield c


@pytest.fixture()
def customer_client(db_engine, product_session, test_password_hash):
    """Signed in as the customer who owns equipment 1-3 (customer_id=1)."""
    user = make_user(
        product_session,
        test_password_hash,
        email="customer@fleetmind.example.com",
        role="customer",
        customer_id=1,
    )
    with TestClient(_product_app(product_session)) as c:
        authorize(c, user)
        yield c


@pytest.fixture()
def other_customer_client(db_engine, product_session, test_password_hash):
    """Signed in as a *different* customer, who owns nothing in the fixture.

    This is the one that makes the customer scoping provable: everything in
    product_session belongs to customer 1, so this caller must see none of it.
    """
    product_session.add(
        Customer(id=2, name="Northwind Steel", email="ops@northwind.test", company="Northwind")
    )
    product_session.commit()
    user = make_user(
        product_session,
        test_password_hash,
        email="other@fleetmind.example.com",
        role="customer",
        customer_id=2,
    )
    with TestClient(_product_app(product_session)) as c:
        authorize(c, user)
        yield c


@pytest.fixture()
def operator_client(db_engine, product_session, test_password_hash):
    """Operator at site 1, where all the fixture's equipment lives."""
    user = make_user(
        product_session,
        test_password_hash,
        email="op1@fleetmind.example.com",
        role="operator",
        depot_id=1,
        technician_id=1,
    )
    with TestClient(_product_app(product_session)) as c:
        authorize(c, user)
        yield c


@pytest.fixture()
def other_operator_client(db_engine, product_session, test_password_hash):
    """Operator at site 3 — a site with none of the fixture's equipment."""
    user = make_user(
        product_session,
        test_password_hash,
        email="op3@fleetmind.example.com",
        role="operator",
        depot_id=3,
    )
    with TestClient(_product_app(product_session)) as c:
        authorize(c, user)
        yield c


# ── synthetic CWRU-shaped archive (decision-layer tests) ─────────────
import random as _random  # noqa: E402

#: CWRU's real per-site recording structure, which is what makes the
#: single-Normal-recording limitation structural rather than incidental.
RECORDINGS_PER_SITE = {"Ball": 4, "InnerRace": 4, "Normal": 1, "OuterRace": 7}
WINDOWS_PER_RECORDING = 6
SAMPLES = 2048


@pytest.fixture()
def archive_session(db_engine):
    """A synthetic archive with CWRU's exact recording structure, in sqlite.

    Deliberately not the live PostgreSQL database: a test that silently skips
    when a server is unreachable is a test that can stop examining anything, and
    this suite exists to catch exactly that. The real split is verified by
    running `python -m fleetmind.decision.{retrain,compare}` against the live
    archive; these tests pin the structure and the guards.

    Four sites, 16 recordings each, laid out as CWRU is -- including exactly one
    Normal recording per site.
    """
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = factory()
    rng = _random.Random(0)
    for depot_id, load_hp in enumerate([0, 1, 2, 3], start=1):
        session.add(Depot(id=depot_id, name=f"Plant {chr(64 + depot_id)}", load_hp=load_hp))
        session.flush()
        for fault, n_recordings in RECORDINGS_PER_SITE.items():
            for _ in range(n_recordings):
                exp = Experiment(
                    depot_id=depot_id,
                    fault_type=fault,
                    sampling_rate_hz=12000,
                    source_file=f"{fault}.mat",
                    source_variable="DE",
                    sensor_position="DE",
                    n_samples=SAMPLES * WINDOWS_PER_RECORDING,
                )
                session.add(exp)
                session.flush()
                # A per-class amplitude offset so the classes are not identical
                # noise -- the selection accuracy must be able to vary at all.
                scale = 0.05 * (1 + list(RECORDINGS_PER_SITE).index(fault))
                for index in range(WINDOWS_PER_RECORDING):
                    session.add(
                        SignalWindow(
                            experiment_id=exp.id,
                            window_index=index,
                            start_sample=index * SAMPLES,
                            end_sample=(index + 1) * SAMPLES,
                            raw_values=[rng.gauss(0, scale) for _ in range(SAMPLES)],
                            label=fault,
                        )
                    )
    session.commit()
    yield session
    session.close()
