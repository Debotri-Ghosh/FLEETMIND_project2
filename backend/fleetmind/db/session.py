"""Engine / session management."""

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from fleetmind.core.config import get_settings


def create_db_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    factory = make_session_factory(create_db_engine())
    session = factory()
    try:
        yield session
    finally:
        session.close()


def ping(engine) -> bool:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
