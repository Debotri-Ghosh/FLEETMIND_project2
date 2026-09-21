"""FastAPI application factory."""

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from fleetmind.api.routers import (
    customers,
    depots,
    equipment,
    experiments,
    service_requests,
    stats,
    technicians,
    windows,
    work_orders,
)
from fleetmind.core.config import get_settings
from fleetmind.db import session as db_session
from fleetmind.db.models import seed_depots_payload


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="FleetMind API",
        version=settings.app_version,
        description="Federated Intelligent Predictive Maintenance for Bearing Fault Diagnosis",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )

    app.include_router(depots.router)
    app.include_router(experiments.router)
    app.include_router(windows.router)
    app.include_router(stats.router)
    app.include_router(equipment.router)
    app.include_router(service_requests.router)
    app.include_router(work_orders.router)
    app.include_router(customers.router)
    app.include_router(technicians.router)

    @app.get("/health", tags=["system"])
    def health(db: Session = Depends(db_session.get_session)):
        # Lightweight connectivity probe; a failure raises and surfaces as a 500.
        db.connection()
        return {"status": "ok", "database": "connected", "version": settings.app_version}

    return app


def seed_depots(database_url: str | None = None) -> None:
    """Dialect-safe depot seeding. Safe to re-run."""
    engine = db_session.create_db_engine(database_url)
    with engine.begin() as conn:
        for row in seed_depots_payload():
            conn.execute(
                text(
                    "INSERT INTO depots (name, load_hp, nominal_rpm) "
                    "VALUES (:name, :load_hp, :nominal_rpm)"
                ),
                row,
            )
    engine.dispose()


app = create_app()
