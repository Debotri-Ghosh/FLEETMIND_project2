import os

import sqlalchemy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="FleetMind API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = sqlalchemy.create_engine(os.environ["DATABASE_URL"])


@app.get("/health")
def health():
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("SELECT 1"))
    return {"status": "ok", "database": "connected"}


@app.get("/depots")
def list_depots():
    with engine.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text(
                """
                SELECT d.id, d.name, d.load_hp, d.nominal_rpm,
                       count(e.id) AS n_experiments
                FROM depots d
                LEFT JOIN experiments e ON e.depot_id = d.id
                GROUP BY d.id
                ORDER BY d.load_hp
                """
            )
        ).mappings().all()
    return [dict(r) for r in rows]


@app.get("/fault-distribution")
def fault_distribution():
    with engine.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT label, count(*) AS n_windows FROM signal_windows GROUP BY label ORDER BY n_windows DESC"
            )
        ).mappings().all()
    return [dict(r) for r in rows]


@app.get("/experiments")
def list_experiments(depot_id: int | None = None, fault_type: str | None = None):
    query = "SELECT id, depot_id, fault_type, fault_diameter_in, sensor_position, source_file, n_samples FROM experiments WHERE 1=1"
    params = {}
    if depot_id is not None:
        query += " AND depot_id = :depot_id"
        params["depot_id"] = depot_id
    if fault_type is not None:
        query += " AND fault_type = :fault_type"
        params["fault_type"] = fault_type
    query += " ORDER BY id"

    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(query), params).mappings().all()
    return [dict(r) for r in rows]


@app.get("/experiments/{experiment_id}/windows")
def list_windows_for_experiment(experiment_id: int, limit: int = 5):
    with engine.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text(
                """
                SELECT id, window_index, label
                FROM signal_windows
                WHERE experiment_id = :experiment_id
                ORDER BY window_index
                LIMIT :limit
                """
            ),
            {"experiment_id": experiment_id, "limit": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


@app.get("/signal-window/{window_id}")
def get_signal_window(window_id: int):
    with engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.text(
                "SELECT id, experiment_id, window_index, label, raw_values FROM signal_windows WHERE id = :id"
            ),
            {"id": window_id},
        ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="window not found")
    return dict(row)
