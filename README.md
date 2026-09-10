# FleetMind

**Federated Intelligent Predictive Maintenance for Bearing Fault Diagnosis** —
M.Tech project (Debotri Ghosh, 25MML1008), jointly offered by VIT Chennai and
LTIMindTree.

FleetMind learns bearing fault patterns across multiple non-IID operating
environments (the 4 CWRU motor load levels act as 4 simulated depots) and
converts model predictions into actionable maintenance insights via a
multi-agent LLM layer.

## Architecture

```
data/raw/cwru  ─▶  Ingestion (FastAPI + Postgres)  ─▶  EDA  ─▶  Local 1D-CNN per depot
                                                                  │
                                                TimeGAN rare-fault augmentation
                                                                  │
                                                Flower federated learning (FedAvg/FedProx)
                                                                  │
                                                LangGraph agents (diagnosis → risk → action)
                                                                  │
                        Next.js dashboard  ◀────  FastAPI endpoints
```

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 14 (App Router) + TypeScript + Tailwind + Recharts + TanStack Query |
| API | FastAPI + SQLAlchemy 2.0 + Alembic migrations + Pydantic v2 |
| Database | PostgreSQL 16 (Docker) |
| ML (planned) | PyTorch (1D-CNN), TimeGAN, Flower federated learning, LangGraph agents |

## Quick start (Docker)

```bash
cp .env.example .env      # fill in real values
docker compose up --build
# Dashboard:  http://localhost:3000
# API docs:   http://localhost:8000/docs
# Adminer:    http://localhost:8080
```

## Backend (local dev)

```bash
cd backend
pip install -e ".[dev]"
alembic upgrade head                       # create/upgrade schema + seed depots
uvicorn fleetmind.api.main:app --reload    # API at :8000

# Data pipeline (after KAGGLE_USERNAME/KAGGLE_KEY are set in .env):
python -m fleetmind.ingestion.download
python -m fleetmind.ingestion.manifest
python -m fleetmind.ingestion.ingest

# EDA report -> eda_output/
python -m fleetmind.eda.report
```

## Frontend (local dev)

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev   # dashboard at :3000
```

## CI

GitHub Actions (`.github/workflows/ci.yml`): ruff lint + format check, mypy,
pytest (with a Postgres service container + Alembic migrations), Next.js build,
and Docker image builds — on every push/PR to `main`.

## Project layout

```
backend/
  fleetmind/            # Python package
    api/                # FastAPI app + routers + schemas
    db/                 # ORM models, engine, session
    ingestion/          # download, manifest, windowing, CWRU ingest
    eda/                # EDA report generation
  alembic/              # migrations
  tests/                # pytest suite (sqlite-backed)
frontend/               # Next.js 14 dashboard
.github/workflows/      # CI pipeline
docker-compose.yml      # postgres + api + frontend + adminer
```

## Dataset

Case Western Reserve University (CWRU) Bearing Dataset — drive-end (DE)
accelerometer, 12 kHz, fault classes: Normal / Ball / Inner Race / Outer Race,
4 motor loads (0–3 HP → 4 simulated depots). Windowed at 2,048 samples.
Cite the original source: https://engineering.case.edu/bearingdatacenter
