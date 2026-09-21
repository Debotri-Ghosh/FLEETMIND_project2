# FleetMind

**Federated Intelligent Predictive Maintenance for Bearing Fault Diagnosis** —
M.Tech project (Debotri Ghosh, 25MML1008), jointly offered by VIT Chennai and
LTIMindTree.

FleetMind learns bearing fault patterns across multiple non-IID operating
environments (the 4 real CWRU motor load levels — 0/1/2/3 HP — act as 4
simulated depots), converts model predictions into maintenance decisions via a
LangGraph multi-agent chain, and bridges those decisions into a small
product layer (equipment, service requests, work orders) with customer and
depot-operator dashboards. No synthetic data anywhere except the documented
TimeGAN/WaveGAN-lite augmentation step.

## Architecture

```
CWRU dataset ──▶ Ingestion ──▶ EDA ──▶ Per-depot 1D-CNN (FFT-magnitude input)
                                          │
                        WaveGAN-lite rare-fault augmentation
                                          │
                        Flower federated learning (FedAvg / FedProx)
                                          │
                LangGraph agents: Monitor → Diagnostic → Risk → Maintenance
                                          │
                fleetmind.agents.sync_equipment_status (the ML → product bridge)
                                          │
        Equipment.status + auto-filed AI ServiceRequests ──▶ FastAPI ──▶ Next.js
                                                        (dashboard / customer / operator)
```

Product-layer tables (`customers`, `equipment`, `technicians`,
`service_requests`, `work_orders`) sit alongside the core ML tables
(`depots`, `experiments`, `signal_windows`, `model_predictions`,
`agent_decisions`) in the same Postgres database — see
`backend/fleetmind/db/models.py` for the full schema.

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 14 (App Router) + TypeScript + Tailwind + Recharts + TanStack Query |
| API | FastAPI + SQLAlchemy 2.0 + Alembic migrations + Pydantic v2 |
| Database | PostgreSQL 16+ |
| ML | PyTorch (1D-CNN on FFT-magnitude input), WaveGAN-lite augmentation, Flower federated learning (FedAvg/FedProx), LangGraph agents |

## Setup

Two ways to run this, depending on your machine:

### Option A — native Postgres + Python venv (used for local development on this project)

Some environments (e.g. a locked-down company laptop) block the
virtualization Docker Desktop needs (WSL2/Hyper-V). This path avoids Docker
entirely for the database and backend.

```bash
# 1. Postgres running natively on localhost:5432, with a `fleetmind` DB/user.
# 2. Python 3.11 venv at the repo root:
python -m venv venv
venv\Scripts\activate            # Windows; `source venv/bin/activate` on macOS/Linux

cd backend
pip install -e ".[dev]"
# backend/.env: DATABASE_URL=postgresql://fleetmind:<password>@localhost:5432/fleetmind

# Migrations run in order: base -> 0001 (core ML schema) -> 0002 (product-layer schema)
alembic upgrade head

# Seed order matters: product tables (customers/equipment/technicians) must
# exist before the agent-sync bridge can link equipment to experiments.
python -m fleetmind.db.seed_product_tables

uvicorn fleetmind.api.main:app --reload    # API at :8000
```

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev   # dashboard at :3000
```

### Option B — Docker Compose

```bash
cp .env.example .env      # fill in real values
docker compose up --build
# Dashboard:  http://localhost:3000
# API docs:   http://localhost:8000/docs
# Adminer:    http://localhost:8080
```

`docker-compose.yml` lives at the repo root; its `api`/`migrate` services
build from `./backend` and `frontend` builds from `./frontend` — both
relative to the repo root, since that's where the compose file is.
`migrate` runs `alembic upgrade head` automatically; run
`python -m fleetmind.db.seed_product_tables` inside the `api` container
afterward to seed the product layer (see Option A's seed-order note above).

## Data & ML pipeline

Run from `backend/`, after ingestion (`python -m fleetmind.ingestion.download`,
`.manifest`, `.ingest` — needs `KAGGLE_USERNAME`/`KAGGLE_KEY` in `.env` to pull
the CWRU mirror):

```bash
# Per-depot classifier (honest experiment-level split, FFT-magnitude input)
python -m fleetmind.ml.train --depot-id 1 --epochs 20
python -m fleetmind.ml.evaluate --depot-id 1

# Rare-class augmentation (WaveGAN-lite) + comparison report, per depot
python -m fleetmind.augmentation.run --depot-id 1

# Federated learning across all 4 depots — FedProx is the primary result;
# plain FedAvg is unstable under real per-depot class imbalance
python -m fleetmind.federated.run --strategy fedprox --mu 0.1 --rounds 30

# Populate model_predictions + agent_decisions for the LangGraph chain
python -m fleetmind.ml.predict --all-depots
python -m fleetmind.agents.run --all-pending

# Bridge agent_decisions into Equipment.status + auto-file AI service requests
python -m fleetmind.agents.sync_equipment_status          # --dry-run to preview
```

Current real results (see `CLAUDE.md` for the full write-up and the two
architecture/data bugs that were fixed to get here — data leakage from a
per-window split, and an `AdaptiveAvgPool1d` collapsing the fault-repetition
signal): per-depot validation accuracy 62.7–72.5%; FedProx over 30 rounds
converges to a mean accuracy of 42.9% vs. a 60.0% mean local baseline — a
real, documented federated-averaging gap under non-IID class imbalance, not
a bug to "fix away."

## Testing

```bash
cd backend
ruff check .            # lint
ruff format --check .   # format
mypy fleetmind          # type check
pytest                  # sqlite-backed test suite, no DB needed
```

```bash
cd frontend
npx tsc --noEmit
npm run build
```

## CI

GitHub Actions (`.github/workflows/ci.yml`): ruff lint + format check, mypy,
pytest (with a Postgres service container + Alembic migrations), Next.js
build, and Docker image builds — on every push/PR to `main`.

## Project layout

```
backend/
  fleetmind/
    ingestion/          # download, manifest, windowing, CWRU ingest
    db/                 # ORM models (core ML + product-layer), session, seed scripts
    api/                # FastAPI app + routers (depots/experiments/stats/windows/
                         #   equipment/service-requests/work-orders/customers/technicians)
    eda/                # EDA report generation
    ml/                 # model, dataset, train, evaluate, predict
    augmentation/       # WaveGAN-lite rare-class augmentation
    federated/          # Flower FedAvg/FedProx simulation
    agents/             # LangGraph Monitor->Diagnostic->Risk->Maintenance chain,
                         #   + sync_equipment_status (the ML -> product bridge)
  alembic/              # migrations (0001 core schema, 0002 product-layer schema)
  tests/                # pytest suite (sqlite-backed)
frontend/
  src/app/              # Next.js 14 dashboard, customer/ and operator/ pages
.github/workflows/      # CI pipeline
docker-compose.yml      # postgres + api + migrate + frontend + adminer
```

## Known, honest limitations

- Location/GPS fields on `Equipment`/`WorkOrder` are **simulated** — CWRU has
  no location telemetry — and are labeled as such wherever the UI shows them.
- WaveGAN-lite augmentation currently generates *time-domain* synthetic
  windows, but the classifier trains on their FFT magnitude; synthetic
  windows can be time-domain-plausible while spectrally unrealistic, which is
  a likely reason augmentation sometimes hurts post-fix accuracy.
- The federated accuracy gap vs. local baselines (see above) is a known,
  published FL phenomenon under real class imbalance — presented as a
  finding, not concealed.

## Dataset

Case Western Reserve University (CWRU) Bearing Dataset — drive-end (DE)
accelerometer, 12 kHz, fault classes: Normal / Ball / Inner Race / Outer Race,
4 motor loads (0–3 HP → 4 simulated depots). Windowed at 2,048 samples.
Cite the original source: https://engineering.case.edu/bearingdatacenter
