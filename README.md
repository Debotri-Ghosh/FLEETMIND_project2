# FleetMind

**Federated Intelligent Predictive Maintenance for Bearing Fault Diagnosis** —
M.Tech project (Debotri Ghosh, 25MML1008), jointly offered by VIT Chennai and
LTIMindTree.

FleetMind learns bearing fault patterns across multiple non-IID operating
environments (the 4 real CWRU motor load levels — 0/1/2/3 HP — act as 4
simulated manufacturing sites), converts model predictions into maintenance decisions via a
LangGraph multi-agent chain, and bridges those decisions into a small
product layer (equipment, service requests, work orders) with customer and
site-operations dashboards. No synthetic data anywhere except the documented
TimeGAN/WaveGAN-lite augmentation step.

## Architecture

```
CWRU dataset ──▶ Ingestion ──▶ EDA ──▶ Per-site 1D-CNN (FFT-magnitude input)
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
# Per-site classifier (honest experiment-level split, FFT-magnitude input)
python -m fleetmind.ml.train --depot-id 1 --epochs 20
python -m fleetmind.ml.evaluate --depot-id 1

# Rare-class augmentation (WaveGAN-lite) + comparison report, per site
python -m fleetmind.augmentation.run --depot-id 1

# Federated learning across all 4 sites — FedProx is the primary result;
# plain FedAvg is unstable under real per-site class imbalance
python -m fleetmind.federated.run --strategy fedprox --mu 0.1 --rounds 30

# Populate model_predictions + agent_decisions for the LangGraph chain
python -m fleetmind.ml.predict --all-depots
python -m fleetmind.agents.run --all-pending

# Bridge agent_decisions into Equipment.status + auto-file AI service requests
python -m fleetmind.agents.sync_equipment_status          # --dry-run to preview

# Demo logins (run AFTER seed_product_tables.py -- the role scopes point at
# customer and site rows, so those have to exist first)
python -m fleetmind.db.seed_users
```

### Signing in

Every endpoint except `POST /auth/login` and the `GET /livez` liveness probe
requires a bearer token, including `/health`. `POST /auth/login` returns one.

Three roles, each with a different scope, enforced **at the API layer** rather
than by hiding things in the UI:

| Role | Sees |
|---|---|
| `customer` | only its own equipment and service requests, at any site |
| `operator` | only its own site's equipment, queue and dispatches |
| `admin` | everything, including the replay controls |

Seeded demo logins — these passwords are published on purpose so a reviewer
can sign in as each role and check that the restrictions are real. The two
operators sit at **different** sites so the site scoping is demonstrable:

| Email | Password | Role |
|---|---|---|
| `admin@fleetmind.example.com` | `admin-demo-2026` | admin |
| `operator.a@fleetmind.example.com` | `operator-demo-2026` | operator — Plant A (0 HP) |
| `operator.c@fleetmind.example.com` | `operator-demo-2026` | operator — Plant C (2 HP) |
| `customer.acme@fleetmind.example.com` | `customer-demo-2026` | customer — Acme Manufacturing |
| `customer.northwind@fleetmind.example.com` | `customer-demo-2026` | customer — Northwind Steel |

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"operator.a@fleetmind.example.com","password":"operator-demo-2026"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

curl -s localhost:8000/equipment -H "Authorization: Bearer $TOKEN"   # 2 machines
curl -s localhost:8000/equipment                                     # 401
curl -s localhost:8000/replay/status -H "Authorization: Bearer $TOKEN"  # 403, admin only
```

The guarantee worth checking: a caller's scope is read from their user row and
applied *before* any query parameter they send, so `?customer_id=…` or
`?depot_id=…` can only ever narrow a result set, never widen it.
`backend/tests/test_auth.py` asserts this directly, and one of its tests walks
the application's own route table so a router added later without a guard
fails the suite instead of shipping open.

### Optional: the LLM write-up layer

Everything above is deterministic Python. One further, optional step adds a
real Claude call that turns a flagged request into the prose a technician
reads:

```bash
pip install -e ".[llm]"                 # from backend/
echo 'ANTHROPIC_API_KEY=sk-ant-...'  >> .env

curl -X POST localhost:8000/service-requests/41/narrative
```

The model is given **no raw signal and no arithmetic**. It can only call four
tools (`fleetmind/agents/tools.py`) that wrap the same functions which already
produced the stored decision — signal statistics, the rule-based risk
assessment, the bearing characteristic frequencies, and a spectral band-energy
probe — and it is forbidden to state a figure that did not come back from one.
A post-hoc check (`verify_numbers_are_grounded`) looks for any number in the
prose that no tool result accounts for and stores what it finds alongside the
narrative, so the guard's own track record is inspectable.

Narratives are cached in `llm_narratives` (one row per service request), so
generation costs one API call per flagged machine, not one per page view. With
no key configured the endpoint returns a clear `503` and nothing else changes:
predictions, risk scores, equipment flagging and auto-filed service requests
are all unaffected.

Current real results (see `CLAUDE.md` for the full write-up and the two
architecture/data bugs that were fixed to get here — data leakage from a
per-window split, and an `AdaptiveAvgPool1d` collapsing the fault-repetition
signal): per-site validation accuracy 62.7–72.5%; FedProx over 30 rounds
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
                         #   + tools.py / llm.py (the optional Claude write-up layer)
    auth/               # bcrypt passwords, JWT issue/decode, role + row scoping
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
- The bearing characteristic frequencies shown as evidence are **independent
  corroboration, not an explanation of the model**. The CNN consumes a
  log-scaled, z-normalised FFT magnitude and was never given these
  frequencies; the UI says so wherever a defect band is drawn.
- The LLM write-up is prose about numbers it was handed, not an analysis. It
  cannot change a risk score, an urgency or an equipment status — every one of
  those still comes from rule-based code — and the grounding check that backs
  this up is a blunt numeric scan, not a proof.
- The frontend keeps its JWT in `localStorage`, not an httpOnly cookie, so it
  is readable by injected script. The API is a separate origin, so a cookie
  would need `SameSite=None; Secure`, credentialed CORS and TLS in front of
  both — more moving parts than a local demo warrants. It does not weaken what
  this layer demonstrates: every restriction is enforced server-side, so a
  stolen or hand-edited token still cannot read another tenant's rows.
  Switching to cookies is a deployment change, not a redesign.
- Demo passwords are weak and published above, deliberately. There is no
  signup, password reset, rate limiting or refresh-token rotation — all of
  which a real deployment needs and none of which this demo has.

## Dataset

Case Western Reserve University (CWRU) Bearing Dataset — drive-end (DE)
accelerometer, 12 kHz, fault classes: Normal / Ball / Inner Race / Outer Race,
4 motor loads (0–3 HP → 4 simulated manufacturing sites). Windowed at 2,048 samples.
Cite the original source: https://engineering.case.edu/bearingdatacenter
