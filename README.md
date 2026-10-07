# FleetMind

**Federated Intelligent Predictive Maintenance for Bearing Fault Diagnosis**

M.Tech Project 2 · Debotri Ghosh (25MML1008) · VIT Chennai + LTIMindTree

---

## What this is

A **predictive-maintenance and field-service platform for rotating industrial
machinery**. A machine builder or service provider monitors motors, pumps, fans
and compressors installed in its customers' factories, detects bearing faults
early from vibration signals, backs every finding with inspectable evidence,
and dispatches technicians to fix them.

**It recommends actions; humans decide.**

> This is **not** a vehicle fleet, ride-hailing or logistics product. "Fleet"
> here means a fleet of *industrial machines*. The `depot` table and column
> names in the schema are legacy for "manufacturing site" and are deliberately
> retained — see
> [`docs/adr/0001-depots-to-sites-rename.md`](docs/adr/0001-depots-to-sites-rename.md)
> for why. Nothing a user sees says "depot".

### Who it is for

| Role | What they get |
|---|---|
| **Customer** (factory owner) | Their own equipment and service requests, across any site |
| **Site operator** | One site's equipment board, AI-flagged queue and technician dispatch |
| **Admin** | Everything, including the replay/automation controls |

Enforced at the API, not by hiding buttons — see [Authentication](#authentication).

---

## What it actually does

1. **Ingests** real CWRU bearing vibration recordings — 64 recordings, 4,367
   windows of 2,048 samples at 12 kHz.
2. **Classifies** each window as Normal / Ball / InnerRace / OuterRace with a
   per-site 1D CNN over the window's FFT magnitude.
3. **Reasons** over each prediction with a deterministic LangGraph chain —
   Monitor → Diagnostic → Risk → Maintenance — producing a 0–1 risk score and an
   urgency band.
4. **Acts**: a HIGH or CRITICAL decision flips a real equipment row to
   `maintenance` and auto-files a service request, which an operator dispatches
   a technician against.
5. **Explains**: every AI-filed request carries the triggering signal, its
   spectrum, the corroborating bearing defect frequency, the documented ground
   truth beside the independent prediction, and an LLM-written narrative in
   which every number came from a deterministic tool.
6. **Federates**: the same architecture trained across all four sites with
   FedProx, as a non-IID federated-learning study.

---

## Results, stated honestly

### Per-site classifier accuracy: **53.8 – 69.5%**, mean **60.4%**

These are the accuracies of the **checkpoints actually deployed** — the files
under `mlruns/depot_{1-4}/cnn_state_dict.pt` that produce every prediction,
every agent decision and every AI-filed service request in the running system.
Measured on an **experiment-level split**, so no window from a held-out
recording appeared in training.

| Site | Load | Deployed accuracy | Correct / held-out | Margin over the 50.0% floor |
|---|---|---|---|---|
| Plant A | 0 HP | **53.81%** | 127 / 236 | **+3.81 pts** |
| Plant B | 1 HP | **69.49%** | 164 / 236 | +19.49 pts |
| Plant C | 2 HP | **61.02%** | 144 / 236 | +11.02 pts |
| Plant D | 3 HP | **57.20%** | 135 / 236 | +7.20 pts |
| **mean** | — | **60.38%** | 570 / 944 | **+10.38 pts** |

Reproduce with `python -m fleetmind.decision.report` from `backend/`.

#### ⚠ Three caveats that must travel with these numbers

**1. They are three-class figures, not four.** Each site has exactly one Normal
recording, and the stratified experiment split assigns a single-recording class
entirely to training — so **no healthy machine is ever held out, at any site**.
The held-out set is identical everywhere: `{OuterRace 118, Ball 59, InnerRace 59}`.
The model emits four classes; only three are ever tested.

**2. The floor is 50.0%, not 25%.** OuterRace is exactly half the held-out set,
so always predicting OuterRace scores 50.00%. That is the baseline these figures
must clear. Plant A clears it by 3.81 points.

**3. The false-alarm rate on a healthy machine has never been measured**, and
cannot be on this split, for the reason in caveat 1. For a predictive-maintenance
product that is the commercially decisive number. It is absent, not zero.

#### Why these differ from an earlier 62.7–72.5% claim

Earlier versions of this README quoted 64.83 / 72.46 / 62.71 / 65.68% from
`mlruns/depot_{1-4}/metrics.json`. Those are `best_val_acc` — the best accuracy
any *epoch* reached during training. `ml/train.py` calls `torch.save()` after the
training loop with no best-epoch guard, so the weights that were saved are the
**last** epoch's, not the best one's. The earlier figures therefore describe
models that were never written to disk. The deployed checkpoints score a mean
6.04 points lower. Confirmed twice: through `fleetmind/decision/probs.py`, and by
replaying `ml/train.py`'s own validation loop and its own dataset class against
the saved checkpoints.

The deployed checkpoints are deliberately left in place rather than retrained,
because all 4,367 stored predictions and decisions were produced by them and
retraining would invalidate every downstream figure. The honest measured number
is quoted instead.

### Federated: a negative result, presented as one

FedProx (mu = 0.1, 30 rounds) reaches **42.86%** mean accuracy against a
**60.05%** mean local baseline. More precisely, and worth saying out loud: its
per-class recall is **OuterRace 1.0 and 0.0 for every other class, at all four
sites** — federated averaging across clients with different class distributions
collapsed the global model into a **majority-class predictor**. Plain FedAvg was
worse still: it oscillated round to round, which is why FedProx is the primary
result.

### ⚠ The 91.44% figure is **not** accuracy

The database holds 4,367 predictions of which 3,993 match the documented label.
That 91.44% is **in-sample model-vs-ground-truth agreement** — each site's model
predicted over all of its own windows, including ones it trained on. It is a
useful integration check and a meaningless generalisation claim.

**Quote 53.8–69.5% (mean 60.4%), three-class, against a 50.0% floor. Never quote 91.44% as accuracy.**

Full detail, including the three real bugs that *lowered* these numbers, is in
[`FleetMind_Implementation_Report.md`](FleetMind_Implementation_Report.md).

---

## Architecture

```
backend/fleetmind/
  ingestion/    CWRU download, manifest, windowing, ingest
  db/           SQLAlchemy models, session, seed scripts
  api/          FastAPI app + 11 routers
  auth/         bcrypt passwords, JWT tokens, role + row scoping
  eda/          exploratory data analysis reports
  ml/           model, dataset, train, evaluate, predict, bearing_freq
  augmentation/ GAN-based rare-class augmentation
  federated/    Flower FedAvg/FedProx simulation
  agents/       LangGraph chain, sync_equipment_status, tools.py, llm.py
  replay/       engine + background worker
  pipeline.py   predict -> agents -> sync, one code path
frontend/src/   Next.js 14 App Router, TanStack Query, Recharts, Tailwind
alembic/        6 migrations
tests/          257 tests
```

**Database:** PostgreSQL, 12 tables. Migrations run `0001` core schema → `0002`
product tables → `0003` AI-request partial unique index → `0004` replay arrival
marker → `0005` LLM narrative cache → `0006` users.

---

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI + SQLAlchemy 2.0 + Alembic + Pydantic v2 |
| Database | PostgreSQL 18 (native — see [Setup](#setup)) |
| ML | PyTorch 1D-CNN on 512-bin FFT magnitude |
| Federated | Flower (FedAvg / FedProx) |
| Agents | LangGraph `StateGraph`, deterministic and rule-based |
| LLM narrative | **Groq**, `openai/gpt-oss-120b` — optional |
| Auth | PyJWT (HS256) + bcrypt, three roles |
| Frontend | Next.js 14, TanStack Query, Recharts, Tailwind, IBM Plex |

### Why Groq

The LLM layer was built against the Anthropic SDK targeting Claude. The API key
available for this project turned out to be a **Groq** key, so the module was
rewritten for Groq.

**The grounding design did not change at all** — four deterministic tools, a
prompt containing no measurements, and a post-hoc numeric check — because none
of it depends on the provider. Only the call shape changed. Swapping providers
again is a rewrite of one module, not of the design.

One provider quirk, found by probing rather than assumed: Groq rejects JSON mode
combined with tool calling (`400 json mode cannot be combined with
tool/function calling`), so the loop runs tools first and collects the
structured answer in a second, tool-free call. A test pins this so the
workaround cannot be undone by accident.

---

## Setup

### Prerequisites on this machine

- **PostgreSQL 18, running natively** — *not* Docker. Docker Desktop cannot run
  here because WSL2 and Hyper-V are blocked by corporate policy.
  Database `fleetmind`, user `fleetmind`, `localhost:5432`.
- **Python 3.11** in a venv at the repository root (`venv/`).
- **Node.js 24** — installed **user-scope via winget**
  (`winget install --scope user`) because the admin-elevated installer was
  blocked. It therefore lives under
  `%LOCALAPPDATA%\Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_*\node-v24.19.0-win-x64\`
  rather than `C:\Program Files\nodejs\`. It **is** on the user PATH and
  `node`/`npm` resolve normally in PowerShell and Git Bash — but if you go
  looking in Program Files you will not find it, and an elevated shell that does
  not inherit the user PATH may not see it.

### Install

```bash
# from the repository root
python -m venv venv
venv\Scripts\activate                 # Windows

cd backend
pip install -e ".[ml,dev,llm]"
```

> **Run all backend commands from `backend/`.** `.env` and several relative
> paths assume it.

**On the extras:** everything needed to *boot the API* is in the core
dependencies, including `torch` and `langgraph` — the replay router reaches both
through `pipeline.py`. `[ml]` adds `flwr` for the federated simulation, `[dev]`
the test and lint tools, `[llm]` the `groq` client.
`tests/test_dependencies.py` fails if anything the API imports drifts back into
an optional extra.

### Configure

`backend/.env`:

```ini
DATABASE_URL=postgresql://fleetmind:<password>@localhost:5432/fleetmind

# Optional. Without it the whole rule-based pipeline still runs; only the
# narrative endpoint returns a clear 503.
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b

# Optional but recommended. With no secret the API generates a random one per
# process and logs a warning -- tokens then stop surviving a restart.
JWT_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(48))">
```

`frontend/.env` (local only, not committed):

```ini
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

> Pin this to `127.0.0.1`, not `localhost`. Docker Desktop's `wslrelay.exe` can
> squat on `[::1]:8000` (IPv6 loopback) and 404 every browser-side fetch while
> server-side fetches keep working — a genuinely confusing failure mode that
> cost real time on this project.

### Migrate and seed

Order matters — the user scopes point at customer and site rows:

```bash
cd backend
alembic upgrade head
python -m fleetmind.db.seed_product_tables
python -m fleetmind.db.seed_users
```

### Run both servers

**One command (recommended):**

```powershell
.\scripts\start-demo.ps1
```

It frees ports 8000 and 3000, starts both servers clean, waits until each
answers healthy, and prints the demo logins.

**Manually:**

```bash
# terminal 1 — from backend/, venv active
python -m uvicorn fleetmind.api.main:app --host 127.0.0.1 --port 8000

# terminal 2 — from frontend/
npm install      # first time only; CI uses npm ci against the committed lockfile
npm run dev
```

Then open **http://localhost:3000**.

> **uvicorn runs without `--reload`.** Restart it after every backend change — a
> stale process serving old code has caused phantom failures here more than once.

> **Never run `npm run build` while `npm run dev` is live in the same
> directory.** The production build overwrites the dev server's `.next` asset
> manifests and every page starts 404-ing its own JS. Stop dev, delete `.next`,
> build, then restart dev fresh.

### Sign in

Every route requires a bearer token except `POST /auth/login` and the
unauthenticated `GET /livez` liveness probe — **including `/health`**, which
reports database state.

These passwords are published deliberately, so a reviewer can sign in as each
role and verify the restrictions are real. The two operators sit at
**different** sites on purpose.

| Email | Password | Role |
|---|---|---|
| `admin@fleetmind.example.com` | `admin-demo-2026` | admin |
| `operator.a@fleetmind.example.com` | `operator-demo-2026` | operator — Plant A (0 HP) |
| `operator.c@fleetmind.example.com` | `operator-demo-2026` | operator — Plant C (2 HP) |
| `customer.acme@fleetmind.example.com` | `customer-demo-2026` | customer — Acme Manufacturing |
| `customer.northwind@fleetmind.example.com` | `customer-demo-2026` | customer — Northwind Steel |

---

## Data & ML pipeline

```bash
cd backend

# Ingest CWRU (needs KAGGLE_USERNAME/KAGGLE_KEY, or download manually)
python -m fleetmind.ingestion.download
python -m fleetmind.ingestion.ingest

# Per-site classifier (experiment-level split, FFT-magnitude input)
python -m fleetmind.ml.train --depot-id 1 --epochs 20
python -m fleetmind.ml.evaluate --depot-id 1

# Rare-class augmentation + comparison report
python -m fleetmind.augmentation.run --depot-id 1

# Federated learning across all 4 sites
python -m fleetmind.federated.run --strategy fedprox --mu 0.1 --rounds 30

# Predictions -> agent decisions -> equipment status + auto-filed requests
python -m fleetmind.ml.predict --all-depots
python -m fleetmind.agents.run --all-pending
python -m fleetmind.agents.sync_equipment_status        # --dry-run to preview
```

### The replay (live automation)

CWRU is a **static archive**, so a scheduler over it would recompute identical
results on a timer. Instead `signal_windows.released_at` marks *arrival*, and
the replay engine progressively releases archived windows as if sensors were
streaming them. The dashboard's Live Automation panel drives it.

> **It is a replay of recorded data — not live telemetry, and no sensors are
> attached.** The UI says so on the panel itself.

---

## Authentication

Three roles, enforced at the API layer:

| Role | Sees |
|---|---|
| `customer` | Only its own equipment and requests |
| `operator` | Only its own site's equipment, queue and dispatches |
| `admin` | Everything, including replay |

The guarantee worth checking: **a caller's scope is read from their user row and
applied before any query parameter they send**, so `?customer_id=…` or
`?depot_id=…` can only ever narrow a result set, never widen it. A row outside
scope returns **404, not 403**, so ids cannot be enumerated from status codes.

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"operator.a@fleetmind.example.com","password":"operator-demo-2026"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

curl -s localhost:8000/equipment -H "Authorization: Bearer $TOKEN"     # 2 machines
curl -s localhost:8000/equipment                                       # 401
curl -s localhost:8000/replay/status -H "Authorization: Bearer $TOKEN"  # 403, admin only
```

`backend/tests/test_auth.py` asserts all of this directly, and one of its tests
walks the application's **own route table**, so a router added later without a
guard fails the suite instead of shipping open.

---

## Testing

```bash
cd backend
pytest                      # 257 tests
ruff check .
ruff format --check .
mypy fleetmind

cd ../frontend
npx tsc --noEmit
npm run build
```

---

## Known honest limitations

1. **Docker is unverified end to end.** `docker compose config` resolves
   cleanly and the images are defined, but the Docker daemon is unreachable on
   the development laptop (WSL2/Hyper-V blocked by policy), so
   `docker compose up` has **never actually been run**. Treat the Compose path
   as untested.
2. **The GAN augmentation has an unresolved domain mismatch.** TimeGAN
   generates *time-domain* synthetic windows, but the classifier now trains on
   their *FFT magnitude*. A window can be plausible in time and unrealistic in
   frequency — the likely reason augmentation sometimes hurts accuracy.
   Documented, not fixed.
3. **GPS / location fields on `Equipment` and `WorkOrder` are simulated.** CWRU
   has no location telemetry. They are labelled as simulated wherever shown and
   never implied to be real.
4. **The four "sites" are four CWRU motor load levels.** A real site could host
   machines at several loads; these cannot. Display names keep the mapping
   visible (*Plant A (CWRU 0 HP)*).
5. **`Equipment.linked_experiment_id` is a simulated association** between a
   product row and a bench recording — the largest open modelling gap.
6. **The federated result is negative** (see above), presented as a finding
   rather than tuned away.
7. **Defect-band corroboration is often weak.** On the sampled windows the
   dominant spectral energy does not sit on the characteristic defect
   frequency. The evidence UI surfaces this rather than cropping it away.
8. **Auth is demo-grade.** The JWT lives in `localStorage` rather than an
   httpOnly cookie, because the API is a separate origin. Every restriction is
   still server-side — a stolen token cannot read another tenant's rows — but
   there is no refresh rotation, rate limiting or password reset, and the demo
   passwords above are public.
9. **No literature review document exists yet.** It is outstanding work.

---

## Documentation

| File | What it is |
|---|---|
| [`FleetMind_Implementation_Report.md`](FleetMind_Implementation_Report.md) | Full technical account, every figure re-verified |
| [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | Click-through for the defence, with caveats to volunteer |
| [`CLAUDE.md`](CLAUDE.md) | Working context + the authoritative Progress log |
| [`docs/adr/`](docs/adr/) | Architecture decision records |
| [`docs/audit.md`](docs/audit.md) | Repo audit, with a status banner per finding |

---

## Dataset

Case Western Reserve University (CWRU) Bearing Data Center — drive-end
accelerometer, 12 kHz, classes Normal / Ball / Inner Race / Outer Race, four
motor loads (0–3 HP) standing in for four manufacturing sites. Windowed at
2,048 samples.

**Cite the original source:** https://engineering.case.edu/bearingdatacenter
