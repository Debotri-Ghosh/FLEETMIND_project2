# Repo audit — 2026-09-30

> ## Status as of 2026-10-02 — read this before acting on anything below
>
> This audit was produced by a review pass on 2026-09-30 and its findings were
> real. Several have since been **closed**; the text below is left unedited as
> the historical record, so check here first before treating a finding as live.
>
> | Finding | Status |
> |---|---|
> | §2.1 LangGraph used but no LLM anywhere | **Closed** — Slice G added a real Claude call (`agents/llm.py`) over deterministic tools. The audit's point that the *agents themselves* are rule-based remains true and is deliberate: the LLM writes prose, it never originates a number. |
> | §2.2 Dependencies — a clean install cannot run this code | **Closed 2026-10-02.** `langgraph` was declared nowhere and `torch` only under `[ml]`, while CI installed `.[dev]`. Both are now core dependencies, CI installs `.[ml,dev]`, dead `scikit-learn`/`structlog` declarations removed, and `tests/test_dependencies.py` now fails if this regresses. Confirmed to fail against the pre-fix manifest. |
> | §2.3 Authentication — there is none | **Closed** — Slice H added JWT auth with three roles, row-level scoping and 49 tests. |
> | §2.4 CLAUDE.md framing predates the product statement | **Closed** — Slice E reframed the copy; CLAUDE.md's header sections were rewritten on 2026-10-02. |
> | §3 Legacy `depot` rename inventory | **Deliberately deferred** — see `docs/adr/0001`. User-facing copy says "site"; the schema keeps `depot`. Re-audited 2026-10-02: zero vehicle/ride/logistics language remains in code, UI, README or the live database. |
> | §1.1/§1.4 counts (10 tables, 4 migrations, 137 tests) | **Superseded** — now 12 tables, 6 migrations, 244 tests. |
> | §0 / §5 Product spec missing, gap analysis blocked | **Still true.** No `docs/PRODUCT_SPEC.md` exists. The product statement at the top of `CLAUDE.md` is the de facto spec and was written after this audit. |
>
> The one finding below that was **not** acted on and is not merely historical
> is §4.1 (`Equipment.linked_experiment_id` is a simulated association) — that
> is a real, documented limitation of mapping a public bench dataset onto a
> product schema, and it is disclosed in the README rather than fixed.


Audit of the FleetMind repo against the product statement now at the top of
`CLAUDE.md`. Every claim below is backed by a command that was actually run;
the command and its real output are shown inline.

> **Scope limit — read this first.** This audit was asked for "against the
> spec" (`docs/PRODUCT_SPEC.md`). **That file does not exist in this repo.**
> See §0. Everything that does not depend on the spec is complete below
> (§1–§4). The gap analysis against the spec (§5) and the whole of
> `docs/plans/M1-plan.md` plus the spec-flagged ADRs are **blocked** until the
> spec is provided, because "what's missing" and "Milestone 1" are defined by
> it. Nothing was invented to fill that gap.

---

## 0. Blocker: the product spec is missing

```
$ ls -la docs/PRODUCT_SPEC.md
ls: cannot access 'docs/PRODUCT_SPEC.md': No such file or directory

$ find docs -type f
docs/plans/2026-09-21-fleetmind-implementation/00-overview.md
docs/plans/2026-09-21-fleetmind-implementation/01-slice-a-harden-phase5.md
docs/plans/2026-09-21-fleetmind-implementation/02-slice-b-close-ml-product-loop.md
docs/plans/2026-09-21-fleetmind-implementation/03-slice-c-unified-dashboard.md
docs/plans/2026-09-21-fleetmind-implementation/04-slice-d-deployment.md

$ find . -iname "*spec*" -o -iname "*PRODUCT*"   # excluding venv/node_modules/.git
./backend/alembic/versions/0002_product_tables.py
./backend/fleetmind/db/seed_product_tables.py
./backend/tests/test_product_api.py
./docs/plans/2026-09-21-fleetmind-implementation/02-slice-b-close-ml-product-loop.md

$ git status --short
(clean — so it is not an uncommitted local file either)
```

No spec document exists anywhere in the repo, tracked or untracked. It was
never committed.

**What this blocks:** the "what is missing" half of this audit, the
Milestone 1 increment plan, and the ADRs for "each decision the spec flags as
needing one". These are all derived from the spec's contents — writing them
from my own assumptions would mean auditing the repo against my invention
rather than your product thinking, and every downstream ADR would inherit
that. Provide the spec and both documents follow immediately.

---

## 1. What exists today (verified)

### 1.1 Database — 10 tables, 4 migrations

```
$ psql -c "SELECT ... FROM information_schema.tables WHERE table_schema='public'"
agent_decisions, alembic_version, customers, depots, equipment, experiments,
model_predictions, service_requests, signal_windows, technicians, work_orders

$ ls alembic/versions/*.py | xargs -n1 basename
0001_initial_schema.py
0002_product_tables.py
0003_ai_service_request_unique_index.py
0004_signal_window_released_at.py
```

Two groups: an ML/evidence group (`experiments`, `signal_windows`,
`model_predictions`, `agent_decisions`) and a service group (`customers`,
`equipment`, `technicians`, `service_requests`, `work_orders`), joined by
`depots` and by `equipment.linked_experiment_id`.

### 1.2 API — 25 routes

```
$ curl -s localhost:8000/openapi.json   # summarised
GET  /health
GET  /depots
GET  /experiments · /experiments/{id} · /experiments/{id}/windows
GET  /signal-window/{window_id}
GET  /stats/fault-distribution · /stats/model-performance · /stats/product-summary
GET  /equipment · /equipment/{equipment_id}
GET  /customers · /customers/{customer_id}
GET  /technicians · /technicians/{technician_id}
GET,POST /service-requests      GET /service-requests/{request_id}
GET,POST /work-orders           GET /work-orders/{work_order_id}
PATCH /work-orders/{work_order_id}/status
GET  /replay/status   POST /replay/{start,stop,step,reset}
```

### 1.3 Detection, evidence, dispatch, and the live loop

| Capability | Where | State |
|---|---|---|
| Vibration ingest + windowing | `backend/fleetmind/ingestion/` | works; CWRU archive only |
| Per-machine fault classifier | `backend/fleetmind/ml/` (`model.py`, `train.py`, `predict.py`) | works; 62.7–72.5% val accuracy |
| Rare-class augmentation | `backend/fleetmind/augmentation/` | works |
| Federated training | `backend/fleetmind/federated/` | works (FedProx) |
| Decision chain | `backend/fleetmind/agents/` | works; **rule-based**, see §2.1 |
| Detection → work bridge | `backend/fleetmind/agents/sync_equipment_status.py` | works; auto-files requests |
| Incremental/live pipeline | `backend/fleetmind/pipeline.py`, `backend/fleetmind/replay/` | works; replay-driven |
| Customer + operator UI | `frontend/src/app/{customer,operator}/page.tsx` | works |
| Evidence shown in UI | operator "AI-Flagged Requests" panel | fault type, risk score, decision text |

### 1.4 Test suite — 137 passing

```
$ ../venv/Scripts/python.exe -m pytest -q -p no:warnings
........................................................................ [ 52%]
.................................................................        [100%]
```

---

## 2. Claims in CLAUDE.md that are wrong or incomplete

### 2.1 LangGraph — claim is TRUE, but the dependency is undeclared

CLAUDE.md describes a "LangGraph Monitor->Diagnostic->Risk->Maintenance chain".
LangGraph is genuinely used, not decorative:

```
$ grep -n "langgraph\|StateGraph\|add_node\|compile()" fleetmind/agents/graph.py
14:from langgraph.graph import END, StateGraph
27:    graph = StateGraph(AgentState)
29:    graph.add_node("monitor", monitor_agent)
30:    graph.add_node("diagnostic", diagnostic_agent)
31:    graph.add_node("risk", risk_agent)
32:    graph.add_node("maintenance", maintenance_agent)
40:    return graph.compile()

$ python -c "import langgraph; print(langgraph.__version__)"
0.1.19
```

**But:** the four agents are deterministic rule-based functions (lookup tables
`FAULT_SEVERITY`, `LOAD_RISK_FACTOR`; `risk = severity × load_factor`). **No LLM
is called anywhere** — no `openai`, `anthropic`, `langchain` or `ollama` import
exists in the codebase, despite `.env.example` advertising Azure OpenAI and
Ollama settings. LangGraph provides orchestration only. Anywhere the project is
described as "LLM-powered", that is wrong.

### 2.2 Dependencies — a fresh install cannot run this code (**high severity**)

```
$ sed -n '/^dependencies/,/^\]/p' pyproject.toml
pydantic-settings, fastapi, uvicorn, sqlalchemy, psycopg2-binary, alembic,
python-dotenv, numpy, pandas, scipy, matplotlib, structlog

$ # per-library: imported by code vs declared in pyproject
torch:     imported in 8 file(s) | declared: only under the [ml] extra
flwr:      imported in 1 file(s) | declared: only under the [ml] extra
langgraph: imported in 1 file(s) | declared: 0   ← declared NOWHERE
sklearn:   imported in 0 file(s) | declared under [ml]   ← dead declaration
kagglehub: imported in 0 file(s) | declared under [ingest] ← dead declaration

$ grep -n "pip install" .github/workflows/ci.yml
42:        run: pip install -e ".[dev]"
```

CI installs `.[dev]` only. That pulls in **neither `[ml]` nor langgraph**, so on
a clean runner `import torch` and `import langgraph` both fail, taking
`test_agents.py`, `test_model.py`, `test_federated.py`, `test_replay.py` and
`test_predict_incremental.py` with them. It passes locally only because this
machine's venv has them installed from earlier manual work. **CI has never
actually run** (the workflow sat in the wrong directory until commit
`627fc882` and has never been pushed), which is why nobody has seen it fail.
README's "pip install -e \".[dev]\"" setup instruction is therefore wrong.

### 2.3 Authentication — there is none

```
$ grep -rniE "oauth|jwt|login|password|authenticate|HTTPBearer|api_key" fleetmind/ --include=*.py
   (no matches, after excluding POSTGRES_PASSWORD / DATABASE_URL strings)
```

Zero authentication, authorization, sessions, or tenancy enforcement. Every
endpoint is fully open. The customer portal's "Viewing as" dropdown is a
client-side simulation of identity — any caller can read or write any
customer's data by passing a different `customer_id`. Any claim that the
product isolates one customer's machines from another's is currently false.

### 2.4 Product framing in CLAUDE.md predates the product statement

The line `4 real motor load levels (0/1/2/3 HP) act as 4 simulated depots`
and `"""Fleet manager / customer who owns equipment at depots."""` both encode
the superseded framing. The new statement at the top now governs; the body of
CLAUDE.md has not yet been rewritten to match.

---

## 3. Legacy "fleet / depot / vehicle" inventory (full rename list)

Scope result: **`vehicle` does not appear anywhere in code** (`grep -rni
"vehicle" backend/fleetmind frontend/src README.md` → no matches), and
**`fleet` appears in a non-product-name sense exactly once**. The entire
problem is `depot`.

### 3.1 Database — expensive, needs a migration

| Identifier | Location | Proposed |
|---|---|---|
| table `depots` | `models.py:31`, `0001_initial_schema.py` | `sites` |
| `depots.name` / `load_hp` / `nominal_rpm` | table columns | `sites.name` / keep / keep |
| `experiments.depot_id` | `models.py:51` | `site_id` |
| `equipment.depot_id` | `models.py:145` | `site_id` |
| `service_requests.depot_id` | `models.py:217` | `site_id` |
| index `idx_experiments_depot` | `models.py:47` | `idx_experiments_site` |
| index `idx_equipment_depot` | `models.py:140` | `idx_equipment_site` |
| index `idx_service_requests_depot` | `models.py:190` | `idx_service_requests_site` |

```
$ psql -c "SELECT table_name||'.'||column_name FROM information_schema.columns
           WHERE table_schema='public' AND (table_name LIKE '%depot%' OR column_name LIKE '%depot%')"
 depots.id | depots.load_hp | depots.name | depots.nominal_rpm
 equipment.depot_id | experiments.depot_id | service_requests.depot_id
```

Seed data also carries depot-shaped values: `depot_0hp`, `depot_1hp`,
`depot_2hp`, `depot_3hp` (`models.py:seed_depots_payload`), which are
site *names*, not just identifiers — renaming the column does not fix these.

### 3.2 API — public contract, breaking change

- Route `GET /depots` → `GET /sites`
- Query parameter `depot_id` on `/equipment`, `/service-requests` (and in
  `ServiceRequestCreate`) → `site_id`
- Response field `depot_id` on Equipment, ServiceRequest, Experiment schemas
- `api/schemas.py`: class `Depot`, `DepotModelMetrics.depot_id`/`depot_name`
- `/stats/model-performance` returns `per_depot` and
  `per_depot_final_accuracy` — both leak the term into ML reporting

### 3.3 Python — 26 files reference `depot`

Notably `ml/train.py`, `ml/predict.py`, `ml/evaluate.py`, `federated/run.py`,
`augmentation/run.py`, `agents/run.py`, `agents/sync_equipment_status.py`,
`pipeline.py`, plus `mlruns/depot_{1-4}/` **on-disk artifact paths**, which
`stats.py` reads by that literal path — renaming the directory convention is a
separate, data-migration-shaped decision from renaming the column.

### 3.4 Frontend — 6 files, 19 user-visible strings

```
$ grep -rhoE "\"[^\"]*[Dd]epot[^\"]*\"|>[A-Za-z ]*[Dd]epot[A-Za-z ]*<" frontend/src --include=*.tsx | sort -u
"Depot Operator"
"Failed to load depots."
"depots"
>All depots<
>Depot Operator<
>Depot<
```

Plus `lib/types.ts` (`Depot`, `depot_id`, `DepotModelMetrics`), `lib/api.ts`
(`api.depots`, `depot_id` params), and the route/nav label **"Depot Operator"**
— which under the new statement should read as the service provider's own
operations console, not a depot.

### 3.5 The single non-product-name "fleet"

```
backend/fleetmind/db/models.py:118:    """Fleet manager / customer who owns equipment at depots."""
```

---

## 4. Other findings worth deciding on

1. **`equipment.linked_experiment_id` is explicitly simulated.** Each machine is
   deterministically mapped to one CWRU experiment; there is no real
   machine↔signal identity. Under a product framing where a customer's real
   motor must map to its real signal stream, this is the deepest modelling gap.
2. **Location/GPS fields on `work_orders` are simulated and unused by the UI.**
   Honest today, but a field-service product presumably needs real ones.
3. **No `sites` concept independent of load level.** `depots` currently encodes
   an *operating condition* (0–3 HP), not a customer's physical factory. The
   rename is therefore not purely cosmetic — the table's *meaning* has to change
   from "load level" to "customer site", which is a data-model decision, not a
   find-and-replace.
4. **No audit trail on human decisions.** "It recommends actions and humans
   decide" — but a technician's accept/reject of a recommendation is not
   recorded anywhere; `work_orders.status` transitions are unlogged and
   deliberately unrestricted.
5. **Evidence is inspectable but not immutable.** `agent_decisions` rows are
   deleted and rebuilt wholesale by `predict_depot(incremental=False)` and by
   `reset_replay`.

---

## 5. Gap analysis vs. the spec — BLOCKED

Cannot be completed: see §0. Provide `docs/PRODUCT_SPEC.md` and this section,
`docs/plans/M1-plan.md`, and the spec-flagged ADRs will be written against it.

One ADR is not blocked, because you named it explicitly and §3 now supplies the
full evidence for it: `docs/adr/0001-depots-to-sites-rename.md`.
