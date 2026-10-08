# FleetMind — the complete walkthrough

**Written for: Debotri, to study from and present.** Everything here was read
from the actual source and the live database on 2026-10-05, across all 53
commits. Nothing is from memory.

This is the document the project did not have: **what was built, in what order,
why, and how the pieces actually talk to each other at runtime.** The other docs
cover slices of it — this is the connective tissue.

| If you want | Read |
|---|---|
| The whole story, build order, runtime flow | **this file** |
| Formal technical account with every figure | `FleetMind_Implementation_Report.md` |
| Setup and overview | `README.md` |
| What to click, in what order, at the defence | `DEMO_SCRIPT.md` |
| The frequency question, as Q&A | `FREQUENCY_EXPLAINED_FOR_DEFENSE.md` |
| Every decision and bug, dated | `CLAUDE.md` Progress log |

---

# PART 1 — The one-paragraph version

> FleetMind is a predictive-maintenance platform for rotating industrial
> machinery. It takes real vibration recordings from bearings, classifies the
> fault with a 1D CNN, converts that prediction into a maintenance decision
> with an expected-cost policy and a sequential test over consecutive windows
> (the original rule-based agent chain is kept as a measured baseline), and acts on it — flipping a machine's status
> and filing a service request a technician gets dispatched against. Every
> decision carries inspectable evidence: the raw signal, its spectrum, the
> bearing physics that should corroborate it, and the dataset's own ground truth
> shown beside the model's independent prediction. It is multi-tenant with
> three roles enforced at the API, and the whole pipeline runs incrementally as
> data "arrives" through a replay engine. **It recommends; humans decide.**

If you can say that, you can start any presentation.

---

# PART 2 — How it was built, in order

The build has a logic to it: **data first, then the model, then decisions, then
the product, then honesty, then polish.** Each layer only became possible once
the one beneath it worked.

## Phase 0 — Foundation (7–11 Sept, commits 1–4)

Repository, production layout (`backend/` Python package + `frontend/` Next.js),
Docker Compose, CI workflow, PostgreSQL schema, and the CWRU ingestion pipeline:
download → manifest → windowing → ingest.

**Result:** 64 recordings became **4,367 signal windows** of 2,048 samples each,
in `signal_windows`.

## Phase 2 — The classifier (11 Sept, commit 5)

`BearingCNN` — a 1D convolutional network — plus train/evaluate scripts and
per-site model artefacts in `mlruns/depot_N/`.

*(Phase 1 was the EDA/ingestion work folded into Phase 0.)*

## Phase 3 — Augmentation (11 Sept, commit 6)

WaveGAN-lite conditional augmentation to synthesise extra windows for
under-represented fault classes, with a baseline-vs-augmented comparison per
site.

## Phase 4 — Federated learning (11 Sept, commit 7)

Flower-style FedAvg and FedProx across the four sites as a non-IID study.

## Phase 5 — The agent chain (12 Sept, commit 8)

LangGraph `StateGraph` wiring four deterministic agents: Monitor → Diagnostic →
Risk → Maintenance.

> **At this point the project was a research pipeline.** It produced numbers. It
> was not yet a product — nothing happened as a *result* of a prediction.

## Slices A–D — Becoming a product (21–22 Sept, commits 9–29)

- **A** — product-layer API routers (equipment, service requests, work orders,
  customers, technicians), the ML→equipment sync bridge, and customer/operator
  UIs.
- **B** — **the loop closes.** A HIGH/CRITICAL decision now auto-files a real
  service request. This is the moment the model starts *causing* something.
- **C** — a unified dashboard merging ML metrics with product KPIs.
- **D** — CI green: ruff, format, mypy, pytest.

Then a full hardening pass over each: input validation, error paths, a
race-condition guard with a partial unique index, graceful handling of corrupt
metrics files.

## Dynamic workflow — Event-driven (25 Sept, commits 30–33)

CWRU is a static archive, so a scheduler over it would recompute identical
results on a timer — automation theatre. Instead `signal_windows.released_at`
marks **arrival**, and a replay engine progressively releases windows as if
sensors were streaming them.

## Slices E–H — Honesty and depth (30 Sept – 1 Oct, commits 34–39)

- **E** — domain reframe: industrial manufacturing, not vehicles/logistics.
- **F** — the evidence package: raw signal, spectrum, bearing defect frequency.
- **G** — fixed a real risk-scoring bug, then added the LLM narrative layer.
- **H** — JWT auth with three roles and row-level scoping.

## Slices I–L — Pre-defence hardening (2–3 Oct, commits 41–44)

- **I** — found that **a clean clone could not start the app** (`langgraph`
  declared nowhere, `torch` extra-only); made the live LLM call.
- **J/K** — the instrument-panel visual system, live automation as the
  dashboard centrepiece, and the evidence/Frequency Lab UI.
- **L** — full verification gate and `DEMO_SCRIPT.md`.

## Slices M–O + fixes — Documentation and operations (3–5 Oct, commits 45–53)

Implementation report, README rewrite, one-command startup script, demo reset,
final verification — and two late bug finds, including **an unauthenticated data
leak** (Part 8).

---

# PART 3 — The runtime workflow, end to end

This is the spine of the project. **Learn this path and you can answer almost
anything.**

```
 CWRU .mat files
      │  ingestion/
      ▼
 ┌──────────────┐
 │ signal_windows│  4,367 rows · 2,048 samples each · released_at marks arrival
 └──────┬───────┘
        │  replay/engine.py releases a batch
        ▼
 ┌──────────────────┐
 │ ml/predict.py     │  to_fft_magnitude() → BearingCNN → class + confidence
 └──────┬───────────┘
        ▼
 ┌────────────────────┐
 │ model_predictions   │  4,367 rows
 └──────┬─────────────┘
        │  agents/graph.py  (LangGraph)
        ▼
 Monitor → Diagnostic → Risk → Maintenance
        │
        ▼
 ┌────────────────────┐
 │ agent_decisions     │  risk_score + urgency + decision text
 └──────┬─────────────┘
        │  agents/sync_equipment_status.py
        ▼
 equipment.status → 'maintenance'  +  auto-filed service_request
        │
        ▼
 operator dispatches a technician → work_order
        │
        ▼
 Next.js UI: live board · evidence panel · Frequency Lab
```

**`pipeline.py:process_arrived_windows()` chains predict → agents → sync** so
the background worker and the manual "Step one batch" button share exactly one
code path. That matters: there is no separate "demo mode" that behaves
differently from the real thing.

## Walking one window through, concretely

Take **signal window #2713** (the example used throughout the docs):

1. **Arrival.** The replay engine sets `released_at`. The window is now visible
   to the pipeline.
2. **Prediction.** `predict.py` calls `to_fft_magnitude()` → 512 numbers → the
   CNN → **"Ball", confidence 0.9991**. Written to `model_predictions`.
3. **Monitor agent.** Label ≠ Normal → flags it anomalous.
4. **Diagnostic agent.** Computes RMS **0.105521 g**, peak-to-peak **0.938875 g**
   from the raw samples; sets a diagnosis confidence.
5. **Risk agent.** `risk = severity × load_factor × (0.7 + 0.3 × confidence)`.
   Ball severity 0.40, Plant D load factor 1.30 → **risk 0.4914**.
6. **Maintenance agent.** 0.4914 falls in the 0.4–0.7 band → urgency **HIGH**,
   action "Plan maintenance within 48 hours."
7. **Sync.** HIGH → `Bearing Motor D2` flips to `maintenance`, and a service
   request is filed automatically.
8. **Operator** sees it in the AI-Flagged panel, opens the evidence, dispatches
   a technician.

That is the whole product in one row of data.

---

# PART 4 — Every subsystem, and how its code works

## 4.1 Ingestion — `fleetmind/ingestion/`

`download.py` (Kaggle mirror or manual), `manifest.py` (catalogues the `.mat`
files and parses fault type, defect diameter, load from the filenames),
`windowing.py` (slices continuous recordings into 2,048-sample windows),
`ingest.py` (writes to PostgreSQL).

**Why 2,048 samples:** at 12 kHz that is 0.1707 s, during which the shaft turns
about 5 times — so a once-per-revolution defect is captured several times per
window.

## 4.2 The classifier — `fleetmind/ml/`

### `model.py` — two things that matter

**`to_fft_magnitude(values, n_bins=512)`** is the single most important function
in the project:

```
2,048 raw samples
  → np.fft.rfft           → 1,025 magnitudes, bin width 12000/2048 = 5.859375 Hz
  → keep first 512        → covers 0–3,000 Hz
  → np.log1p              → compresses a ~4,800× dynamic range to ~58×
  → (x − mean) / std      → mean 0, std 1 (verified: 1.18e-16, 1.000000)
```

Log-scaling stops loud bins drowning quiet ones. Z-normalisation removes
absolute loudness, so the model learns the *shape* of the spectrum rather than
how hard the motor is working.

**`BearingCNN`** — three Conv1d blocks (16→32→64 channels, each with BatchNorm +
ReLU), `AdaptiveAvgPool1d(1)`, then a `Linear(64, 4)` classifier.

**`class_weights()`** — inverse-frequency weights for `CrossEntropyLoss`, because
OuterRace is 37.8% of the data and Normal only 19.0%.

### `predict.py`

Loads the per-site checkpoint, applies `to_fft_magnitude`, runs the CNN, writes
`model_predictions`. `incremental=True` processes only arrived-and-unpredicted
windows; the default `False` preserves the original batch behaviour exactly.

### `bearing_freq.py` — the physics

Computes BPFI / BPFO / BSF / FTF from shaft speed and published bearing
multipliers. **The CNN never sees these.** They are independent corroboration.

## 4.3 The agent chain — `fleetmind/agents/`

`state.py` defines `AgentState`, a TypedDict flowing through the graph.
`graph.py` compiles the LangGraph chain. `agents.py` holds the four pure
functions — each is `state → state`, so each is trivially testable and runs
without any LLM.

| Agent | What it does |
|---|---|
| **Monitor** | Flags non-Normal predictions as anomalous |
| **Diagnostic** | Computes RMS and peak-to-peak; downgrades a fault call if signal energy is implausibly low |
| **Risk** | `severity × load_factor`, scaled by confidence |
| **Maintenance** | Maps risk to urgency band + recommended action |

`sync_equipment_status.py` is the **ML → product bridge**: it flips
`equipment.status` and files a service request — guarded so re-runs are
idempotent. Under the heuristic rule it takes the worst decision per machine;
under the default expected-cost rule it runs a repeated SPRT over the policy's
decisions in window order and files against the window where it crossed.

## 4.3a The decision layer — `fleetmind/decision/`

The authoritative path since Phase 5. `policy.py` prices every action from the
stored softmax (`P(faulted) = 1 − P(Normal)`) using the explicit cost matrix in
`costs.py` and picks the cheapest; `sequential.py` is the repeated SPRT that
decides when a machine is flagged. `holdout.py`, `calibration.py`, `quality.py`,
`crosssite.py` and the `phase*.py` reports are the measurement side. Results and
caveats: `FleetMind_Implementation_Report.md` §8A. The four rule agents above
remain as the baseline (`DECISION_POLICY=heuristic`).

## 4.4 The replay engine — `fleetmind/replay/`

`engine.py` (release / status / reset) and `worker.py` (an asyncio tick loop with
the CPU-bound work offloaded to a thread so the API stays responsive).

> **Say this before you are asked:** it is a replay of a recorded archive, not
> live telemetry. No sensors are attached. The UI says so on the panel.

## 4.5 The evidence layer — `agents/tools.py` + the API

Four tools, each a thin read-only wrapper over code that already ran:

| Tool | Wraps |
|---|---|
| `get_signal_statistics` | `_signal_stats` (RMS, peak-to-peak) |
| `get_risk_assessment` | `risk_agent` + `maintenance_agent` |
| `get_bearing_frequency_match` | `matched_frequency_for_fault` |
| `measure_spectral_band_energy` | an FFT over the stored window |

## 4.6 The LLM layer — `agents/llm.py`

**Provider: Groq, `openai/gpt-oss-120b`.** The model writes prose and nothing
else. Four mechanisms stop it inventing numbers:

1. The tools are the only source of figures.
2. **The prompt contains no measurements** — only identifiers. It *must* call a
   tool to learn any number.
3. `verify_numbers_are_grounded()` scans the output for numbers absent from
   every tool result and **stores** what it finds.
4. The system prompt forbids revising the risk score.

**The guard has fired twice on real output, both times correctly** — once when
the model converted `0.000325` to `0.0325 %`, once when it rounded `258.23` to
`260`. Neither was a fabrication; both were the model originating a figure, and
both were caught.

Groq rejects JSON mode combined with tool calling, so the loop runs tools first,
then a separate tool-free call for the structured answer.

## 4.7 Auth — `fleetmind/auth/`

`passwords.py` (bcrypt), `tokens.py` (PyJWT HS256), `deps.py` (the dependencies).

Three things kept deliberately separate:

- **Authentication** — `get_current_user` re-reads the user row *every request*,
  so deactivating an account takes effect immediately.
- **Authorization** — `require_roles` refuses a role outright.
- **Row scoping** — `scope_by_customer` / `scope_by_depot` return SQL conditions
  each endpoint composes into its own query.

> **The guarantee:** scope comes from the user row and is applied *before* any
> client query parameter, so `?customer_id=…` can only narrow a result, never
> widen it. Out-of-scope rows return **404, not 403**, so ids cannot be
> enumerated.

## 4.8 Frontend — `frontend/src/`

16 TypeScript files. Next.js 14 App Router, TanStack Query, Recharts, Tailwind,
self-hosted IBM Plex.

| File | Role |
|---|---|
| `live-board.tsx` | The dashboard centrepiece — counters, flashing equipment cards, animating requests |
| `frequency-lab.tsx` | Waveform → press button → same chart morphs into spectrum |
| `ground-truth.tsx` | Documented truth vs independent prediction, side by side |
| `auth-provider.tsx` | Session handling; `RequireAuth` route guard |
| `theme.ts` | Validated chart colours in one place |

**The design rule:** amber means *live*, and nothing else. A pulsing amber dot
always means automation is running.

---

# PART 5 — The database

12 tables (+ `alembic_version`). Current live counts:

| Table | Rows | What it holds |
|---|---|---|
| `signal_windows` | 4,367 | Raw vibration + `released_at` arrival marker |
| `model_predictions` | 4,367 | CNN output per window |
| `agent_decisions` | 4,367 | Risk score + urgency per prediction |
| `experiments` | 64 | One per CWRU recording |
| `depots` | 4 | The four sites (legacy name for "site") |
| `equipment` | 8 | Machines, with status |
| `service_requests` | 10 | 6 AI-filed + customer-filed |
| `work_orders` | 4 | Technician dispatch |
| `customers` | 3 | Tenants |
| `technicians` | 2 | Field staff |
| `users` | 5 | Logins with role + scope |
| `llm_narratives` | 1 | Cached LLM write-ups |

Six migrations, verified to run from empty against real PostgreSQL.

---

# PART 6 — The results, said correctly

### Accuracy: **53.8 – 69.5%** per site, mean **60.4%**

These are the **deployed** checkpoints — the files that produced every prediction
and decision in the database. `python -m fleetmind.decision.report` reproduces them.

| Site | Load | Deployed accuracy | Correct / held out | Margin over the 50.0% floor |
|---|---|---|---|---|
| Plant A | 0 HP | **53.81%** | 127 / 236 | **+3.81 pts** |
| Plant B | 1 HP | **69.49%** | 164 / 236 | +19.49 pts |
| Plant C | 2 HP | **61.02%** | 144 / 236 | +11.02 pts |
| Plant D | 3 HP | **57.20%** | 135 / 236 | +7.20 pts |
| **mean** | — | **60.38%** | 570 / 944 | **+10.38 pts** |

**Three things are part of this number, not footnotes to it.**

1. **It is a three-class figure.** Every site has exactly one Normal recording and
   the split assigns a single-recording class entirely to training, so **no
   healthy machine is ever held out**. The held-out set is the same everywhere:
   `{OuterRace 118, Ball 59, InnerRace 59}`.
2. **The floor is 50.0%, not 25%** — OuterRace is exactly half the held-out set.
   Plant A beats always-guessing-OuterRace by 3.81 points, with InnerRace recall
   0.0 and an OuterRace predicted share of 0.915.
3. **The false-alarm rate on a healthy machine has never been measured** and
   cannot be on this split. Absent, not zero.

### Why this replaces an earlier 62.7–72.5% claim

`metrics.json` stores `best_val_acc` — the best accuracy any *epoch* reached.
`ml/train.py` calls `torch.save()` **after** the training loop with no best-epoch
guard, so the weights on disk are the **last** epoch's. The old figures
(64.83 / 72.46 / 62.71 / 65.68%) describe models that were never saved; the
deployed ones score a mean 6.04 points lower. Confirmed twice — through
`decision/probs.py`, and by replaying `train.py`'s own validation loop against the
saved checkpoints.

The checkpoints are **deliberately not retrained over**: all 4,367 stored
predictions and decisions came from them. A properly three-way-split retrain goes
to separate `mlruns/depot_N_v2` files so v1 and v2 can be compared on data neither
trained on.

### ⚠ The 91.44% figure is **not** accuracy

It is **in-sample agreement** — each model predicted over its own training
windows too. Useful as an integration check, meaningless as generalisation.
**Quote 53.8–69.5% (mean 60.4%), three-class, against a 50.0% floor. Never quote 91.44% as accuracy.**

### Federated: a negative result, stated as one

FedProx reaches **42.86%** against a **60.05%** local baseline. The per-class
recall is **OuterRace 1.0 and 0.0 for everything else, at all four sites** —
federated averaging collapsed the model into a **majority-class predictor**.
42.86% is simply OuterRace's share of the validation sets.

That is a more precise and more defensible statement than "a 17-point gap".

---

# PART 7 — The three ML bugs

All three **lowered** the numbers. That is the point.

| Bug | Before | After |
|---|---|---|
| **Data leakage** — split by window, so windows from one recording landed on both sides | ~100% (fake) | 53.8–69.5% (real) |
| **InnerRace unlearnable** — `AdaptiveAvgPool1d` destroyed the fault repetition rate | 100% misclassified as OuterRace, every site | Learnable via FFT input |
| **FedAvg instability** — oscillated under class imbalance | unstable | stable (but degenerate — see Part 6) |

A failed attempt worth citing: widening the CNN kernels while keeping
time-domain input made things **worse — 0.306 vs 0.527**. The FFT change was the
actual fix, not the architecture.

---

# PART 8 — The methodology that is actually the strongest story

A thesis panel has seen many bearing classifiers. What they have rarely seen is
a project that **systematically hunts for its own false claims**.

### Three tests were found to be incapable of failing

1. **`test_dependencies.py`** used a `sys.modules` diff — but another test had
   already imported `torch`, so the diff came back empty and it could never
   fail. Rewritten as a static import-graph walk.
2. **The Slice J animation check** sampled every 6 seconds for cues lasting 2–4
   seconds, and walked straight past them. Rewritten with a `MutationObserver`.
3. **`test_every_route_refuses_an_anonymous_caller`** — see below.

### The one that mattered (found 2026-10-05)

This FastAPI version does not flatten `include_router()` into `app.routes`; it
wraps each call in an internal `_IncludedRouter`. The test filtered for
`APIRoute`, so it saw **2 routes out of 34**, one of which was public.

**It had been probing exactly one endpoint — `/health` — since Slice H**, while
being cited in four documents as the project's strongest access-control
guarantee.

With the walk fixed it failed immediately, and `curl` confirmed a **real
unauthenticated leak**: `GET /service-requests/60` with no token returned HTTP
200 and the full record. It was the only handler in the file missing a user
dependency — missed precisely *because* the broken test reported full coverage.

Fixed, verified live (no token → 401, admin → 200, other customer → 404), and a
companion test now asserts the walk finds ≥30 routes so it cannot go vacuous
again.

### The rule this project now follows

> **A test that asserts "nothing is wrong" must also assert that it looked at
> something.**

Every reachability-style test here has a companion guard pinning a floor on what
it inspected. **If you remember one methodological point for the defence, make
it this one.**

### Other real finds

- A clean clone **could not start the application** — `langgraph` declared
  nowhere, `torch` extra-only, CI installing neither.
- The Frequency Lab initially cropped the spectrum at 1 kHz, which would have
  **hidden the dominant peak** at 1,382 Hz.
- `LOAD_RISK_FACTOR` was keyed by site id instead of motor load, so the
  heaviest-loaded site silently got the neutral default.

---

# PART 9 — What is real vs simulated

**Real:** the dataset, the classifier and its accuracy, the federated result
(including its failure), 4,367 predictions and decisions, the incremental
processing, the bearing physics, the LLM call and its audit trail, the access
control.

**Simulated, and labelled in the UI:**
- The four "sites" are four CWRU **motor load levels**.
- **Sensor arrival is a replay** over a static archive.
- **GPS fields are invented** — CWRU has no location data.
- `Equipment.linked_experiment_id` is a simulated association — the largest open
  modelling gap.

**Limitations:** GAN augmentation generates time-domain windows while the
classifier trains on spectra; defect-band corroboration is often weak and the UI
shows it; Compose has only ever run images built on 2026-09-10, before most of
this system existed (7 routes vs 31), so the current code is unverified under Docker;
auth is demo-grade (`localStorage`, no refresh rotation).

---

# PART 10 — Verification status

| Check | Result |
|---|---|
| Backend tests | **260 passing** across 20 files |
| ruff / format / mypy | clean (64 source files) |
| tsc / production build | clean (5 static routes) |
| docker compose config | resolves (5 services) |
| Migrations from empty | all 6, against real PostgreSQL |
| Browser walkthrough | 15 page-visits, 5 roles, zero console errors |

Largest test files: `test_auth.py` (39), `test_llm_narrative.py` (29),
`test_product_api.py` (29), `test_sync_equipment_status.py` (20).

---

# PART 11 — Likely questions

**"Why only 60%?"**
> Because it is measured honestly, twice over. The first version reported ~100% —
> data leakage from splitting by window instead of by recording. Fixing that
> dropped it to the 62–72% range read from `metrics.json`. Then we found that
> `metrics.json` records the best *epoch*, while `train.py` saves the *last* one —
> so even those figures described models that were never on disk. The deployed
> checkpoints measure 53.8–69.5%, mean 60.4%. I would rather defend 60% that is
> real, and name the 50% floor it has to clear, than quote a number twice removed
> from the model that is actually running.

**"Why is federated learning worse?"**
> It is, and that is the finding. Per-class recall shows the global model
> collapsed to predicting OuterRace for everything — 42.86% is just OuterRace's
> share. That is the documented behaviour of federated averaging under non-IID
> class imbalance, and it is why FedProx is reported rather than FedAvg, which
> did not even converge stably.

**"How is there frequency data if the dataset is static?"**
> Only vibration amplitude over time was ever recorded. Frequency is calculated
> from that same recording by Fourier Transform — like describing a chord as
> notes instead of as air pressure. Same information, different description.
> *(Full answer: `FREQUENCY_EXPLAINED_FOR_DEFENSE.md`.)*

**"Does the model use the bearing defect frequencies?"**
> No — and that is checkable by grep. BPFI/BPFO/BSF/FTF appear in five files,
> none of which are `model.py`, `train.py`, `predict.py` or `dataset.py`. The
> classifier only ever sees a generic 512-bin spectrum, which is exactly why the
> frequency check counts as independent verification.

**"Is this production-ready?"**
> No, and I would not claim it. GPS is simulated, the replay is not live
> telemetry, auth is demo-grade. What it demonstrates is the full path from raw
> vibration to a dispatched technician, with evidence attached at every step.

**"What would you do next?"**
> Fix the GAN domain mismatch — it generates time-domain windows while the
> classifier trains on spectra. Then model sites properly rather than as load
> levels, and calibrate the classifier's confidence, which the weak-corroboration
> finding shows is overconfident.

---

# PART 12 — The 60-second version

> FleetMind monitors industrial machines for bearing faults. Real CWRU vibration
> data, 4,367 windows. A 1D CNN on FFT-magnitude input classifies the fault at
> 60.4% mean accuracy over three classes, against a 50% majority-class floor —
> measured on the checkpoints actually deployed. A LangGraph agent chain turns
> that into a risk score
> and an urgency, which automatically flips a machine to maintenance and files a
> service request an operator dispatches against. Every decision carries evidence
> — raw signal, spectrum, the bearing physics that should corroborate it, and the
> dataset's ground truth beside the model's independent prediction. An LLM writes
> the technician's summary but cannot originate a number: it only calls
> deterministic tools, and a guard catches any figure it invents. It is
> multi-tenant with three roles enforced server-side, and the pipeline runs
> incrementally through a replay engine over the archive. Federated learning
> across four sites underperformed local training — it collapsed to a
> majority-class predictor — and I report that as a finding. Along the way I found
> three tests that could not fail, one of which was hiding a real unauthenticated
> data leak.

---

*Compiled 2026-10-05 from all 53 commits and the live system. 260 tests passing.*
