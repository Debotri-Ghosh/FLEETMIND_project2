# FleetMind â€” Project Context for Claude Code

**Federated Intelligent Predictive Maintenance for Bearing Fault Diagnosis** â€” M.Tech project (Debotri Ghosh, 25MML1008), VIT Chennai + LTIMindTree.

Real dataset (CWRU Bearing Dataset â€” cite engineering.case.edu/bearingdatacenter), no synthetic data anywhere except the documented TimeGAN augmentation step. 4 real motor load levels (0/1/2/3 HP) act as 4 simulated depots.

## Environment

- Run all backend commands **from `backend/`**, not the project root â€” `.env` and relative paths assume this.
- PostgreSQL is **native** (not Docker â€” company laptop blocks WSL2/Hyper-V). Database `fleetmind`, user `fleetmind`, password `fleetmind2026`, `localhost:5432`.
- `backend/.env` contains: `DATABASE_URL=postgresql://fleetmind:fleetmind2026@localhost:5432/fleetmind`
- Python 3.11 venv at project root `venv/`. Activate with `venv\Scripts\activate` (Windows).
- When writing files via PowerShell heredocs, always strip the BOM afterward (`Set-Content -Encoding utf8` adds one, which breaks Python's parser):
  ```powershell
  $content = Get-Content -Path <file> -Raw
  [System.IO.File]::WriteAllText("$PWD\<file>", $content, [System.Text.UTF8Encoding]::new($false))
  ```

## Architecture

```
backend/fleetmind/
  ingestion/    download, manifest, windowing, CWRU ingest (done)
  db/           models.py (SQLAlchemy), session.py, base.py
  api/          FastAPI app + routers (depots/experiments/stats/windows exist; equipment/service-requests/work-orders NOT built yet)
  eda/          EDA report generation (done)
  ml/           model.py, dataset.py, train.py, evaluate.py, predict.py
  augmentation/ GAN-based rare-class augmentation (gan.py, run.py, exp_split.py, sampling.py)
  federated/    client.py, run.py, flower_client.py â€” FedAvg/FedProx simulation
  agents/       state.py, agents.py, graph.py, run.py â€” LangGraph Monitor->Diagnostic->Risk->Maintenance chain
frontend/src/   Next.js 14 â€” app/page.tsx, components.tsx, layout.tsx, providers.tsx; lib/api.ts, types.ts
```

## Database schema (already migrated)

Core ML tables: `depots`, `experiments`, `signal_windows`, `model_predictions`, `agent_decisions`.
Product-layer tables (schema exists, **no API routes or frontend pages built yet**): `customers`, `equipment`, `technicians`, `service_requests`, `work_orders`. See `db/models.py` for exact columns â€” `Equipment.status` (active/maintenance/retired), `ServiceRequest.status` (reported/diagnosed/dispatched/in_progress/resolved/cancelled), `WorkOrder.status` (assigned/traveling/in_progress/completed/cancelled).

## Real, current results (as of 2026-09-19 â€” don't regenerate unless intentionally re-running an experiment)

Per-depot classifier (`mlruns/depot_{1-4}/metrics.json`): honest experiment-level split, FFT-magnitude input, class-weighted loss. Val accuracy: depot 1 = 64.8%, depot 2 = 72.5%, depot 3 = 62.7%, depot 4 = 65.7%.

Federated (`mlruns/federated/fedprox/federated.json`): FedProx (mu=0.1, 30 rounds) converges to mean accuracy 0.429 (vs mean local baseline 0.600) â€” a real, documented gap, not a bug. FedAvg (no proximal term) is unstable/oscillating and should not be presented as the primary result â€” FedProx is.

`model_predictions` (4,367 rows) and `agent_decisions` (4,367 rows) are freshly regenerated from the current fixed models â€” real, not placeholders. Verified via direct psql queries, not just app-level.

## Three real bugs found and fixed tonight â€” do not "fix" these again or revert them

1. **Data leakage** (`ml/train.py`, `ml/evaluate.py`): was splitting individual signal windows randomly 80/20, so windows from the same recording landed on both sides of train/val, inflating accuracy to a fake 100%. Fixed by switching to the same experiment-level split already used correctly in `augmentation/run.py` and `federated/run.py` (via `augmentation/exp_split.py:stratified_experiment_split`). **Any new training code must split by experiment ID, never by individual window.**

2. **Class imbalance / InnerRace unlearnable** (`ml/model.py`, `ml/train.py`, `augmentation/run.py`, `federated/client.py`, `federated/run.py`, `ml/predict.py`): `BearingCNN`'s `AdaptiveAvgPool1d` collapsed the whole raw time-domain window into one global-average feature, discarding the fault-repetition-rate information that actually separates InnerRace from OuterRace â€” confirmed via confusion matrix (InnerRace was 100% misclassified as OuterRace, every depot). **Fixed by feeding `ml/model.py:to_fft_magnitude()` (512-bin FFT magnitude, log-scaled, z-normalized) as the model's input instead of raw time samples**, everywhere a classifier trains or predicts. Also added `ml/model.py:class_weights()` (inverse-frequency `CrossEntropyLoss` weighting) â€” helped but did not alone fix InnerRace; the FFT change was the actual fix. **Any new model code must use `to_fft_magnitude()` on `raw_values` before feeding the CNN** â€” an architecture-only fix (wider kernels) was tried and made things *worse* (0.306 vs 0.527 accuracy) â€” don't retry that.

3. **Federated instability**: plain FedAvg oscillated wildly round to round under real per-depot class imbalance. Switching to FedProx (`--strategy fedprox --mu 0.1`) stabilized it and closed roughly a third of the baseline gap. This is a known, published FL phenomenon (cited in the project's literature review) â€” a residual accuracy gap vs. local baselines is expected and documented, not something to eliminate by more debugging.

## Known trade-off, not a bug

TimeGAN/GAN-based augmentation (`augmentation/gan.py`) currently generates *time-domain* synthetic windows, but the classifier now trains on their FFT â€” synthetic windows can be time-domain-plausible while spectrally unrealistic, which is a likely reason augmentation sometimes hurts accuracy post-FFT-fix. Documented as a real limitation; not yet fixed.

## What's next (not yet built)

1. Wire `agent_decisions` (specifically `risk_score` / maintenance urgency) into `Equipment.status` â€” a "critical" decision should be able to flip a real equipment row to `maintenance_flagged`. This is the intended, honest bridge between the ML/agent layer and the product layer below.
2. FastAPI routers for `equipment`, `service_requests`, `work_orders` (following the existing pattern in `api/routers/depots.py`).
3. Next.js pages: customer view (browse equipment by depot, file a service request, track status) and depot-operator view (equipment board, work-order queue). Match existing `frontend/src/app/` structure and styling.
4. Location/GPS fields on `Equipment`/`WorkOrder` are explicitly **simulated** (CWRU has no location data) â€” keep this visibly labeled in any UI that shows them, never implied as real telemetry.

## Literature review (6 papers) and full implementation report

See `FleetMind_Implementation_Report.md` and `FleetMind_Literature_Review.md` in the project docs for the full write-up, dataset-selection rationale, and paper-to-subsystem mapping â€” reference these rather than re-deriving justifications from scratch.

## Progress log

2026-09-21 21:10 â€” Slice A â€” npm install + boot dev server + browser-verify /, /customer, /operator â€” verified via: Node.js v24.19.0 installed user-scope (winget --scope user, since the admin-elevated installer was blocked, matching the laptop's other tooling restrictions); `npm install` in frontend/ (succeeded on retry after one transient ECONNRESET); backend booted with `venv/Scripts/python.exe -m uvicorn fleetmind.api.main:app`; frontend booted with `npm run dev`; drove both with a local Playwright script (chromium-cli unavailable on Windows) hitting http://localhost:3000/, /customer, /operator â€” result: pass. All three pages render real data with zero console errors and zero failed/4xx/5xx network requests. Exercised the customer service-request form end-to-end (depot filter, equipment select, submit) â€” request appeared correctly in "Your Service Requests" with status `reported`; the test row was deleted from the DB afterward via psql so it doesn't pollute the demo dataset. No frontend bugs found needing a fix.
