# Slice C — Unified demo dashboard

Goal: one dashboard view showing the ML layer (per-depot and federated
accuracy, fault distribution, signal viewer) and the product layer (equipment
health, open requests, work orders) together, for a committee demo.

## Done already — re-verify, don't redo

- [x] `GET /stats/model-performance`: reads `mlruns/depot_{1-4}/metrics.json` +
  `mlruns/federated/fedprox/federated.json`, returns per-depot best_val_acc plus
  federated mean/per-depot final accuracy; gracefully returns `per_depot: []`,
  `federated: null` if those files are missing (they're gitignored, so a fresh
  clone won't have them). Commit `26845cee`.
- [x] `GET /stats/product-summary`: equipment-by-status, open-requests-by-
  priority, AI-flagged-open count, active-work-orders count — all DB
  aggregates. Commit `26845cee`.
- [x] Dashboard home page (`frontend/src/app/page.tsx`): new "System Overview"
  KPI row and "Model Performance: Local vs. Federated" grouped bar chart
  (`#2563eb` local baseline / `#eb6834` FedProx, palette validated with the
  dataviz skill's `validate_palette.js`), above the existing depot table / fault
  distribution / signal viewer. Commit `26845cee`.
- [x] 5 backend tests covering both endpoints, including missing-file and
  empty-DB cases. Commit `26845cee`.

Re-verify: `curl http://127.0.0.1:8000/stats/model-performance` and
`.../stats/product-summary` against the real running backend, confirm the
numbers match `CLAUDE.md`'s documented real results (depot val acc 62.7-72.5%,
FedProx mean baseline 60.0% vs mean final 42.9%).

## Hardening — do this now

### Backend: don't let one bad file take down the whole endpoint

- [x] `model_performance()` currently does `json.loads(metrics_path.read_text())`
  with no error handling — a truncated or corrupted `metrics.json` (e.g. from a
  training run that crashed mid-write) would throw `json.JSONDecodeError` and
  500 the *entire* endpoint, hiding the other three depots' good data. Wrap each
  file read in a try/except, skip (and log a warning with the path) on failure
  rather than crashing the whole response. Same for the federated.json read.
  Add a test: one corrupted depot metrics file among four valid ones -> the
  endpoint still returns the three valid depots, 200, not 500.
- [x] `model_performance()` assumes `m["best_val_acc"]`, `m["train_windows"]`,
  `m["val_windows"]` are always present — a metrics.json from an older/different
  training run shape could be missing a key and raise `KeyError` -> 500. Use
  `.get()` with a sensible fallback (or skip the depot, matching the corrupted-
  file handling above) and add a test for a metrics.json missing one of these
  keys.
- [x] `federated["final"]` is assumed to be a dict of `{depot_id: {"accuracy":
  ...}}` — a federated.json from a run that failed before producing a `final`
  block would raise `KeyError` on `f["final"]`. Guard with `.get("final", {})`
  and confirm `per_depot_final_accuracy` comes back `{}` rather than crashing.

### Frontend: partial-data and loading-state polish

- [x] `ModelAccuracyChart`: confirm the chart renders sensibly when `federated`
  is `null` but `per_depot` has data (single-series chart, no "Federated
  (FedProx)" bar or legend entry) — this path exists in the code
  (`{federated && <Bar .../>}`) but has not been screenshot-verified; do that.
- [x] `ProductKpiRow` / `ModelAccuracyChart`: both show a bare "Loading…" text
  line while fetching. Not a bug, but confirm it doesn't flash an
  empty/zero-value state before the real data arrives (i.e. `isLoading` is
  checked before rendering zeros) — verify by throttling the network in the
  browser check and screenshotting the loading state.
- [x] Numeric formatting: `gain_vs_baseline` is already correctly negative in
  the real data (federated underperforms baseline) — confirm the dashboard
  doesn't imply this is an error (e.g. no red "down" styling that reads as
  broken) since CLAUDE.md is explicit this is a documented, expected finding,
  not a bug. Check the caption text under the chart still reads clearly if
  `gain_vs_baseline` were ever positive (i.e. it's not hardcoded to only make
  sense for a negative gap).

### Verification for this slice

- `cd backend && ../venv/Scripts/python.exe -m pytest -q` — all passing,
  including new corrupted/partial-file tests.
- `npx tsc --noEmit && npm run build` — clean.
- Browser-verify the dashboard with a temporarily corrupted
  `mlruns/depot_2/metrics.json` (rename it, or point `MLRUNS` at a fixture dir
  with 3 good files + 1 bad one) and confirm the page still renders the other
  three depots without a full-page error.

Record the exact commands and real output in `CLAUDE.md`'s Progress log.

## Completed — 2026-09-22

All items above are done. Commits `45d1c478` (backend: `_load_json_safe`,
per-key `KeyError` handling, `federated["final"]` guard, 7 new tests) and
`ac21b5af` (progress log). Frontend verification (no code changes needed,
just confirmation) was done via Playwright route interception: mocking
`/stats/model-performance` with `federated: null` shows a clean single-series
chart with no "Federated (FedProx)" legend entry and no crash; delaying both
stats responses by 3s and screenshotting mid-fetch confirms `isLoading` is
checked before any zero-value tile renders (no premature "0" flash). The
`gain_vs_baseline` caption text (`text-sm text-slate-500`, no red/error
styling) reads as "a real, documented gap ... not a bug" regardless of sign —
confirmed by inspection, no hardcoded negative-only wording. Live end-to-end
check against the real backend: temporarily truncated the real
`mlruns/depot_2/metrics.json` to invalid JSON, confirmed
`GET /stats/model-performance` still returned depots 1/3/4 with `200` (not
500), then restored the original file from a backup and confirmed all 4
depots return correctly again via a follow-up request.
