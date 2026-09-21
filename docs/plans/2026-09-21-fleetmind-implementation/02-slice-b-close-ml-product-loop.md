# Slice B — Close the ML -> product loop

Goal: when the agent pipeline flags equipment as needing maintenance, that
decision should be visible to a depot operator as more than a status badge —
with the actual fault type, risk score, and decision text behind it — not just
a silent `Equipment.status` flip.

## Product decision already made (do not re-litigate)

**Auto-create vs. suggest-only**: the user chose **auto-create** — when equipment
computes to `maintenance`, `sync_equipment_status` files a real `ServiceRequest`
automatically (not a suggestion the operator has to click to create), guarded by
an idempotency check so re-runs don't duplicate it. This is already implemented
and is the standing default for any related decision in this slice.

## Done already — re-verify, don't redo

- [x] `sync_equipment_status` now auto-files an AI-sourced `ServiceRequest`
  (status=`reported`, priority = the triggering urgency lower-cased,
  `agent_decision_id`/`prediction_id` set to the worst-ranked decision) for any
  equipment computing to `maintenance`, guarded by an "already has an open
  AI-filed request" check (idempotent re-runs; refiles if a previous AI request
  was resolved/cancelled while the risk persists). Commit `2f9d0b03`.
- [x] `service-requests` list/get endpoints LEFT JOIN through
  `agent_decision_id -> prediction_id` to surface `agent_decision_text` /
  `agent_risk_score` / `predicted_fault` (null for customer-filed requests), and
  support `?ai_flagged=true|false`. Commit `2f9d0b03`.
- [x] Operator page has a distinct "AI-Flagged Requests" panel using the above,
  separate from the unified "Awaiting Dispatch" action queue. Commit `2f9d0b03`.
- [x] 16 tests covering the filing logic, idempotency, refiling, and the API
  join/filter. Commit `2f9d0b03`.

Re-verify: `cd backend && ../venv/Scripts/python.exe -m pytest -q
tests/test_sync_equipment_status.py tests/test_product_api.py`; then
`python -m fleetmind.agents.sync_equipment_status` against the real DB and
confirm `0 equipment updated, 0 AI service requests filed` (already idempotent
from the first run this session).

## Hardening — do this now

### Concurrency and partial-failure edge cases

- [x] **Race condition**: two concurrent `sync_equipment_status` runs (e.g. a
  cron job overlapping a manual run) could both pass the "already has an open
  AI-filed request" check before either commits, filing two AI requests for the
  same equipment. There's no unique constraint preventing this today. Reasoned
  default: add a partial unique index (e.g. on `service_requests(equipment_id)`
  where `agent_decision_id IS NOT NULL AND status NOT IN ('resolved',
  'cancelled')`) via a new Alembic migration (`0003_...`), and catch the
  resulting `IntegrityError` in `_ensure_ai_service_request` as a no-op (someone
  else already filed it, that's fine) rather than crashing the whole sync run.
  This is a technical hardening call, not a product one — proceed with it.
  Write a test that simulates the race (two sessions, or a mocked duplicate
  insert) and confirms only one request survives and the sync run doesn't crash.
- [x] **Malformed/missing decision data**: an `AgentDecision.decision_text` that
  doesn't start with a recognized `[URGENCY]` tag is already handled
  (`_worst_urgency`/`_worst_decision` skip it) — add a test with *only*
  untagged decisions for an equipment's experiment and confirm it correctly
  stays `active` / never files a request (no test currently covers the
  "linked but zero usable decisions" case explicitly, only "zero decisions at
  all" implicitly).
- [x] **Null `risk_score`**: `AgentDecision.risk_score` is nullable in the
  schema. Confirm `agent_risk_score` in the API response is `null` (not a
  crash) when the underlying decision has no risk score, and that the frontend
  `AiFlaggedPanel` renders `—` instead of `NaN%` or a blank cell in that case
  (check the `r.agent_risk_score !== null` guard already there — add a test/
  screenshot confirming it, since it was written defensively but never
  exercised with an actual null value).
- [x] **Dispatching an AI-flagged request twice**: once Slice A's work-order
  double-dispatch hardening (409) lands, confirm it also covers AI-filed
  requests specifically (they go through the same `create_work_order`
  endpoint, so this should be free, but add one test that dispatches an
  AI-flagged request twice and confirms 409, not a 500).
- [x] `sync_equipment_status.main()`'s CLI currently has no error handling
  around DB connectivity — if Postgres is down mid-run, confirm the failure is
  a clear, non-corrupting error (transaction rolls back cleanly, no partial
  commits) rather than leaving some equipment updated and others not. This is
  mostly a "confirm SQLAlchemy's session semantics already handle it" check —
  add a test that forces an error mid-loop (e.g. monkeypatch the second
  equipment's query to raise) and confirms `session.rollback()` in the caller
  leaves the DB unchanged from before the run, or document that partial commits
  are acceptable (decide and record the reasoning either way).

### Frontend resilience

- [x] `AiFlaggedPanel` and `DispatchQueue` on the operator page each fetch
  `service-requests` independently (different `ai_flagged` filters) — confirm
  one query failing (e.g. backend briefly down) doesn't blank both panels
  identically without indication of which one is affected; each `isError`
  branch should be scoped to just that panel (check this is already true from
  the component structure — it should be, since they're separate `useQuery`
  calls, but verify in the browser by simulating one failing).

### Verification for this slice

- `cd backend && ../venv/Scripts/python.exe -m pytest -q` — all passing.
- Live: run `sync_equipment_status` twice in a row against the real DB, confirm
  idempotent; if the race-condition migration lands, verify
  `alembic upgrade head` applies it cleanly and `alembic downgrade -1` reverts
  it cleanly (both real commands, real output).
- Browser-verify the operator page's AI-Flagged panel and Awaiting Dispatch
  queue with the backend intentionally killed and restarted, confirming no
  uncaught console errors either way.

Record the exact commands and real output in `CLAUDE.md`'s Progress log.

## Completed — 2026-09-22

All items above are done. Commits: `0b255e8c` (race-condition guard, null
risk_score, double-dispatch, main() rollback — the bulk of this checklist,
salvaged from a subagent that was cut off by a rate limit mid-task but had
already left complete, passing, uncommitted work in the tree; the parent
session independently re-ran the full verification before committing it,
rather than trusting the cut-off agent's own unfinished claim), `546ae4ad`/
`fbd2703f` (progress log entries). The frontend-resilience item and the
migration downgrade/upgrade cycle (both left unverified by the cut-off agent)
were completed directly by the parent session: a Playwright script that
blocks only the `ai_flagged=true` request confirms `AiFlaggedPanel` shows its
own error+retry state while `DispatchQueue` keeps rendering real data
unaffected, no uncaught exceptions; `alembic downgrade -1` against the real
dev DB cleanly drops `uq_service_requests_open_ai_request` (confirmed absent
via `psql \d service_requests`) and `alembic upgrade head` cleanly recreates
it (confirmed present again).
