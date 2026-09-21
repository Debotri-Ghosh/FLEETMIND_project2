# Slice A — Harden Phase 5

Goal: verify Phase 5 (product-layer routers: `equipment`/`service-requests`/
`work-orders`/`customers`/`technicians`; customer/operator Next.js pages) actually
works end to end, add real test coverage, fix deployment scaffolding, then close
the edge cases and error handling a happy-path demo pass doesn't touch.

## Done already — re-verify, don't redo

- [x] **Task 1**: Node.js installed, `npm install` run, backend + frontend dev
  servers booted, all three pages (`/`, `/customer`, `/operator`) driven with a
  headless-Chromium Playwright script, the customer service-request form
  exercised end to end. Commit `a7395fdf`.
  Re-verify: boot both servers, re-run the Playwright check script (or an
  equivalent), confirm zero console errors and zero failed/4xx/5xx requests.
- [x] **Task 2**: 24 pytest tests added for the product-layer routers and
  `sync_equipment_status`'s aggregation logic (`_worst_urgency`, retired-equipment
  exclusion, dry-run, idempotency). Commit `afa7bf3f`.
  Re-verify: `cd backend && ../venv/Scripts/python.exe -m pytest -q` — expect all
  passing (as of the Slice D pass, 89/89 with the 2 previously-known federated
  test bugs also fixed — see `04-slice-d-deployment.md`).
- [x] **Task 3**: Removed six accidental duplicate scaffolding files/dirs from
  `frontend/` (`docker-compose.yml`, `.env.example`, `.gitattributes`,
  `.gitignore`, `README.md`, `.github/workflows/ci.yml` — all byte-identical to
  the real root-level files) plus a stale duplicate `frontend/backend/` tree.
  Commit `627fc882`.
  Re-verify: `docker compose config` from the repo root resolves both build
  contexts correctly; `frontend/` has no stray `backend/`, `.github/`,
  `docker-compose.yml`, etc.

## Hardening — do this now

The routers and pages above were built and tested for the happy path. This pass
closes the gaps a real user (or a flaky network, or a double-click) would hit.

### Backend: reject bad input cleanly instead of a raw 500

- [x] `POST /service-requests` (`api/routers/service_requests.py`): the current
  implementation does a raw `insert()` with whatever `customer_id`/`depot_id`/
  `equipment_id` the client sends, with no existence check. An invalid
  `customer_id` or `depot_id` currently either violates a DB foreign-key
  constraint (unhandled `IntegrityError` -> FastAPI 500) or, if the columns
  aren't enforced at the DB level for some reason, silently creates an orphaned
  row. Add an explicit existence check for `customer_id`, `depot_id`, and
  `equipment_id` (if given) before the insert, returning `404` with a clear
  `detail` message naming which reference is invalid. Add tests: invalid
  customer_id -> 404, invalid depot_id -> 404, invalid equipment_id -> 404,
  valid equipment_id from a *different* depot than the one given -> decide and
  document the behavior (reject as inconsistent, or allow — pick one, note the
  reasoning in a code comment, this is a technical call, not a product one).
- [x] `POST /work-orders` (`api/routers/work_orders.py`): already checks the
  `service_request_id` exists, but not `technician_id`. Add the same existence
  check, `404` on missing technician. Also: `WorkOrder` has a unique constraint
  on `service_request_id`, so creating a second work order for an
  already-dispatched request currently raises an unhandled `IntegrityError` ->
  500. Catch that specific case *before* hitting the DB (query for an existing
  work order on that service_request_id first) and return `409 Conflict` with a
  message pointing at the existing work order's id. Add tests for both: missing
  technician -> 404, double-dispatch -> 409.
- [x] `PATCH /work-orders/{id}/status`: no validation that the transition makes
  sense (e.g. `completed` -> `assigned` backwards, or setting the same status
  twice). Decide: is this a real bug worth blocking, or intentionally permissive
  for manual correction? Reasoned default: **allow any transition** (ops need to
  correct mistakes without a support ticket) but add a test documenting the
  current permissive behavior explicitly, so a future change here is a
  conscious decision, not an accidental regression.
- [x] `GET` list endpoints with a `limit` query param (`equipment`,
  `service-requests`, `work-orders`): confirm `limit=0` and negative `limit`
  are rejected by FastAPI/Pydantic's `Query(default=100, le=500)` — add a `ge=1`
  bound and a test for `limit=0` -> 422.

### Frontend: handle the network/empty/error paths, not just the happy path

- [x] `frontend/src/app/customer/page.tsx`: the "File a Service Request" form
  has no client-side required-field validation beyond the `<select>`'s native
  `disabled` placeholder — an empty title currently either silently fails
  server-side (schema requires `title: str`, so FastAPI returns 422) or, worse,
  submits successfully with an empty string if the API doesn't reject it. Check
  what actually happens today, then: add a `required` attribute + a visible
  inline error if the mutation fails (currently a failed `createServiceRequest`
  mutation has no error UI at all — check `useMutation`'s `isError` state and
  show something better than a silent no-op).
- [x] Same page: after the FK-validation hardening above lands on the backend,
  the equipment dropdown could reference stale equipment (e.g. retired between
  page load and submit). Confirm the form surfaces the resulting 404 as a
  visible error rather than swallowing it.
- [x] `frontend/src/app/operator/page.tsx`'s `DispatchQueue`: dispatching with
  no technicians available (`technicians.data` empty) currently renders a
  `<select>` with only the disabled placeholder option and an always-enabled
  "Dispatch" button gated on `selectedTechnician[r.id]` being falsy — confirm
  this can't submit an invalid dispatch, and add a visible "No technicians
  available" message instead of a silently empty dropdown.
- [x] All three pages: `useQuery`'s `isError` branches currently render a bare
  one-line red message with no retry action. Add a "Retry" button (React
  Query's `refetch()`) to at least the top-level page-critical queries (depots,
  equipment board, service requests) so a transient network blip doesn't require
  a full page reload.
- [x] Confirm CORS/API-unreachable behavior end to end: stop the backend, load
  each page, confirm the existing `StatusBanner` / per-section error states
  degrade gracefully (no uncaught exceptions, no infinite loading spinners).
  This exercises the actual failure mode a demo laptop's flaky network would hit.

### Verification for this slice

- `cd backend && ../venv/Scripts/python.exe -m pytest -q` — all passing,
  including new edge-case tests above.
- `ruff check . && ruff format --check . && mypy fleetmind` — clean.
- `npx tsc --noEmit && npm run build` — clean.
- Boot both servers, re-run the Playwright browser check, **plus** a new pass
  that: submits the service-request form with the backend killed (network
  error path), attempts a double-dispatch via direct API call (`curl`) and
  confirms `409`, and posts a service request with a bogus `customer_id` and
  confirms `404`.

Record the exact commands and real output in `CLAUDE.md`'s Progress log, one
line per completed task, per the working rules in `00-overview.md`.
