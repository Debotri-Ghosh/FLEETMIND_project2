# Slice D — Deployment & hardening

Goal: reproducible and deployable, not adding capability. Get CI genuinely
green, fix deployment scaffolding, make sure the docs match reality.

## Done already — re-verify, don't redo

- [x] `.github/workflows/ci.yml`'s frontend job now uses `cache: npm` +
  `npm ci` (was `npm install` with caching disabled via a stale comment, from
  before `package-lock.json` existed). Commit `aae2e081`.
- [x] `ruff check .`, `ruff format --check .` clean across the whole backend (14
  files reformatted this pass — 7 new this session, 7 pre-existing Phase 5
  files that had never been formatted). Commit `aae2e081`.
- [x] `mypy fleetmind` clean (8 real type errors fixed in
  `sync_equipment_status.py` and `stats.py`). Commit `aae2e081`.
- [x] `pytest -q` fully green, 89/89 — including fixing the 2 previously-known
  pre-existing `test_federated.py` failures (a test-only bug:
  `BearingCNN(num_classes=2)` predated the class-imbalance fix and never got
  updated to match `class_weights()` always sizing to the full 4-class
  `LABEL_MAP`). Commit `aae2e081`.
- [x] `README.md` rewritten top to bottom to match reality: native-Postgres+venv
  path alongside Docker Compose, migration order (`base -> 0001 -> 0002`), seed
  order (`seed_product_tables.py` before the agent-sync bridge), the full
  ML/federated/agent pipeline commands, current real results, known honest
  limitations. Commit `aae2e081`.

Re-verify: `ruff check . && ruff format --check . && mypy fleetmind && pytest -q`
from `backend/`; `npx tsc --noEmit && npm run build` from `frontend/`.

## Not verifiable on this laptop — do not re-attempt without a different
## environment or explicit instruction

- `docker compose up` end to end: the Docker daemon is unreachable on this
  machine (`failed to connect to ... dockerDesktopLinuxEngine`), consistent
  with CLAUDE.md's note that this laptop's WSL2/Hyper-V is company-policy
  restricted. `docker compose config` confirms the compose file itself is
  correct (both build contexts resolve). Needs a machine with a working Docker
  daemon, or the CI `docker` job actually running via a real push.
- A genuinely fresh `alembic upgrade head` against an empty database: the local
  `fleetmind` Postgres role lacks `CREATEDB`, so a throwaway test DB can't be
  created here without a privilege change that wasn't requested. Verified
  migration *order* instead via `alembic history` (clean, non-branching
  `base -> 0001 -> 0002`). If a throwaway DB becomes available, run
  `alembic upgrade head` against it fresh and confirm no errors.

If you hit either of these tonight and still can't clear them, don't loop on
it — note it in the Progress log exactly as above and move to the next task.

## Hardening — do this now

### Configuration and startup error handling

- [ ] `fleetmind/core/config.py`'s `Settings` has hardcoded fallback defaults
  (e.g. `database_url: str = "postgresql://fleetmind:change_me@localhost:5432/
  fleetmind"`) — if `backend/.env` is missing entirely, the app currently starts
  up silently against that placeholder DB instead of failing fast with a clear
  error. Decide: is silent-fallback-to-placeholder acceptable for a dev/demo
  project, or should missing required env vars fail startup loudly? Reasoned
  default: **leave the fallback as-is** (this is a demo/thesis project, not a
  production service, and the fallback lets `pytest` run without any `.env` at
  all via the sqlite override in `conftest.py`) — but add one comment in
  `config.py` explaining this is intentional, not an oversight, so it isn't
  "fixed" into a startup-crash later by someone who doesn't know why it's there.
- [ ] `uvicorn fleetmind.api.main:app` with no `DATABASE_URL` reachable (e.g.
  Postgres not running) currently fails how? Check: does `/health` return a
  clean 500 with a readable error, or does the whole process crash on an
  unrelated request? `db.connection()` inside `/health` should surface a clear
  error either way — confirm by stopping Postgres, hitting `/health`, and
  reading the actual response/log output.

### CI robustness

- [ ] The `docker` job in `ci.yml` runs unconditionally after `backend`/
  `frontend` pass — since Docker builds can't be tested locally, at minimum
  confirm the Dockerfiles' `COPY`/`WORKDIR` paths match the actual project
  layout by reading them against the current `backend/`/`frontend/` directory
  contents (not just trusting they were right when written before Phase 5's
  product-layer files existed). Look specifically for anything that would
  break now that `backend/fleetmind/agents/sync_equipment_status.py`,
  `backend/fleetmind/db/seed_product_tables.py`, etc. exist as new files/
  directories the Dockerfile's `COPY . .`-style instructions need to pick up.
- [ ] Confirm `.dockerignore` (both `backend/.dockerignore` and
  `frontend/.dockerignore`) don't accidentally exclude anything Phase 5 added
  (e.g. a blanket `tests/` or `*.py` exclusion that predates the new routers).

### Verification for this slice

- Full local CI-equivalent pass: `ruff check .`, `ruff format --check .`,
  `mypy fleetmind`, `pytest -q` (backend); `npx tsc --noEmit`, `npm run build`
  (frontend) — all green, real output shown.
- `docker compose config` from repo root — resolves cleanly.
- Read-through verification of both Dockerfiles against current directory
  contents (can't build-test without a daemon, but can catch a stale `COPY`
  path by inspection).

Record the exact commands and real output in `CLAUDE.md`'s Progress log.

## Wrap-up (once all four slices are hardened)

- [ ] Sweep `OPEN_QUESTIONS.md` (if any entries were added during this pass) —
  confirm each has a recorded default that was actually applied, not left as a
  dangling question.
- [ ] One final full-repo verification pass: backend lint/format/type/test,
  frontend type-check/build, a live browser check of all three pages, and a
  live `curl` smoke test of every new/changed endpoint from this pass.
- [ ] Final `git log --oneline` review to confirm every task in this plan that
  got marked done has a corresponding commit, and `CLAUDE.md`'s Progress log
  has a matching line for each.
