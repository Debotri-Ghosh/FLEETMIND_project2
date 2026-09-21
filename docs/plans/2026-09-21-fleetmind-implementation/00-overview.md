# FleetMind implementation plan — 2026-09-21

Vertical-slice plan for finishing the product layer on top of the already-shipped
ML/agent pipeline (Phases 0-5: ingestion, per-depot CNN, WaveGAN-lite augmentation,
FedAvg/FedProx federated learning, LangGraph agent chain — all done and committed
before this plan started). Each slice cuts through DB -> ML/agent logic -> API ->
UI -> verification, so every slice ends with something demoable.

## Slices

| # | File | Goal |
|---|---|---|
| A | `01-slice-a-harden-phase5.md` | Verify Phase 5 (product-layer routers, customer/operator UI) actually works, add test coverage, fix deployment scaffolding, then harden edge cases/error handling |
| B | `02-slice-b-close-ml-product-loop.md` | Auto-file AI service requests from agent decisions, surface them distinctly in the operator UI, harden the sync bridge |
| C | `03-slice-c-unified-dashboard.md` | Merge the ML dashboard with product-layer KPIs and real federated/per-depot accuracy numbers |
| D | `04-slice-d-deployment.md` | Get CI green, fix docker-compose, rewrite the README, close remaining deployment/error-handling gaps |

## Status as of 2026-09-22 (first pass, same session that wrote this plan)

Slices A-D's *original* task lists (as scoped when this plan was first discussed,
before it was written down as these files) are already implemented, verified with
real commands, and committed — see `CLAUDE.md`'s Progress log for the exact
verification commands and output for each. Commits, in order:

- `a7395fdf` Slice A Task 1 — product-layer routers, ML->equipment sync, customer/operator UI, browser-verified
- `afa7bf3f` Slice A Task 2 — pytest coverage for product-layer routers + sync_equipment_status
- `627fc882` Slice A Task 3 — remove accidental duplicate scaffolding from frontend/
- `2f9d0b03` Slice B — auto-file AI service requests + operator AI-Flagged Requests panel
- `26845cee` Slice C — unified demo dashboard (ML + product KPIs + federated accuracy)
- `aae2e081` Slice D — get CI green (ruff/format/mypy/pytest) + README rewrite

**This plan's job now** is the pass requested on top of that baseline: closing
frontend/backend edge cases and error handling so the product layer holds up
under real misuse (bad input, missing data, concurrent requests, partial
failures), not just the happy path that got it demo-ready. Each slice file below
has a "Done already" section (do not redo — re-verify with the listed command,
then move on) and a "Hardening — do this now" section (the actual work for this
pass).

## Working rules for whoever (agent or human) executes this plan

1. **Re-read `CLAUDE.md`'s Progress log before starting any task in any slice.**
   Do not redo finished work. Do not trust a "done" claim you have not personally
   re-verified this session — re-run the verification command and read its real
   output before building on top of that task.
2. **Every task**: implement -> run the real verification command for it (test
   suite, dev server boot, actual `docker compose up`, a live `curl`, whatever
   applies) -> only mark it done if that command's real output shows a pass.
   Never mark something done because it "looks right."
3. **Commit after every completed task**, message naming the slice + task.
4. **Append one line to `CLAUDE.md`'s Progress log after every commit**, in the
   format already established there.
5. **Genuine product/design decisions** (not technical ones) go in
   `OPEN_QUESTIONS.md` at the repo root, with your reasoned best-guess default —
   then proceed with that default. Do not stop and wait to be asked.
6. **Do not touch** anything under CLAUDE.md's "Three real bugs found and fixed
   tonight" section without re-reading that section first, and do not revert or
   re-"fix" what it describes.
7. If you run out of budget mid-task: commit whatever is in a working state,
   write exactly what's unfinished into the Progress log (and into the relevant
   slice file's checklist below), and leave the repo in a clean, committed state
   — never a broken or half-edited one.
