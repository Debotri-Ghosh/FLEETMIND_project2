# ADR 0001 — Renaming `depots` to `sites`: migrate now vs. alias for one release

- **Status:** **Decided 2026-09-30 — Option D (defer), recorded 2026-10-02**
- **Date:** 2026-09-30
- **Context source:** product statement at the top of `CLAUDE.md`; evidence in
  `docs/audit.md` §3

## Context

The product statement establishes that FleetMind monitors machines installed in
**customers' factories**. The schema calls those places `depots`, a term the
statement explicitly supersedes.

The rename is **not cosmetic**, and this is the crux of the decision.
`docs/audit.md` §4.3 establishes that `depots` today does not model a place at
all — it models an **operating condition**:

```
depots.name        = 'depot_0hp' | 'depot_1hp' | 'depot_2hp' | 'depot_3hp'
depots.load_hp     = 0 | 1 | 2 | 3
depots.nominal_rpm = 1797 | 1772 | 1750 | 1730
```

These are the four CWRU motor load levels. A real customer site has none of
these properties inherently; a site could host machines at several load levels,
and two customers' sites could share one. So `depots -> sites` conflates two
changes:

1. **Lexical:** `depots` -> `sites`, `depot_id` -> `site_id` (3 FK columns, 3
   indexes, 1 route, N query params, 19 UI strings, `mlruns/depot_N/` paths).
2. **Semantic:** a site must become a real customer location, with load level
   demoted to a property of the *experiment* or the *machine*, not the place.

Doing (1) without (2) produces a table called `sites` whose rows are load
levels — which is worse than the status quo, because the name would then
actively lie rather than merely being legacy.

### Blast radius (measured, not estimated)

| Surface | Count | Breaking? |
|---|---|---|
| DB table/columns/indexes | 1 table, 3 FK columns, 3 indexes | needs migration |
| API routes + params + response fields | `/depots`, `depot_id`, `per_depot*` | **yes — public contract** |
| Python files referencing `depot` | 26 | internal |
| On-disk ML artifacts `mlruns/depot_N/` | 4 dirs, read by `stats.py` by literal path | needs data move |
| Frontend files / visible strings | 6 files / 19 strings | user-visible |

## Options

### Option A — Migrate now, single breaking change

One Alembic migration renames table, columns and indexes; all 26 Python files,
the API contract, the `mlruns/` layout and the UI change in one commit.

- **For:** one coherent state; no dual vocabulary; no compatibility code to
  remove later; the audit's rename list closes completely.
- **Against:** a hard break for any existing API consumer; large single diff
  touching ML, API and UI at once; `mlruns/` path move risks orphaning the real
  training artifacts that `/stats/model-performance` reads.

### Option B — Alias for one release

Add `sites` as the real table; keep `depots` as a SQL view (and `depot_id` as
accepted-but-deprecated API params) for one release; remove in the next.

- **For:** no consumer breaks immediately; rename can land in smaller pieces;
  reversible.
- **Against:** two names for one concept in the codebase for a whole release —
  precisely the ambiguity the product statement was written to end; the view
  cannot carry the *semantic* change (a view over load-level rows is still
  load-level rows); deprecation removal is a second scheduled piece of work that
  in practice often never happens.

### Option C — Split the change: semantic first, lexical second

1. Introduce `sites` as a **genuinely new** table (customer, name, address),
   move `equipment` to point at it, and demote load level onto `experiments`
   where it actually belongs.
2. Then retire `depots` — by then it has no rows that mean anything, so the
   "rename" is a deletion, not a migration.

- **For:** the only option that resolves the semantic conflation rather than
  renaming around it; each step is independently shippable and testable; no
  release ever contains a table whose name misdescribes its rows.
- **Against:** more total work; two migrations; `equipment.site_id` and
  `experiments.load_hp` must be backfilled, and the current 1:1
  depot↔load-level mapping means the initial backfill is a judgement call
  (one site per customer? one per load level, renamed?).

## Recommendation

**Option C**, with the backfill decision escalated to you.

Options A and B both assume the problem is the word. It isn't — §4.3 shows the
table's *rows* are load levels, so both A and B end with a table named `sites`
containing things that are not sites. C is the only option where the name is
true at every step. Its extra cost is real but bounded: one additional
migration and a backfill rule.

If the spec's Milestone 1 does not include customer-site modelling, then A is
preferable to B — take the single break, keep one vocabulary, and accept that
`sites` temporarily means "operating condition" with an explicit comment saying
so and a dated follow-up. **B is not recommended in any case**: a release-long
dual vocabulary is the specific cost the product statement exists to avoid, and
the alias cannot carry the semantic fix anyway.

## Decision

**Option D — defer the rename entirely; change only user-facing vocabulary.**

None of A, B or C was taken. When this ADR's question reached the project owner
at the start of Slice E, the instruction was explicit:

> "Keep the `depot` table/column names as-is (renaming is high-risk for low
> benefit) — only change what's user-facing or documentation."

So the schema, the API contract, the `depot_id` query parameters, the
`--depot-id` CLI flags and the `mlruns/depot_N/` artifact paths all keep the
legacy word, and the *vocabulary a human reads* became "site" everywhere:
headings, table columns, dropdowns, empty states, seed display names
(`depot_0hp` → `Plant A (CWRU 0 HP)`) and the customer/company names. Verified
on 2026-10-02: zero occurrences of depot/vehicle/logistics/transport language
in the UI, README or the live database's customer, site, equipment or
`agent_decisions.decision_text` rows.

### What this decision does *not* resolve

The analysis above still stands and is the reason this ADR is kept rather than
deleted: **`depots` rows are operating conditions, not places.** Deferring the
rename means the schema continues to model a CWRU motor-load level as if it
were a factory. That is an accurate description of the demo — four load levels
standing in for four sites — and it is disclosed as such in `CLAUDE.md` and the
README. It becomes a real problem only if the system ever needs a customer with
machines at more than one load level, or two customers sharing a site, neither
of which the thesis demo does.

If that day comes, **Option C remains the recommendation**: introduce a genuine
`sites` table and demote load level onto `experiments`, rather than renaming a
table into a name that misdescribes its rows.

## Consequences

Of Option D, as taken:

- `docs/audit.md` §3 is **closed as deferred**, not as done — the lexical list
  there remains accurate for whoever eventually does the rename.
- One vocabulary for humans, another for the schema. The mismatch is contained
  because the schema word is never rendered: a reviewer reading the UI sees
  "site" throughout, and a developer reading the code sees a comment in
  `db/models.py` explaining that `depot` is legacy for "site".
- `experiments` does **not** gain `load_hp`; load level stays on `depots`. The
  Slice G risk fix works around this by passing `depots.load_hp` explicitly
  through `AgentState` rather than deriving it from the site id — which is
  correct regardless of this ADR, but is also the first concrete place the
  modelling conflation cost real work.
- §4.1 (`linked_experiment_id` being a simulated association) is now the
  largest open modelling gap, as predicted.

For the record, had A or C been chosen: A would have closed §3 in one breaking
commit; C would have closed §3.1 in two steps and resolved §4.3 as a side
effect.
