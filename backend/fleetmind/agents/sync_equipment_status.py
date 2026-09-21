"""Sync Equipment.status from the agent_decisions produced for its linked experiment.

For each equipment row with a linked_experiment_id, this looks at every
agent_decision recorded for windows under that experiment (chain:
experiment -> signal_windows -> model_predictions -> agent_decisions) and
takes the worst (max-severity) urgency among them, read from decision_text's
own leading "[URGENCY]" tag (agents.py's source of truth, not re-derived
thresholds). Any HIGH or CRITICAL decision flips the equipment to
'maintenance'; otherwise it's set to 'active'. Equipment already 'retired'
is never touched — that's a manual, permanent state.

Idempotent and bidirectional: re-running can both escalate and clear a flag
as agent_decisions change.

Whenever an equipment computes to 'maintenance', this also ensures there is
an open, AI-filed ServiceRequest for it (agent_decision_id set, status not
resolved/cancelled) — carrying the triggering agent_decision_id/prediction_id
so the operator UI can show the actual fault type/risk score/decision text
instead of just a status badge. This is checked on every run (not only on
the active->maintenance transition), so if an operator resolves/cancels the
AI-filed request while the underlying risk is still HIGH/CRITICAL, the next
sync re-files one rather than silently leaving the equipment unexplained.
The open-request check is what keeps re-runs idempotent — it does not refile
once a request is already open for that equipment.

Usage:
    python -m fleetmind.agents.sync_equipment_status
    python -m fleetmind.agents.sync_equipment_status --dry-run
"""

from __future__ import annotations

import argparse
import re

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from fleetmind.db.models import (
    AgentDecision,
    Equipment,
    ModelPrediction,
    ServiceRequest,
    SignalWindow,
)
from fleetmind.db.session import create_db_engine, make_session_factory

URGENCY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
FLAG_THRESHOLD = URGENCY_ORDER["HIGH"]
_URGENCY_RE = re.compile(r"^\[(\w+)\]")

# Requests in these states no longer count as "open" for the idempotency
# guard, so a resolved/cancelled AI-filed request will not block a refile.
_CLOSED_REQUEST_STATUSES = ("resolved", "cancelled")


def _worst_urgency(decision_texts: list[str]) -> str | None:
    worst: str | None = None
    worst_rank = -1
    for text in decision_texts:
        match = _URGENCY_RE.match(text)
        if not match:
            continue
        urgency = match.group(1)
        rank = URGENCY_ORDER.get(urgency, -1)
        if rank > worst_rank:
            worst, worst_rank = urgency, rank
    return worst


def _worst_decision(rows: list[tuple[int, str, int]]) -> tuple[int, str, int] | None:
    """rows: [(decision_id, decision_text, prediction_id), ...]. Returns
    (decision_id, urgency, prediction_id) for the worst-ranked tagged
    decision, or None if none of the rows carry a recognized urgency tag."""
    best: tuple[int, str, int] | None = None
    best_rank = -1
    for decision_id, text, prediction_id in rows:
        match = _URGENCY_RE.match(text)
        if not match:
            continue
        rank = URGENCY_ORDER.get(match.group(1), -1)
        if rank > best_rank:
            best, best_rank = (decision_id, match.group(1), prediction_id), rank
    return best


def _ensure_ai_service_request(
    session,
    *,
    equip_id: int,
    customer_id: int,
    depot_id: int,
    decision_rows: list[tuple[int, str, int]],
    dry_run: bool,
) -> str | None:
    """Creates an AI-filed ServiceRequest for this equipment if none is
    currently open. Returns the urgency it filed at, or None if it left
    things alone (an open AI request already exists, or there's no tagged
    decision to file from, or a concurrent run already won the race -- see
    the IntegrityError handling below)."""
    already_open = session.execute(
        select(ServiceRequest.id).where(
            ServiceRequest.equipment_id == equip_id,
            ServiceRequest.agent_decision_id.is_not(None),
            ServiceRequest.status.notin_(_CLOSED_REQUEST_STATUSES),
        )
    ).first()
    if already_open is not None:
        return None

    worst_decision = _worst_decision(decision_rows)
    if worst_decision is None:
        return None
    decision_id, urgency, prediction_id = worst_decision

    if not dry_run:
        # Two concurrent sync runs can both pass the "already_open" read-check
        # above before either commits. The partial unique index
        # (uq_service_requests_open_ai_request, see db/models.py and alembic
        # migration 0003) makes the DB itself reject the loser's insert. Do
        # that insert inside a SAVEPOINT (session.begin_nested()) so a caught
        # IntegrityError only unwinds this one insert, not every change this
        # sync run has staged so far for other equipment in the same
        # transaction -- without the savepoint, Postgres would mark the whole
        # outer transaction as aborted and the caller's eventual commit()
        # would fail or lose unrelated work.
        try:
            with session.begin_nested():
                session.execute(
                    insert(ServiceRequest).values(
                        customer_id=customer_id,
                        depot_id=depot_id,
                        equipment_id=equip_id,
                        title=f"AI-flagged: {urgency.title()} urgency detected",
                        description=(
                            f"Automatically filed by the agent pipeline after a {urgency} "
                            "risk decision from the diagnosis/risk agent chain."
                        ),
                        priority=urgency.lower(),
                        status="reported",
                        agent_decision_id=decision_id,
                        prediction_id=prediction_id,
                    )
                )
        except IntegrityError:
            # Someone else (a concurrent run) already filed it. That's fine --
            # treat exactly like the already_open check above finding one.
            return None
    return urgency


def sync_equipment_status(
    session, dry_run: bool = False, filed_requests_log: list[tuple[str, str]] | None = None
) -> list[tuple[str, str, str, str | None]]:
    """Returns [(equipment_name, old_status, new_status, worst_urgency), ...] for changed rows.

    As a side effect, also files (or reports, in dry-run) AI-flagged service
    requests per _ensure_ai_service_request's rules — see module docstring.
    If filed_requests_log is given, (equipment_name, urgency) is appended to
    it for every request filed (or, in dry-run, that would be filed).
    """
    equipment_rows = session.execute(
        select(
            Equipment.id,
            Equipment.name,
            Equipment.status,
            Equipment.customer_id,
            Equipment.depot_id,
            Equipment.linked_experiment_id,
        ).where(Equipment.linked_experiment_id.is_not(None), Equipment.status != "retired")
    ).all()

    changes = []
    for equip_id, name, old_status, customer_id, depot_id, experiment_id in equipment_rows:
        decision_rows = session.execute(
            select(AgentDecision.id, AgentDecision.decision_text, AgentDecision.prediction_id)
            .join(ModelPrediction, ModelPrediction.id == AgentDecision.prediction_id)
            .join(SignalWindow, SignalWindow.id == ModelPrediction.window_id)
            .where(SignalWindow.experiment_id == experiment_id)
        ).all()

        worst = _worst_urgency([text for _, text, _ in decision_rows])
        worst_rank = URGENCY_ORDER.get(worst, -1) if worst is not None else -1
        new_status = "maintenance" if worst_rank >= FLAG_THRESHOLD else "active"

        if new_status != old_status:
            changes.append((name, old_status, new_status, worst))
            if not dry_run:
                session.execute(
                    update(Equipment).where(Equipment.id == equip_id).values(status=new_status)
                )

        if new_status == "maintenance":
            filed_urgency = _ensure_ai_service_request(
                session,
                equip_id=equip_id,
                customer_id=customer_id,
                depot_id=depot_id,
                decision_rows=decision_rows,
                dry_run=dry_run,
            )
            if filed_urgency is not None and filed_requests_log is not None:
                filed_requests_log.append((name, filed_urgency))

    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Equipment.status from agent_decisions.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without committing.")
    args = parser.parse_args()

    engine = create_db_engine()
    factory = make_session_factory(engine)
    session = factory()

    filed_requests: list[tuple[str, str]] = []
    try:
        changes = sync_equipment_status(
            session, dry_run=args.dry_run, filed_requests_log=filed_requests
        )
    except Exception:
        # Nothing in this run has been committed yet (commit() only happens
        # below, after the whole loop finishes) -- but roll back explicitly
        # rather than relying on the process exit / GC to eventually close the
        # connection, so a mid-loop failure (e.g. the DB going away) is a
        # clean, non-corrupting error: no partial commits, and the next run
        # starts from the same state as before this one.
        session.rollback()
        session.close()
        raise

    if not changes:
        print("No status changes.")
    else:
        for name, old, new, worst in changes:
            print(f"  {name}: {old} -> {new} (worst decision: {worst})")

    verb = "would file" if args.dry_run else "filed"
    for name, urgency in filed_requests:
        print(f"  {verb} AI service request for {name} ({urgency})")

    if args.dry_run:
        session.rollback()
        print(
            f"Dry run: {len(changes)} equipment would change, {len(filed_requests)} requests would be filed."
        )
    else:
        session.commit()
        print(
            f"Done. {len(changes)} equipment updated, {len(filed_requests)} AI service requests filed."
        )
    session.close()


if __name__ == "__main__":
    main()
