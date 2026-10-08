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

Two flagging rules, chosen by ``Settings.decision_policy`` (OPEN_QUESTIONS.md Q6):

* ``heuristic`` -- the rule described above: any single HIGH/CRITICAL window
  flags the machine. Reads every decision for the experiment.
* ``expected_cost`` (default) -- reads only the expected-cost policy's own
  decisions (``policy.AGENT_NAME``), whose ``risk_score`` is P(faulted), and
  flags the machine only when a repeated SPRT over them, in window order,
  crosses Wald's upper boundary (``decision/sequential.py``). The request it
  files points at the decision on which the boundary was crossed.

Usage:
    python -m fleetmind.agents.sync_equipment_status
    python -m fleetmind.agents.sync_equipment_status --dry-run
    python -m fleetmind.agents.sync_equipment_status --policy heuristic
"""

from __future__ import annotations

import argparse
import re

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from fleetmind.core.config import get_settings
from fleetmind.db.models import (
    AgentDecision,
    Equipment,
    ModelPrediction,
    ServiceRequest,
    SignalWindow,
)
from fleetmind.db.session import create_db_engine, make_session_factory
from fleetmind.decision import policy as decision_policy
from fleetmind.decision import sequential

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
    trigger: tuple[int, str, int] | None = None,
    description: str | None = None,
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

    worst_decision = trigger if trigger is not None else _worst_decision(decision_rows)
    if worst_decision is None:
        return None
    decision_id, urgency, prediction_id = worst_decision
    if description is None:
        description = (
            f"Automatically filed by the agent pipeline after a {urgency} "
            "risk decision from the diagnosis/risk agent chain."
        )

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
                        description=description,
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


def policy_decision_rows(session, experiment_id: int) -> list[tuple[int, str, int, float | None]]:
    """The policy's own decisions for one recording, in window order.

    (decision_id, decision_text, prediction_id, P(faulted)). Shared by the sync
    bridge and the evidence API so both run the SPRT over identical evidence.
    """
    rows = session.execute(
        select(
            AgentDecision.id,
            AgentDecision.decision_text,
            AgentDecision.prediction_id,
            AgentDecision.risk_score,
        )
        .join(ModelPrediction, ModelPrediction.id == AgentDecision.prediction_id)
        .join(SignalWindow, SignalWindow.id == ModelPrediction.window_id)
        .where(
            SignalWindow.experiment_id == experiment_id,
            AgentDecision.agent_name == decision_policy.AGENT_NAME,
        )
        .order_by(SignalWindow.window_index)
    ).all()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


#: Urgency recorded when the SPRT flags a machine: a flag means the policy's
#: answer is to plan the repair, which costs.ACTION_TO_URGENCY maps to HIGH.
SPRT_FLAG_URGENCY = "HIGH"


def sprt_trigger(
    rows: list[tuple[int, str, int, float | None]],
) -> tuple[tuple[int, str, int] | None, sequential.SprtResult | None]:
    """Run the repeated SPRT over one machine's policy decisions, in window order.

    rows: [(decision_id, decision_text, prediction_id, p_faulted), ...], already
    ordered by window. A row with no P(faulted) is not evidence and is skipped,
    never read as zero. Returns (trigger, result); trigger is the decision on
    which the upper boundary was crossed, or None if the machine is not flagged.
    """
    usable = [(d, t, pr, p) for d, t, pr, p in rows if p is not None]
    if not usable:
        return None, None
    result = sequential.run([sequential.llr(p) for _, _, _, p in usable])
    if not result.flagged or result.flag_index is None:
        return None, result
    decision_id, _, prediction_id, _ = usable[result.flag_index]
    return (decision_id, SPRT_FLAG_URGENCY, prediction_id), result


def sync_equipment_status(
    session,
    dry_run: bool = False,
    filed_requests_log: list[tuple[str, str]] | None = None,
    policy: str | None = None,
) -> list[tuple[str, str, str, str | None]]:
    """Returns [(equipment_name, old_status, new_status, worst_urgency), ...] for changed rows.

    As a side effect, also files (or reports, in dry-run) AI-flagged service
    requests per _ensure_ai_service_request's rules — see module docstring.
    If filed_requests_log is given, (equipment_name, urgency) is appended to
    it for every request filed (or, in dry-run, that would be filed).

    ``policy`` overrides ``Settings.decision_policy`` ("heuristic" or
    "expected_cost"); see the module docstring for the two flagging rules.
    """
    mode = policy or get_settings().decision_policy
    if mode not in ("heuristic", "expected_cost"):
        raise ValueError(f"unknown decision policy {mode!r}")
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
        trigger: tuple[int, str, int] | None = None
        description: str | None = None
        decision_rows: list[tuple[int, str, int]]
        if mode == "heuristic":
            decision_rows = [
                (r[0], r[1], r[2])
                for r in session.execute(
                    select(
                        AgentDecision.id, AgentDecision.decision_text, AgentDecision.prediction_id
                    )
                    .join(ModelPrediction, ModelPrediction.id == AgentDecision.prediction_id)
                    .join(SignalWindow, SignalWindow.id == ModelPrediction.window_id)
                    .where(SignalWindow.experiment_id == experiment_id)
                ).all()
            ]
            worst = _worst_urgency([text for _, text, _ in decision_rows])
            worst_rank = URGENCY_ORDER.get(worst, -1) if worst is not None else -1
            new_status = "maintenance" if worst_rank >= FLAG_THRESHOLD else "active"
        else:
            policy_rows = policy_decision_rows(session, experiment_id)
            decision_rows = [(r[0], r[1], r[2]) for r in policy_rows]
            trigger, result = sprt_trigger(policy_rows)
            worst = trigger[1] if trigger is not None else None
            new_status = "maintenance" if trigger is not None else "active"
            if trigger is not None and result is not None:
                a, _ = sequential.boundaries()
                description = (
                    "Automatically filed by the expected-cost policy: a repeated "
                    "sequential probability ratio test over this machine's windows "
                    f"crossed its fault boundary (A = {a:.4f}) after "
                    f"{result.n_examined} window(s), with false-alarm target "
                    f"alpha = {sequential.DEFAULT_ALPHA} and miss target "
                    f"beta = {sequential.DEFAULT_BETA}."
                )

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
                trigger=trigger,
                description=description,
            )
            if filed_urgency is not None and filed_requests_log is not None:
                filed_requests_log.append((name, filed_urgency))

    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Equipment.status from agent_decisions.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without committing.")
    parser.add_argument(
        "--policy",
        choices=("heuristic", "expected_cost"),
        default=None,
        help=(
            "Flagging rule; defaults to Settings.decision_policy. Use 'heuristic' after "
            "agents/run.py --all-pending, which writes heuristic decisions only."
        ),
    )
    args = parser.parse_args()

    engine = create_db_engine()
    factory = make_session_factory(engine)
    session = factory()

    filed_requests: list[tuple[str, str]] = []
    try:
        changes = sync_equipment_status(
            session, dry_run=args.dry_run, filed_requests_log=filed_requests, policy=args.policy
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
