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

Usage:
    python -m fleetmind.agents.sync_equipment_status
    python -m fleetmind.agents.sync_equipment_status --dry-run
"""

from __future__ import annotations

import argparse
import re

from sqlalchemy import select

from fleetmind.db.models import AgentDecision, Equipment, ModelPrediction, SignalWindow
from fleetmind.db.session import create_db_engine, make_session_factory

URGENCY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
FLAG_THRESHOLD = URGENCY_ORDER["HIGH"]
_URGENCY_RE = re.compile(r"^\[(\w+)\]")


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


def sync_equipment_status(session, dry_run: bool = False) -> list[tuple[str, str, str, str | None]]:
    """Returns [(equipment_name, old_status, new_status, worst_urgency), ...] for changed rows."""
    equipment_rows = session.execute(
        select(Equipment.id, Equipment.name, Equipment.status, Equipment.linked_experiment_id).where(
            Equipment.linked_experiment_id.is_not(None), Equipment.status != "retired"
        )
    ).all()

    changes = []
    for equip_id, name, old_status, experiment_id in equipment_rows:
        decision_texts = session.execute(
            select(AgentDecision.decision_text)
            .join(ModelPrediction, ModelPrediction.id == AgentDecision.prediction_id)
            .join(SignalWindow, SignalWindow.id == ModelPrediction.window_id)
            .where(SignalWindow.experiment_id == experiment_id)
        ).scalars().all()

        worst = _worst_urgency(decision_texts)
        worst_rank = URGENCY_ORDER.get(worst, -1)
        new_status = "maintenance" if worst_rank >= FLAG_THRESHOLD else "active"

        if new_status != old_status:
            changes.append((name, old_status, new_status, worst))
            if not dry_run:
                session.execute(
                    Equipment.__table__.update()
                    .where(Equipment.id == equip_id)
                    .values(status=new_status)
                )

    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Equipment.status from agent_decisions.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without committing.")
    args = parser.parse_args()

    engine = create_db_engine()
    factory = make_session_factory(engine)
    session = factory()

    changes = sync_equipment_status(session, dry_run=args.dry_run)

    if not changes:
        print("No status changes.")
    else:
        for name, old, new, worst in changes:
            print(f"  {name}: {old} -> {new} (worst decision: {worst})")

    if args.dry_run:
        session.rollback()
        print(f"Dry run: {len(changes)} equipment would change.")
    else:
        session.commit()
        print(f"Done. {len(changes)} equipment updated.")
    session.close()


if __name__ == "__main__":
    main()
