"""Partial unique index: at most one open, AI-filed service request per equipment.

Hardening for a real race condition: two concurrent
fleetmind.agents.sync_equipment_status runs (e.g. a cron job overlapping a
manual run) could both pass the "already has an open AI-filed request" check
before either commits, filing two AI requests for the same equipment. This
index makes Postgres itself reject the second insert; the resulting
IntegrityError is caught as a no-op in
fleetmind/agents/sync_equipment_status.py:_ensure_ai_service_request (someone
else already filed it, that's fine).

Mirrors the Index already declared on ServiceRequest in fleetmind/db/models.py
so the sqlite test DB (built from Base.metadata.create_all, not migrations)
enforces the same constraint.
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_INDEX_NAME = "uq_service_requests_open_ai_request"
_WHERE = "agent_decision_id IS NOT NULL AND status NOT IN ('resolved', 'cancelled')"


def upgrade() -> None:
    op.create_index(
        _INDEX_NAME,
        "service_requests",
        ["equipment_id"],
        unique=True,
        postgresql_where=sa.text(_WHERE),
        sqlite_where=sa.text(_WHERE),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="service_requests")
