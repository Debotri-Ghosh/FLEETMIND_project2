"""Replay-arrival marker on signal_windows.

CWRU is a static historical archive, so there is no live sensor feed to react
to. The replay engine (fleetmind/replay/engine.py) simulates one by
progressively "releasing" already-ingested windows: released_at IS NULL means
the window has not arrived yet and the live pipeline must skip it.

Existing rows are backfilled to now() on upgrade, so the state this migration
lands on is "everything has already arrived" -- which is exactly the state the
documented offline batch results were produced in. A replay run starts by
resetting these back to NULL.
"""

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

_INDEX_NAME = "idx_windows_released_at"


def upgrade() -> None:
    op.add_column(
        "signal_windows",
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Backfill: every window already ingested counts as already arrived.
    op.execute("UPDATE signal_windows SET released_at = now()")
    # The replay worker polls for "released but not yet predicted" on every
    # tick, so this column is on the hot path.
    op.create_index(_INDEX_NAME, "signal_windows", ["released_at"])


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="signal_windows")
    op.drop_column("signal_windows", "released_at")
