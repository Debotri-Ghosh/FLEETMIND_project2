"""Store each prediction's full class distribution.

The expected-cost decision policy (fleetmind/decision/policy.py) needs
P(faulted) = 1 - P(Normal). model_predictions held only the argmax label and
its probability, which cannot supply that. Nullable, so every existing row --
and the heuristic decision path, which never reads it -- is unaffected.

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_predictions",
        sa.Column(
            "class_probabilities",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("model_predictions", "class_probabilities")
