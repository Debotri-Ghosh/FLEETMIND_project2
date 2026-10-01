"""Cache table for Claude-generated service-request narratives.

The narrative layer (fleetmind/agents/llm.py) makes a real API call per
flagged service request, so the result is cached here rather than regenerated
on every page view. `tool_calls` keeps the full tool call-and-result trail that
produced the prose, which is what makes a narrative auditable after the fact.

Dropping this table loses only cached text -- no decision, risk score or
service request depends on it, and the rule-based layer runs with the LLM
layer entirely absent.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_narratives",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "service_request_id",
            sa.BigInteger(),
            sa.ForeignKey("service_requests.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("diagnosis_narrative", sa.Text(), nullable=False),
        sa.Column("maintenance_recommendation", sa.Text(), nullable=False),
        sa.Column("corroboration", sa.Text(), nullable=False),
        sa.Column("evidence_cited", JSONB(), nullable=False),
        sa.Column("tool_calls", JSONB(), nullable=False),
        sa.Column("ungrounded_numbers", JSONB(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("llm_narratives")
