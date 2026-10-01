"""Logins, with the scope each role is allowed to act within.

`users` carries the authorization scope itself (customer_id for a customer,
depot_id for an operator) rather than leaving it to application code, and a
CHECK constraint makes the role/scope pairing a database invariant -- a
half-configured login such as an operator with no site cannot be inserted and
then silently behave like an admin.

No rows are created here. Run `python -m fleetmind.db.seed_users` to create the
documented demo logins; a fresh database with no users simply refuses every
request, which is the correct default.
"""

import sqlalchemy as sa

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("hashed_password", sa.Text(), nullable=False),
        sa.Column("full_name", sa.Text(), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("customer_id", sa.Integer(), sa.ForeignKey("customers.id"), nullable=True),
        sa.Column("depot_id", sa.Integer(), sa.ForeignKey("depots.id"), nullable=True),
        sa.Column("technician_id", sa.Integer(), sa.ForeignKey("technicians.id"), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("role IN ('customer', 'operator', 'admin')", name="ck_users_role"),
        sa.CheckConstraint(
            "(role = 'customer' AND customer_id IS NOT NULL AND depot_id IS NULL) "
            "OR (role = 'operator' AND depot_id IS NOT NULL AND customer_id IS NULL) "
            "OR (role = 'admin' AND customer_id IS NULL AND depot_id IS NULL)",
            name="ck_users_role_scope",
        ),
    )
    op.create_index("idx_users_role", "users", ["role"])


def downgrade() -> None:
    op.drop_index("idx_users_role", table_name="users")
    op.drop_table("users")
