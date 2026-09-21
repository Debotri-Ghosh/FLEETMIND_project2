"""Product-layer schema: customers, equipment, technicians, service_requests, work_orders.

Also adds Equipment.linked_experiment_id, a simulated 1:1 link from a demo
equipment row to one real CWRU experiment (CWRU has no per-motor identity of
its own). See fleetmind/db/seed_product_tables.py for how it's populated.
"""

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), unique=True),
        sa.Column("company", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "equipment",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("depot_id", sa.Integer(), sa.ForeignKey("depots.id"), nullable=False),
        sa.Column("customer_id", sa.Integer(), sa.ForeignKey("customers.id"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("equipment_type", sa.Text(), nullable=False, server_default="Motor"),
        sa.Column("installed_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column("linked_experiment_id", sa.Integer(), sa.ForeignKey("experiments.id")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('active', 'maintenance', 'retired')", name="ck_equipment_status"
        ),
    )
    op.create_index("idx_equipment_depot", "equipment", ["depot_id"])
    op.create_index("idx_equipment_customer", "equipment", ["customer_id"])

    op.create_table(
        "technicians",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("phone", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="available"),
        sa.CheckConstraint(
            "status IN ('available', 'busy', 'offline')", name="ck_technician_status"
        ),
    )

    op.create_table(
        "service_requests",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("customer_id", sa.Integer(), sa.ForeignKey("customers.id"), nullable=False),
        sa.Column("depot_id", sa.Integer(), sa.ForeignKey("depots.id"), nullable=False),
        sa.Column("equipment_id", sa.BigInteger(), sa.ForeignKey("equipment.id")),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("priority", sa.Text(), nullable=False, server_default="medium"),
        sa.Column("status", sa.Text(), nullable=False, server_default="reported"),
        sa.Column("agent_decision_id", sa.BigInteger(), sa.ForeignKey("agent_decisions.id")),
        sa.Column("prediction_id", sa.BigInteger(), sa.ForeignKey("model_predictions.id")),
        sa.Column("technician_id", sa.Integer(), sa.ForeignKey("technicians.id")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'critical')",
            name="ck_service_requests_priority",
        ),
        sa.CheckConstraint(
            "status IN ('reported', 'diagnosed', 'dispatched', 'in_progress', 'resolved', 'cancelled')",
            name="ck_service_requests_status",
        ),
    )
    op.create_index("idx_service_requests_customer", "service_requests", ["customer_id"])
    op.create_index("idx_service_requests_depot", "service_requests", ["depot_id"])
    op.create_index("idx_service_requests_status", "service_requests", ["status"])

    op.create_table(
        "work_orders",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "service_request_id",
            sa.BigInteger(),
            sa.ForeignKey("service_requests.id"),
            nullable=False,
        ),
        sa.Column("technician_id", sa.Integer(), sa.ForeignKey("technicians.id"), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="assigned"),
        sa.Column("location_lat", sa.Float()),
        sa.Column("location_lng", sa.Float()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('assigned', 'traveling', 'in_progress', 'completed', 'cancelled')",
            name="ck_work_orders_status",
        ),
        sa.UniqueConstraint("service_request_id", name="uq_work_orders_request"),
    )


def downgrade() -> None:
    op.drop_table("work_orders")
    op.drop_table("service_requests")
    op.drop_table("technicians")
    op.drop_table("equipment")
    op.drop_table("customers")
