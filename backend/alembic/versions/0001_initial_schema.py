"""Initial FleetMind schema: depots, experiments, signal_windows, model_predictions, agent_decisions."""

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "depots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("load_hp", sa.SmallInteger(), nullable=False),
        sa.Column("nominal_rpm", sa.SmallInteger()),
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("depot_id", sa.Integer(), sa.ForeignKey("depots.id"), nullable=False),
        sa.Column("fault_type", sa.Text(), nullable=False),
        sa.Column("fault_diameter_in", sa.Numeric(5, 3)),
        sa.Column("sensor_position", sa.Text(), nullable=False),
        sa.Column("sampling_rate_hz", sa.Integer(), nullable=False, server_default="12000"),
        sa.Column("source_file", sa.Text(), nullable=False),
        sa.Column("source_variable", sa.Text(), nullable=False),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "fault_type IN ('Normal', 'InnerRace', 'OuterRace', 'Ball')",
            name="ck_experiments_fault_type",
        ),
        sa.CheckConstraint("sensor_position IN ('DE', 'FE', 'BA')", name="ck_experiments_sensor"),
    )
    op.create_index("idx_experiments_depot", "experiments", ["depot_id"])

    op.create_table(
        "signal_windows",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.Integer(),
            sa.ForeignKey("experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("window_index", sa.Integer(), nullable=False),
        sa.Column("start_sample", sa.Integer(), nullable=False),
        sa.Column("end_sample", sa.Integer(), nullable=False),
        sa.Column("raw_values", sa.JSON(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.UniqueConstraint("experiment_id", "window_index", name="uq_windows_exp_idx"),
    )
    op.create_index("idx_windows_experiment", "signal_windows", ["experiment_id"])
    op.create_index("idx_windows_label", "signal_windows", ["label"])

    op.create_table(
        "model_predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("window_id", sa.BigInteger(), sa.ForeignKey("signal_windows.id"), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("predicted_label", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("idx_predictions_window", "model_predictions", ["window_id"])

    op.create_table(
        "agent_decisions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "prediction_id", sa.BigInteger(), sa.ForeignKey("model_predictions.id"), nullable=False
        ),
        sa.Column("agent_name", sa.Text(), nullable=False),
        sa.Column("decision_text", sa.Text(), nullable=False),
        sa.Column("risk_score", sa.Float()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.execute(
        "INSERT INTO depots (name, load_hp, nominal_rpm) VALUES "
        "('depot_0hp', 0, 1797), ('depot_1hp', 1, 1772), "
        "('depot_2hp', 2, 1750), ('depot_3hp', 3, 1730) "
        "ON CONFLICT (name) DO NOTHING"
    )


def downgrade() -> None:
    op.drop_table("agent_decisions")
    op.drop_table("model_predictions")
    op.drop_table("signal_windows")
    op.drop_table("experiments")
    op.drop_table("depots")
