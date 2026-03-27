"""create pattern service persistence tables

Revision ID: 20260328_0001
Revises:
Create Date: 2026-03-28 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "20260328_0001"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_registry_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("artifact_key", sa.String(length=128), nullable=False),
        sa.Column("model_type", sa.String(length=64), nullable=False),
        sa.Column("features_count", sa.Integer(), nullable=False),
        sa.Column("model_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("feature_names", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model_file", sa.String(length=1024), nullable=False),
        sa.Column("scaler_file", sa.String(length=1024), nullable=False),
        sa.Column("metadata_file", sa.String(length=1024), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_key", name="uq_model_registry_entries_artifact_key"),
    )
    op.create_index(op.f("ix_model_registry_entries_symbol"), "model_registry_entries", ["symbol"], unique=False)
    op.create_table(
        "prediction_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("model_registry_entry_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("requested_as_of_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.Column("predicted_return", sa.Numeric(precision=12, scale=8), nullable=True),
        sa.Column("predicted_direction", sa.String(length=16), nullable=True),
        sa.Column("latest_close", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("predicted_next_close", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("signal_label", sa.String(length=16), nullable=True),
        sa.Column("signal_confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("error_message", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["model_registry_entry_id"], ["model_registry_entries.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_prediction_jobs_model_registry_entry_id"), "prediction_jobs", ["model_registry_entry_id"], unique=False)
    op.create_index(op.f("ix_prediction_jobs_symbol"), "prediction_jobs", ["symbol"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_prediction_jobs_symbol"), table_name="prediction_jobs")
    op.drop_index(op.f("ix_prediction_jobs_model_registry_entry_id"), table_name="prediction_jobs")
    op.drop_table("prediction_jobs")
    op.drop_index(op.f("ix_model_registry_entries_symbol"), table_name="model_registry_entries")
    op.drop_table("model_registry_entries")
