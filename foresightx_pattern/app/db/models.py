from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Numeric, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from foresightx_pattern.app.db.base import Base


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry_entries"
    __table_args__ = (UniqueConstraint("artifact_key", name="uq_model_registry_entries_artifact_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    artifact_key: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    features_count: Mapped[int] = mapped_column(nullable=False)
    model_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    feature_names: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    model_file: Mapped[str] = mapped_column(String(1024), nullable=False)
    scaler_file: Mapped[str] = mapped_column(String(1024), nullable=False)
    metadata_file: Mapped[str] = mapped_column(String(1024), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    predictions: Mapped[list["PredictionJob"]] = relationship(back_populates="model_entry")


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    model_registry_entry_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_registry_entries.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    requested_as_of_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    predicted_return: Mapped[float | None] = mapped_column(Numeric(12, 8), nullable=True)
    predicted_direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    latest_close: Mapped[float | None] = mapped_column(Numeric(14, 4), nullable=True)
    predicted_next_close: Mapped[float | None] = mapped_column(Numeric(14, 4), nullable=True)
    signal_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    signal_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_entry: Mapped[ModelRegistryEntry | None] = relationship(back_populates="predictions")
