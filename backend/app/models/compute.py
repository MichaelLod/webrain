from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TaskStatus(str, PyEnum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"


class ComputeTask(Base):
    __tablename__ = "compute_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_type: Mapped[str] = mapped_column(default="matmul")
    training_step: Mapped[int] = mapped_column(Integer, nullable=False)
    layer_index: Mapped[int] = mapped_column(Integer, nullable=False)
    tile_i: Mapped[int] = mapped_column(Integer, nullable=False)
    tile_j: Mapped[int] = mapped_column(Integer, nullable=False)
    tile_k: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ComputeResult(Base):
    __tablename__ = "compute_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("compute_tasks.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    compute_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    tokens_awarded: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
