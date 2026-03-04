from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TrainingState(Base):
    __tablename__ = "training_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[int] = mapped_column(Integer, default=1)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    current_loss: Mapped[float] = mapped_column(Float, default=0.0)
    total_flops: Mapped[float] = mapped_column(Float, default=0.0)
    checkpoint_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
