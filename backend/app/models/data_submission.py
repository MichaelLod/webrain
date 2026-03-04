from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SubmissionStatus(str, PyEnum):
    PENDING = "pending"
    FETCHING = "fetching"
    READY = "ready"
    FAILED = "failed"


class ContentType(str, PyEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    OTHER = "other"


class DataSubmission(Base):
    __tablename__ = "data_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    url: Mapped[str] = mapped_column(String(2000), nullable=False, index=True)
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType), default=ContentType.TEXT)
    status: Mapped[SubmissionStatus] = mapped_column(Enum(SubmissionStatus), default=SubmissionStatus.PENDING)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    image_s3_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    trained: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
