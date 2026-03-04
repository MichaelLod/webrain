import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis import enqueue_ingest
from app.core.security import get_current_user_id
from app.models.data_submission import ContentType, DataSubmission, SubmissionStatus

router = APIRouter()
log = logging.getLogger(__name__)


class SubmitURLRequest(BaseModel):
    url: HttpUrl
    content_type: ContentType = ContentType.TEXT


class SubmissionResponse(BaseModel):
    id: int
    url: str
    content_type: str
    status: str
    title: str | None
    image_s3_key: str | None = None
    trained: bool
    created_at: str


class SubmissionListResponse(BaseModel):
    submissions: list[SubmissionResponse]
    total: int


class DataStatsResponse(BaseModel):
    total_submissions: int
    ready_count: int
    total_text_chars: int
    contributors: int


@router.post("/submit", response_model=SubmissionResponse, status_code=201)
async def submit_url(
    body: SubmitURLRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    submission = DataSubmission(
        user_id=user_id,
        url=str(body.url),
        content_type=body.content_type,
        status=SubmissionStatus.PENDING,
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    queued = await enqueue_ingest(submission.id)
    if not queued:
        log.warning("Redis unavailable — submission %d stays PENDING for recovery", submission.id)

    return SubmissionResponse(
        id=submission.id,
        url=submission.url,
        content_type=submission.content_type.value,
        status=submission.status.value,
        title=submission.title,
        image_s3_key=submission.image_s3_key,
        trained=submission.trained,
        created_at=submission.created_at.isoformat(),
    )


@router.get("/submissions", response_model=SubmissionListResponse)
async def list_submissions(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DataSubmission)
        .order_by(DataSubmission.created_at.desc())
        .limit(min(limit, 200))
    )
    submissions = result.scalars().all()

    total_q = await db.execute(select(func.count(DataSubmission.id)))
    total = total_q.scalar() or 0

    return SubmissionListResponse(
        submissions=[
            SubmissionResponse(
                id=s.id,
                url=s.url,
                content_type=s.content_type.value,
                status=s.status.value,
                title=s.title,
                image_s3_key=s.image_s3_key,
                trained=s.trained,
                created_at=s.created_at.isoformat(),
            )
            for s in submissions
        ],
        total=total,
    )


@router.get("/stats", response_model=DataStatsResponse)
async def data_stats(db: AsyncSession = Depends(get_db)):
    total_q = await db.execute(select(func.count(DataSubmission.id)))
    ready_q = await db.execute(
        select(func.count(DataSubmission.id))
        .where(DataSubmission.status == SubmissionStatus.READY)
    )
    chars_q = await db.execute(
        select(func.coalesce(func.sum(func.length(DataSubmission.extracted_text)), 0))
        .where(DataSubmission.status == SubmissionStatus.READY)
    )
    contributors_q = await db.execute(
        select(func.count(func.distinct(DataSubmission.user_id)))
    )

    return DataStatsResponse(
        total_submissions=total_q.scalar() or 0,
        ready_count=ready_q.scalar() or 0,
        total_text_chars=chars_q.scalar() or 0,
        contributors=contributors_q.scalar() or 0,
    )
