import asyncio

import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.data_submission import ContentType, DataSubmission, SubmissionStatus

router = APIRouter()


class SubmitURLRequest(BaseModel):
    url: HttpUrl
    content_type: ContentType = ContentType.TEXT


class SubmissionResponse(BaseModel):
    id: int
    url: str
    content_type: str
    status: str
    title: str | None
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

    # Fetch content in background
    asyncio.create_task(_fetch_content(submission.id))

    return SubmissionResponse(
        id=submission.id,
        url=submission.url,
        content_type=submission.content_type.value,
        status=submission.status.value,
        title=submission.title,
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


async def _fetch_content(submission_id: int):
    """Background task to fetch and extract text from a URL."""
    from app.core.database import async_session

    async with async_session() as db:
        result = await db.execute(
            select(DataSubmission).where(DataSubmission.id == submission_id)
        )
        submission = result.scalar_one_or_none()
        if not submission:
            return

        submission.status = SubmissionStatus.FETCHING
        await db.commit()

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(submission.url, headers={
                    "User-Agent": "WeBrain/1.0 (Training Data Collector)"
                })
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type or "text/plain" in content_type:
                if "text/html" in content_type:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Remove scripts, styles, nav
                    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                        tag.decompose()
                    text = soup.get_text(separator="\n", strip=True)
                    title = soup.title.string if soup.title else None
                else:
                    text = resp.text
                    title = submission.url.split("/")[-1]

                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = "\n".join(lines)

                if len(text) < 50:
                    submission.status = SubmissionStatus.FAILED
                    submission.error_message = "Not enough text content found"
                else:
                    submission.extracted_text = text[:500000]  # Cap at 500K chars
                    submission.title = (title or "")[:500]
                    submission.status = SubmissionStatus.READY
            else:
                # For non-text content, just store the URL as reference
                submission.title = submission.url.split("/")[-1]
                submission.status = SubmissionStatus.READY
                submission.extracted_text = f"[{submission.content_type.value}] {submission.url}"

        except Exception as e:
            submission.status = SubmissionStatus.FAILED
            submission.error_message = str(e)[:500]

        await db.commit()

        # Invalidate trainer cache so it picks up new data
        from app.ml.trainer import trainer
        trainer._training_text = None
