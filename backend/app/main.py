import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.core.config import CORS_ORIGINS
from app.core.database import Base, engine, async_session
from app.core.redis import close_redis, enqueue_ingest
from app.api.v1 import auth, tokens, compute, chat, leaderboard, data
from app.models.data_submission import DataSubmission, SubmissionStatus
from app.services.weight_server import router as weight_router

log = logging.getLogger(__name__)


async def _requeue_stuck_submissions():
    try:
        async with async_session() as db:
            result = await db.execute(
                select(DataSubmission.id).where(
                    DataSubmission.status.in_([SubmissionStatus.PENDING, SubmissionStatus.FETCHING])
                )
            )
            ids = [row[0] for row in result.all()]

        if not ids:
            return

        count = 0
        for sub_id in ids:
            if await enqueue_ingest(sub_id):
                count += 1
        log.info("Requeued %d/%d stuck submissions", count, len(ids))
    except Exception as e:
        log.warning("Failed to requeue stuck submissions: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _requeue_stuck_submissions()
    yield
    await close_redis()
    await engine.dispose()


app = FastAPI(
    title="WeBrain - AI from the people, for the people",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(tokens.router, prefix="/api/v1/tokens", tags=["tokens"])
app.include_router(compute.router, prefix="/api/v1/compute", tags=["compute"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(leaderboard.router, prefix="/api/v1/leaderboard", tags=["leaderboard"])
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(weight_router, prefix="/api/v1/weights", tags=["weights"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
