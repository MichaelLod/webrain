import json
import logging

import redis.asyncio as aioredis

from app.core.config import REDIS_URL

log = logging.getLogger(__name__)

INGEST_QUEUE = "webrain:ingest"
TRAINING_READY_QUEUE = "webrain:training-ready"

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is not None:
        try:
            await _redis.ping()
            return _redis
        except Exception:
            _redis = None
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    await _redis.ping()
    return _redis


async def enqueue_ingest(submission_id: int, retry_count: int = 0) -> bool:
    try:
        r = await get_redis()
        payload = json.dumps({"submission_id": submission_id, "retry_count": retry_count})
        await r.lpush(INGEST_QUEUE, payload)
        return True
    except Exception as e:
        log.warning("Failed to enqueue ingest for submission %d: %s", submission_id, e)
        return False


async def enqueue_training_ready(submission_id: int, content_type: str) -> bool:
    try:
        r = await get_redis()
        payload = json.dumps({"submission_id": submission_id, "content_type": content_type})
        await r.lpush(TRAINING_READY_QUEUE, payload)
        return True
    except Exception as e:
        log.warning("Failed to enqueue training-ready for submission %d: %s", submission_id, e)
        return False


async def close_redis():
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
