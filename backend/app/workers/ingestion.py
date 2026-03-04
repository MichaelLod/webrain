"""Standalone ingestion worker — run with: python -m app.workers.ingestion"""

import asyncio
import hashlib
import json
import logging
import signal

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select

from app.core.database import async_session
from app.core.redis import (
    INGEST_QUEUE,
    enqueue_ingest,
    enqueue_training_ready,
    get_redis,
)
from app.models.data_submission import ContentType, DataSubmission, SubmissionStatus
import app.models.user  # noqa: F401 — register User table for FK resolution

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_RETRIES = 3
TRANSIENT_ERRORS = (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, ConnectionError, OSError)

_shutdown = asyncio.Event()


def _handle_signal():
    log.info("Shutdown signal received")
    _shutdown.set()


async def process_submission(submission_id: int, retry_count: int):
    from app.ml.vision import preprocess_image, save_image_tensor_to_s3

    async with async_session() as db:
        result = await db.execute(
            select(DataSubmission).where(DataSubmission.id == submission_id)
        )
        submission = result.scalar_one_or_none()
        if not submission:
            log.warning("Submission %d not found, skipping", submission_id)
            return

        submission.status = SubmissionStatus.FETCHING
        await db.commit()

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(submission.url, headers={
                    "User-Agent": "WeBrain/1.0 (Training Data Collector)"
                })
                resp.raise_for_status()

            content_type_header = resp.headers.get("content-type", "")

            if content_type_header.startswith("image/"):
                tensor = preprocess_image(resp.content)
                s3_key = save_image_tensor_to_s3(tensor, submission.id)
                content_hash = hashlib.sha256(resp.content).hexdigest()

                dupe = await db.execute(
                    select(DataSubmission.id)
                    .where(DataSubmission.content_hash == content_hash)
                    .where(DataSubmission.id != submission.id)
                    .limit(1)
                )
                if dupe.scalar_one_or_none():
                    submission.status = SubmissionStatus.FAILED
                    submission.error_message = "Duplicate image already exists"
                else:
                    submission.content_type = ContentType.IMAGE
                    submission.content_hash = content_hash
                    submission.image_s3_key = s3_key
                    submission.title = submission.url.split("/")[-1][:500]
                    submission.extracted_text = submission.title
                    submission.status = SubmissionStatus.READY

            elif "text/html" in content_type_header:
                soup = BeautifulSoup(resp.text, "html.parser")

                if submission.content_type == ContentType.IMAGE:
                    await _extract_image_from_html(submission, soup, db)
                else:
                    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                        tag.decompose()
                    text = soup.get_text(separator="\n", strip=True)
                    title = soup.title.string if soup.title else None

                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    text = "\n".join(lines)

                    if len(text) < 50:
                        submission.status = SubmissionStatus.FAILED
                        submission.error_message = "Not enough text content found"
                    else:
                        content_hash = hashlib.sha256(text.encode()).hexdigest()
                        dupe = await db.execute(
                            select(DataSubmission.id)
                            .where(DataSubmission.content_hash == content_hash)
                            .where(DataSubmission.id != submission.id)
                            .limit(1)
                        )
                        if dupe.scalar_one_or_none():
                            submission.status = SubmissionStatus.FAILED
                            submission.error_message = "Duplicate content already exists"
                        else:
                            submission.extracted_text = text[:500000]
                            submission.content_hash = content_hash
                            submission.title = (title or "")[:500]
                            submission.status = SubmissionStatus.READY

            elif "text/plain" in content_type_header:
                text = resp.text
                title = submission.url.split("/")[-1]
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = "\n".join(lines)

                if len(text) < 50:
                    submission.status = SubmissionStatus.FAILED
                    submission.error_message = "Not enough text content found"
                else:
                    content_hash = hashlib.sha256(text.encode()).hexdigest()
                    dupe = await db.execute(
                        select(DataSubmission.id)
                        .where(DataSubmission.content_hash == content_hash)
                        .where(DataSubmission.id != submission.id)
                        .limit(1)
                    )
                    if dupe.scalar_one_or_none():
                        submission.status = SubmissionStatus.FAILED
                        submission.error_message = "Duplicate content already exists"
                    else:
                        submission.extracted_text = text[:500000]
                        submission.content_hash = content_hash
                        submission.title = (title or "")[:500]
                        submission.status = SubmissionStatus.READY
            else:
                submission.title = submission.url.split("/")[-1]
                submission.status = SubmissionStatus.READY
                ref_text = f"[{submission.content_type.value}] {submission.url}"
                submission.extracted_text = ref_text
                submission.content_hash = hashlib.sha256(ref_text.encode()).hexdigest()

        except TRANSIENT_ERRORS as e:
            if retry_count < MAX_RETRIES:
                log.warning("Transient error for submission %d (attempt %d): %s", submission_id, retry_count + 1, e)
                await db.rollback()
                await enqueue_ingest(submission_id, retry_count + 1)
                return
            submission.status = SubmissionStatus.FAILED
            submission.error_message = f"Failed after {MAX_RETRIES} retries: {e}"[:500]

        except Exception as e:
            submission.status = SubmissionStatus.FAILED
            submission.error_message = str(e)[:500]

        await db.commit()

        if submission.status == SubmissionStatus.READY:
            await enqueue_training_ready(submission.id, submission.content_type.value)
            log.info("Submission %d ready (type=%s)", submission.id, submission.content_type.value)


async def _extract_image_from_html(submission: DataSubmission, soup: BeautifulSoup, db):
    from app.ml.vision import preprocess_image, save_image_tensor_to_s3
    from urllib.parse import urljoin

    og_image = soup.find("meta", property="og:image")
    img_url = og_image["content"] if og_image and og_image.get("content") else None

    if not img_url:
        images = soup.find_all("img", src=True)
        best_img = None
        best_area = 0
        for img in images:
            w = int(img.get("width", 0) or 0)
            h = int(img.get("height", 0) or 0)
            area = w * h
            if area > best_area:
                best_area = area
                best_img = img
        if best_img is None and images:
            best_img = images[0]
        if best_img:
            img_url = best_img["src"]

    if not img_url:
        submission.status = SubmissionStatus.FAILED
        submission.error_message = "No image found on page"
        return

    img_url = urljoin(submission.url, img_url)

    caption = ""
    if soup.title:
        caption = soup.title.string or ""

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        img_resp = await client.get(img_url, headers={
            "User-Agent": "WeBrain/1.0 (Training Data Collector)"
        })
        img_resp.raise_for_status()

    tensor = preprocess_image(img_resp.content)
    s3_key = save_image_tensor_to_s3(tensor, submission.id)
    content_hash = hashlib.sha256(img_resp.content).hexdigest()

    dupe = await db.execute(
        select(DataSubmission.id)
        .where(DataSubmission.content_hash == content_hash)
        .where(DataSubmission.id != submission.id)
        .limit(1)
    )
    if dupe.scalar_one_or_none():
        submission.status = SubmissionStatus.FAILED
        submission.error_message = "Duplicate image already exists"
    else:
        submission.content_type = ContentType.IMAGE
        submission.content_hash = content_hash
        submission.image_s3_key = s3_key
        submission.title = (caption or img_url.split("/")[-1])[:500]
        submission.extracted_text = caption[:500000] if caption else submission.title
        submission.status = SubmissionStatus.READY


async def run_worker():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    log.info("Ingestion worker started, listening on %s", INGEST_QUEUE)

    while not _shutdown.is_set():
        try:
            r = await get_redis()
            result = await r.brpop(INGEST_QUEUE, timeout=1)
            if result is None:
                continue

            _, raw = result
            msg = json.loads(raw)
            submission_id = msg["submission_id"]
            retry_count = msg.get("retry_count", 0)

            log.info("Processing submission %d (attempt %d)", submission_id, retry_count + 1)
            await process_submission(submission_id, retry_count)

        except Exception as e:
            log.error("Worker loop error: %s", e)
            await asyncio.sleep(2)

    log.info("Ingestion worker shutting down")


if __name__ == "__main__":
    asyncio.run(run_worker())
