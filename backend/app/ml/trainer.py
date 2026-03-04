import asyncio
import io
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn

from app.ml.model import TinyGPT, TinyGPTConfig, VisionEncoder
from app.ml.tokenizer import CharTokenizer
from app.ml.tiling import TileTask, assemble_tiles, decompose_matmul
from app.ml.vision import load_image_tensor_from_s3, delete_image_from_s3
from app.core.config import (
    TILE_SIZE,
    S3_BUCKET,
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    S3_ENDPOINT,
    S3_REGION,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
S3_CHECKPOINT_KEY = "checkpoints/latest.pt"

log = logging.getLogger(__name__)


def _get_s3_client():
    if not S3_BUCKET or not S3_ACCESS_KEY_ID:
        return None
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
    )


class TrainingOrchestrator:
    def __init__(self):
        self.cfg = TinyGPTConfig()
        self.model = TinyGPT(self.cfg)
        self.vision_encoder = VisionEncoder(self.cfg)
        self.tokenizer = CharTokenizer()
        all_params = list(self.model.parameters()) + list(self.vision_encoder.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=3e-4)
        self.loss_fn = nn.CrossEntropyLoss()
        self.step = 0
        self.current_loss = 0.0
        self.total_flops = 0.0
        self.is_training = False
        self._training_text: str | None = None
        self._load_checkpoint()

    def _load_checkpoint(self):
        # Try S3 first
        s3 = _get_s3_client()
        if s3:
            try:
                buf = io.BytesIO()
                s3.download_fileobj(S3_BUCKET, S3_CHECKPOINT_KEY, buf)
                buf.seek(0)
                state = torch.load(buf, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state["model"])
                if "vision_encoder" in state:
                    self.vision_encoder.load_state_dict(state["vision_encoder"])
                self.optimizer.load_state_dict(state["optimizer"])
                self.step = state.get("step", 0)
                self.current_loss = state.get("loss", 0.0)
                log.info("Loaded checkpoint from S3 at step %d", self.step)
                return
            except Exception as e:
                log.info("No S3 checkpoint found (%s), checking local", e)

        # Fallback to local
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state["model"])
            if "vision_encoder" in state:
                self.vision_encoder.load_state_dict(state["vision_encoder"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.step = state.get("step", 0)
            self.current_loss = state.get("loss", 0.0)
            log.info("Loaded local checkpoint at step %d", self.step)

    def save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "vision_encoder": self.vision_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "loss": self.current_loss,
        }

        # Save to S3
        s3 = _get_s3_client()
        if s3:
            try:
                buf = io.BytesIO()
                torch.save(state, buf)
                buf.seek(0)
                s3.upload_fileobj(buf, S3_BUCKET, S3_CHECKPOINT_KEY)
                log.info("Saved checkpoint to S3 at step %d", self.step)
            except Exception as e:
                log.error("Failed to save checkpoint to S3: %s", e)

        # Always save locally too
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        torch.save(state, path)

    def load_training_data(self) -> str:
        if self._training_text:
            return self._training_text

        texts = []

        # Load from local files
        if os.path.isdir(DATA_DIR):
            for fname in sorted(os.listdir(DATA_DIR)):
                if fname.endswith(".txt"):
                    with open(os.path.join(DATA_DIR, fname)) as f:
                        texts.append(f.read())

        # Load from database (user-submitted URLs)
        try:
            import asyncio
            texts += asyncio.get_event_loop().run_until_complete(self._load_db_texts())
        except Exception as e:
            log.warning("Failed to load DB texts: %s", e)

        if not texts:
            texts = ["The quick brown fox jumps over the lazy dog. " * 1000]

        self._training_text = "\n".join(texts)
        log.info("Loaded training data: %d chars from %d sources", len(self._training_text), len(texts))
        return self._training_text

    @staticmethod
    async def _load_db_texts() -> list[str]:
        from app.core.database import async_session
        from app.models.data_submission import DataSubmission, SubmissionStatus
        from sqlalchemy import select, update

        async with async_session() as db:
            result = await db.execute(
                select(DataSubmission.id, DataSubmission.extracted_text)
                .where(DataSubmission.status == SubmissionStatus.READY)
                .where(DataSubmission.extracted_text.isnot(None))
            )
            rows = result.all()
            texts = [row[1] for row in rows if row[1] and len(row[1]) > 50]
            ids = [row[0] for row in rows if row[1] and len(row[1]) > 50]
            if ids:
                await db.execute(
                    update(DataSubmission)
                    .where(DataSubmission.id.in_(ids))
                    .values(trained=True)
                )
                await db.commit()
            return texts

    def get_batch(self, batch_size: int = 32, seq_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.load_training_data()
        tokens = self.tokenizer.encode(text)
        data = np.array(tokens, dtype=np.int64)

        ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
        x = np.stack([data[i : i + seq_len] for i in ix])
        y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix])
        return torch.tensor(x), torch.tensor(y)

    def decompose_forward_layer(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        step: int,
        layer: int,
        op_name: str,
    ) -> list[TileTask]:
        return decompose_matmul(x, weight, step, layer, op_name, TILE_SIZE)

    def assemble_forward_result(
        self,
        tiles: dict[tuple[int, int, int], np.ndarray],
        M: int,
        N: int,
    ) -> np.ndarray:
        return assemble_tiles(tiles, M, N, TILE_SIZE)

    async def get_vision_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]] | None:
        """Load image tensors + paired captions from DB for vision training."""
        from app.core.database import async_session
        from app.models.data_submission import DataSubmission, ContentType, SubmissionStatus
        from sqlalchemy import select

        try:
            async with async_session() as db:
                result = await db.execute(
                    select(DataSubmission.id, DataSubmission.extracted_text, DataSubmission.image_s3_key)
                    .where(DataSubmission.content_type == ContentType.IMAGE)
                    .where(DataSubmission.status == SubmissionStatus.READY)
                    .where(DataSubmission.image_s3_key.isnot(None))
                    .where(DataSubmission.trained == False)  # noqa: E712
                    .limit(8)
                )
                rows = result.all()
        except Exception as e:
            log.warning("Failed to query vision data: %s", e)
            return None

        if not rows:
            return None

        images = []
        captions = []
        ids = []
        for sub_id, text, _s3_key in rows:
            tensor = load_image_tensor_from_s3(sub_id)
            if tensor is None:
                continue
            images.append(tensor)
            captions.append(text or "an image")
            ids.append(sub_id)

        if not images:
            return None

        image_batch = torch.stack(images)  # [B, 3, 224, 224]

        # Tokenize captions into target sequences
        seq_len = 128
        all_x = []
        all_y = []
        for caption in captions:
            tokens = self.tokenizer.encode(caption)
            if len(tokens) < 2:
                tokens = self.tokenizer.encode("an image")
            data = np.array(tokens, dtype=np.int64)
            # Pad or truncate to seq_len + 1
            if len(data) < seq_len + 1:
                data = np.pad(data, (0, seq_len + 1 - len(data)))
            else:
                data = data[: seq_len + 1]
            all_x.append(data[:seq_len])
            all_y.append(data[1: seq_len + 1])

        return (
            image_batch,
            torch.tensor(np.stack(all_x)),
            torch.tensor(np.stack(all_y)),
            ids,
        )

    async def _mark_vision_trained(self, ids: list[int]):
        from app.core.database import async_session
        from app.models.data_submission import DataSubmission
        from sqlalchemy import update

        try:
            async with async_session() as db:
                await db.execute(
                    update(DataSubmission)
                    .where(DataSubmission.id.in_(ids))
                    .values(trained=True)
                )
                await db.commit()
            for sub_id in ids:
                delete_image_from_s3(sub_id)
        except Exception as e:
            log.warning("Failed to mark vision data as trained: %s", e)

    async def check_training_ready_queue(self):
        from app.core.redis import TRAINING_READY_QUEUE, get_redis

        try:
            r = await get_redis()
            count = 0
            while True:
                item = await r.rpop(TRAINING_READY_QUEUE)
                if item is None:
                    break
                count += 1
            if count > 0:
                self._training_text = None
                log.info("Training-ready queue: %d items, cache invalidated", count)
        except Exception as e:
            log.debug("Could not check training-ready queue: %s", e)

    async def run_training_step(self, dispatch_fn) -> float:
        await self.check_training_ready_queue()
        self.model.train()
        self.vision_encoder.train()

        # 30% chance of vision batch when vision data is available
        use_vision = random.random() < 0.3
        if use_vision:
            vision_data = await self.get_vision_batch()
        else:
            vision_data = None

        if vision_data is not None:
            image_batch, x, y, sub_ids = vision_data
            image_embeds = self.vision_encoder(image_batch)
            logits = self.model(x, image_embeds=image_embeds)
            B, T, V = logits.shape
            loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))
            log.info("Vision training step %d, batch_size=%d", self.step, B)
        else:
            x, y = self.get_batch()
            logits = self.model(x)
            B, T, V = logits.shape
            loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        self.current_loss = loss.item()

        if self.step % 50 == 0:
            self.save_checkpoint()

        if vision_data is not None:
            await self._mark_vision_trained(sub_ids)

        return self.current_loss


# Global instance
trainer = TrainingOrchestrator()
