import asyncio
import io
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

from app.ml.model import TinyGPT, TinyGPTConfig
from app.ml.tokenizer import CharTokenizer
from app.ml.tiling import TileTask, assemble_tiles, decompose_matmul
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
        self.tokenizer = CharTokenizer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
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
            self.optimizer.load_state_dict(state["optimizer"])
            self.step = state.get("step", 0)
            self.current_loss = state.get("loss", 0.0)
            log.info("Loaded local checkpoint at step %d", self.step)

    def save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
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

    async def run_training_step(self, dispatch_fn) -> float:
        self.model.train()
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

        return self.current_loss


# Global instance
trainer = TrainingOrchestrator()
