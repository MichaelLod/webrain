import io
import json
import logging
import math
import os
import random
import tempfile
import threading
import time

import numpy as np
import torch
import torch.nn as nn

from app.ml.model import WeBrainGPT, ModelConfig, VisionEncoder
from app.ml.tokenizer import BPETokenizer, CharTokenizer, get_tokenizer
from app.ml.tiling import TileTask, assemble_tiles, decompose_matmul
from app.ml.vision import load_image_tensor_from_s3, delete_image_from_s3
from app.core.config import (
    TILE_SIZE,
    S3_BUCKET,
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    S3_ENDPOINT,
    S3_REGION,
    HF_REPO_ID,
    HF_TOKEN,
    MODEL_VERSION,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
S3_CHECKPOINT_KEY = "checkpoints/latest.pt"

log = logging.getLogger(__name__)
_HF_PUSH_INTERVAL = 7 * 24 * 60 * 60

CONFIG_VERSION = 2

# Cosine LR schedule parameters
WARMUP_STEPS = 500
MAX_LR = 3e-4
MIN_LR = 3e-5
TOTAL_STEPS = 100_000


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


def _get_lr(step: int) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    decay_steps = max(TOTAL_STEPS - WARMUP_STEPS, 1)
    progress = min((step - WARMUP_STEPS) / decay_steps, 1.0)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


class TrainingOrchestrator:
    def __init__(self):
        self.cfg = ModelConfig()
        self.model = WeBrainGPT(self.cfg)
        self.vision_encoder = VisionEncoder(self.cfg)
        self.tokenizer = get_tokenizer()
        all_params = list(self.model.parameters()) + list(self.vision_encoder.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.step = 0
        self.current_loss = 0.0
        self.total_flops = 0.0
        self.is_training = False
        self._training_text: str | None = None
        self._last_hf_push: float = 0.0
        self._use_amp = torch.cuda.is_available()
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            self.model = self.model.to(self._device)
            self.vision_encoder = self.vision_encoder.to(self._device)
        self._load_checkpoint()

    def _load_checkpoint(self):
        s3 = _get_s3_client()
        if s3:
            try:
                buf = io.BytesIO()
                s3.download_fileobj(S3_BUCKET, S3_CHECKPOINT_KEY, buf)
                buf.seek(0)
                state = torch.load(buf, map_location="cpu", weights_only=True)
                if self._try_load_state(state):
                    return
            except Exception as e:
                log.info("No S3 checkpoint found (%s), checking local", e)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            self._try_load_state(state)

    def _try_load_state(self, state: dict) -> bool:
        """Load checkpoint state. Returns True on success, False on version mismatch."""
        version = state.get("config_version", 1)
        if version != CONFIG_VERSION:
            log.warning(
                "Checkpoint version %d != current %d. Reinitializing model (old checkpoint ignored).",
                version, CONFIG_VERSION,
            )
            return False

        try:
            self.model.load_state_dict(state["model"])
            if "vision_encoder" in state:
                self.vision_encoder.load_state_dict(state["vision_encoder"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.step = state.get("step", 0)
            self.current_loss = state.get("loss", 0.0)
            log.info("Loaded checkpoint at step %d (version %d)", self.step, version)
            return True
        except Exception as e:
            log.warning("Failed to load checkpoint state: %s. Reinitializing.", e)
            return False

    def save_checkpoint(self):
        state = {
            "config_version": CONFIG_VERSION,
            "model": self.model.state_dict(),
            "vision_encoder": self.vision_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "loss": self.current_loss,
            "model_config": {
                "vocab_size": self.cfg.vocab_size,
                "n_layers": self.cfg.n_layers,
                "n_heads": self.cfg.n_heads,
                "n_kv_heads": self.cfg.n_kv_heads,
                "d_model": self.cfg.d_model,
                "d_ff": self.cfg.d_ff,
                "max_seq_len": self.cfg.max_seq_len,
            },
        }

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

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        torch.save(state, path)

        now = time.monotonic()
        if now - self._last_hf_push >= _HF_PUSH_INTERVAL:
            self._last_hf_push = now
            threading.Thread(target=self._push_to_hub, daemon=True).start()

    def _push_to_hub(self):
        if not HF_REPO_ID or not HF_TOKEN:
            return
        try:
            from huggingface_hub import HfApi

            with tempfile.TemporaryDirectory() as tmp:
                torch.save(self.model.state_dict(), os.path.join(tmp, "model.pt"))
                torch.save(self.vision_encoder.state_dict(), os.path.join(tmp, "vision_encoder.pt"))

                cfg = self.cfg
                config = {
                    "model_type": "WeBrainGPT",
                    "config_version": CONFIG_VERSION,
                    "vocab_size": cfg.vocab_size,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "n_kv_heads": cfg.n_kv_heads,
                    "d_model": cfg.d_model,
                    "d_ff": cfg.d_ff,
                    "max_seq_len": cfg.max_seq_len,
                    "dropout": cfg.dropout,
                    "rope_theta": cfg.rope_theta,
                    "training_step": self.step,
                    "current_loss": self.current_loss,
                }
                with open(os.path.join(tmp, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)

                text_params = sum(p.numel() for p in self.model.parameters())
                vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
                card = f"""---
license: apache-2.0
tags:
  - community-trained
  - distributed-compute
  - webrain
---

# The People's AI — WeBrainGPT

A collectively-trained language model built by the WeBrain community.

## Architecture

| Component | Value |
|-----------|-------|
| Type | {cfg.n_layers}-layer Transformer (RoPE + GQA + SwiGLU) + 2-layer Vision Encoder |
| Text parameters | {text_params:,} |
| Vision parameters | {vision_params:,} |
| Hidden dim | {cfg.d_model} |
| Query heads | {cfg.n_heads} |
| KV heads | {cfg.n_kv_heads} (Grouped Query Attention) |
| Feed-forward dim | {cfg.d_ff} (SwiGLU) |
| Normalization | RMSNorm |
| Position encoding | RoPE (theta={cfg.rope_theta}) |
| Vocabulary | {cfg.vocab_size} tokens (BPE) |
| Max context | {cfg.max_seq_len} |

## Training

- **Step**: {self.step:,}
- **Loss**: {self.current_loss:.4f}
- **Method**: Distributed browser-based compute via tile decomposition
- **Optimizer**: AdamW (lr={MAX_LR}, weight_decay=0.1, cosine schedule)
- **Precision**: {'Mixed (FP16)' if self._use_amp else 'FP32'}

## Usage

```python
import torch
from model import WeBrainGPT, ModelConfig

cfg = ModelConfig()
model = WeBrainGPT(cfg)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
```

## Community

Join us at [webrain.dev](https://webrain.dev) to contribute your compute.
"""
                with open(os.path.join(tmp, "README.md"), "w") as f:
                    f.write(card)

                api = HfApi(token=HF_TOKEN)
                api.upload_folder(
                    folder_path=tmp,
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    commit_message=f"Update checkpoint at step {self.step}",
                )
            log.info("Pushed checkpoint to HF Hub (%s) at step %d", HF_REPO_ID, self.step)
        except Exception as e:
            log.warning("Failed to push to HF Hub: %s", e)

    def load_training_data(self) -> str:
        if self._training_text:
            return self._training_text

        texts = []

        if os.path.isdir(DATA_DIR):
            for fname in sorted(os.listdir(DATA_DIR)):
                if fname.endswith(".txt"):
                    with open(os.path.join(DATA_DIR, fname)) as f:
                        texts.append(f.read())

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

    def get_batch(self, batch_size: int = 32, seq_len: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.load_training_data()
        tokens = self.tokenizer.encode(text)
        data = np.array(tokens, dtype=np.int64)

        ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
        x = np.stack([data[i : i + seq_len] for i in ix])
        y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix])
        return torch.tensor(x, device=self._device), torch.tensor(y, device=self._device)

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

        image_batch = torch.stack(images).to(self._device)

        seq_len = 256
        all_x = []
        all_y = []
        for caption in captions:
            tokens = self.tokenizer.encode(caption)
            if len(tokens) < 2:
                tokens = self.tokenizer.encode("an image")
            data = np.array(tokens, dtype=np.int64)
            if len(data) < seq_len + 1:
                data = np.pad(data, (0, seq_len + 1 - len(data)))
            else:
                data = data[: seq_len + 1]
            all_x.append(data[:seq_len])
            all_y.append(data[1: seq_len + 1])

        return (
            image_batch,
            torch.tensor(np.stack(all_x), device=self._device),
            torch.tensor(np.stack(all_y), device=self._device),
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

    def _update_lr(self):
        lr = _get_lr(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_layer_parameters(self, start_layer: int, end_layer: int) -> dict[str, torch.Tensor]:
        """Returns parameter tensors for a range of layers."""
        params = {}
        blocks = list(self.model.blocks)
        for i in range(start_layer, min(end_layer, len(blocks))):
            for name, param in blocks[i].named_parameters():
                params[f"blocks.{i}.{name}"] = param
        return params

    def apply_gradients(self, grad_dict: dict[str, torch.Tensor]):
        """Apply externally computed gradients to model parameters."""
        for name, grad in grad_dict.items():
            parts = name.split(".")
            param = self.model
            for part in parts:
                if part.isdigit():
                    param = list(param)[int(part)]
                else:
                    param = getattr(param, part)
            if hasattr(param, "grad") and param.grad is not None:
                param.grad += grad.to(param.device)
            else:
                param.grad = grad.to(param.device)

    def save_sharded_checkpoint(self):
        """Save model as sharded checkpoint for progressive download."""
        from app.ml.sharded_checkpoint import ShardedCheckpoint
        from app.core.config import LAYERS_PER_SHARD
        shards_dir = os.path.join(CHECKPOINT_DIR, "shards")
        ShardedCheckpoint.save_sharded(self.model, shards_dir, LAYERS_PER_SHARD)

    async def run_training_step(self, dispatch_fn) -> float:
        await self.check_training_ready_queue()
        self.model.train()
        self.vision_encoder.train()
        self._update_lr()

        use_vision = random.random() < 0.3
        if use_vision:
            vision_data = await self.get_vision_batch()
        else:
            vision_data = None

        amp_ctx = torch.amp.autocast(self._device) if self._use_amp else torch.amp.autocast(self._device, enabled=False)

        with amp_ctx:
            if vision_data is not None:
                image_batch, x, y, sub_ids = vision_data
                image_embeds = self.vision_encoder(image_batch)
                logits, _ = self.model(x, image_embeds=image_embeds)
                B, T, V = logits.shape
                loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))
                log.info("Vision training step %d, batch_size=%d", self.step, B)
            else:
                x, y = self.get_batch()
                logits, _ = self.model(x)
                B, T, V = logits.shape
                loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))

        self.optimizer.zero_grad()
        if self._use_amp and self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.vision_encoder.parameters()),
                max_norm=1.0,
            )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.vision_encoder.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

        self.step += 1
        self.current_loss = loss.item()

        if self.step % 50 == 0:
            self.save_checkpoint()
            if self.step % 200 == 0:
                try:
                    self.save_sharded_checkpoint()
                except Exception as e:
                    log.debug("Sharded checkpoint save failed: %s", e)

        if vision_data is not None:
            await self._mark_vision_trained(sub_ids)

        return self.current_loss


trainer = TrainingOrchestrator()
