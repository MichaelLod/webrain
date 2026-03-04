import asyncio
import base64
import random
import time
import uuid

import numpy as np
from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import TILE_SIZE
from app.core.database import async_session
from app.ml.tiling import TileTask, decompose_matmul, assemble_tiles
from app.ml.trainer import trainer
from app.models.token import TxType
from app.services.token_service import credit_tokens


class WorkerConnection:
    __slots__ = ("ws", "user_id", "ready", "current_task", "tasks_completed", "tokens_earned")

    def __init__(self, ws: WebSocket, user_id: int):
        self.ws = ws
        self.user_id = user_id
        self.ready = False
        self.current_task: TileTask | None = None
        self.tasks_completed = 0
        self.tokens_earned = 0


class ComputeManager:
    def __init__(self):
        self.workers: dict[WebSocket, WorkerConnection] = {}
        self.is_training = False
        self._training_task: asyncio.Task | None = None
        self.canary_results: dict[str, np.ndarray] = {}

    @property
    def worker_count(self) -> int:
        return len(self.workers)

    async def connect(self, ws: WebSocket, user_id: int):
        self.workers[ws] = WorkerConnection(ws, user_id)
        if not self.is_training and self.worker_count >= 1:
            self.start_training()

    def disconnect(self, ws: WebSocket):
        self.workers.pop(ws, None)
        if self.worker_count == 0:
            self.stop_training()

    async def handle_message(self, ws: WebSocket, user_id: int, data: dict):
        msg_type = data.get("type")
        worker = self.workers.get(ws)
        if not worker:
            return

        if msg_type == "ready":
            worker.ready = True
            # Send a tile task immediately
            await self._send_tile_task(worker)

        elif msg_type == "result":
            task_id = data.get("task_id", "")
            compute_time_ms = data.get("compute_time_ms", 0)

            # Verify canary if applicable
            is_verified = True
            if task_id in self.canary_results:
                try:
                    c_tile_b64 = data.get("c_tile", "")
                    c_tile_bytes = base64.b64decode(c_tile_b64)
                    c_tile = np.frombuffer(c_tile_bytes, dtype=np.float32).reshape(
                        TILE_SIZE, TILE_SIZE
                    )
                    expected = self.canary_results.pop(task_id)
                    if not np.allclose(c_tile, expected, atol=1e-2):
                        is_verified = False
                except Exception:
                    is_verified = False

            if is_verified:
                tokens = max(1, int(compute_time_ms / 2))
                worker.tasks_completed += 1
                worker.tokens_earned += tokens

                # Credit tokens in DB
                try:
                    async with async_session() as db:
                        await credit_tokens(
                            db, user_id, tokens,
                            tx_type=TxType.COMPUTE_REWARD,
                            reference_id=task_id,
                        )
                        await db.commit()
                except Exception as e:
                    print(f"Token credit error: {e}")

                await ws.send_json({
                    "type": "credited",
                    "tokens_earned": tokens,
                    "total_earned": worker.tokens_earned,
                    "tasks_completed": worker.tasks_completed,
                })

            worker.current_task = None
            worker.ready = True
            # Send next task
            await self._send_tile_task(worker)

    async def _send_tile_task(self, worker: WorkerConnection):
        """Generate and send a tile matmul task to the worker."""
        if not worker.ready:
            return

        task_id = str(uuid.uuid4())[:12]

        # Generate a real tile from the model's current weights
        model = trainer.model
        blocks = list(model.blocks)
        layer_idx = random.randint(0, len(blocks) - 1) if blocks else 0

        # Collect all 2D weight matrices >= TILE_SIZE from the model
        weight_choices = []
        for name, param in model.named_parameters():
            if param.dim() == 2 and param.shape[0] >= TILE_SIZE and param.shape[1] >= TILE_SIZE:
                weight_choices.append((name, param.detach().numpy()))

        if not weight_choices:
            a_tile = np.random.randn(TILE_SIZE, TILE_SIZE).astype(np.float32) * 0.1
            b_tile = np.random.randn(TILE_SIZE, TILE_SIZE).astype(np.float32) * 0.1
        else:
            name, weight = random.choice(weight_choices)
            r = random.randint(0, max(0, weight.shape[0] - TILE_SIZE))
            c = random.randint(0, max(0, weight.shape[1] - TILE_SIZE))
            a_tile = weight[r:r+TILE_SIZE, c:c+TILE_SIZE].copy()
            b_tile = np.random.randn(TILE_SIZE, TILE_SIZE).astype(np.float32) * 0.02

            if a_tile.shape != (TILE_SIZE, TILE_SIZE):
                padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                padded[:a_tile.shape[0], :a_tile.shape[1]] = a_tile
                a_tile = padded

        # Canary check: 1 in 10 tasks
        is_canary = random.random() < 0.1
        if is_canary:
            canary_a = np.eye(TILE_SIZE, dtype=np.float32)
            self.canary_results[task_id] = b_tile.copy()  # I @ B = B
            a_tile = canary_a

        a_b64 = base64.b64encode(a_tile.tobytes()).decode()
        b_b64 = base64.b64encode(b_tile.tobytes()).decode()

        task = TileTask(
            task_id=task_id,
            a_tile=a_tile,
            b_tile=b_tile,
            i=0, j=0, k=0,
            meta={"step": trainer.step, "layer": layer_idx, "op": "fwd"},
        )
        worker.current_task = task
        worker.ready = False

        await worker.ws.send_json({
            "type": "task",
            "task_id": task_id,
            "a_tile": a_b64,
            "b_tile": b_b64,
            "tile_size": TILE_SIZE,
            "position": {"i": 0, "j": 0, "k": 0},
            "meta": {"step": trainer.step, "layer": layer_idx, "op": "fwd"},
        })

    def start_training(self):
        if self.is_training:
            return
        self.is_training = True
        self._training_task = asyncio.create_task(self._training_loop())

    async def _training_loop(self):
        """Run training steps on the server while workers compute tiles."""
        while self.is_training and self.worker_count > 0:
            try:
                loss = await trainer.run_training_step(None)
                print(f"Step {trainer.step}: loss={loss:.4f}")
                # Pace the training so workers can keep up
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Training error: {e}")
                await asyncio.sleep(2.0)

        self.is_training = False

    def stop_training(self):
        self.is_training = False
        if self._training_task:
            self._training_task.cancel()


manager = ComputeManager()
