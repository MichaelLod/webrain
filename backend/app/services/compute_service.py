import asyncio
import base64
import hashlib
import random
import time
import uuid

import numpy as np
from fastapi import WebSocket

from app.core.config import TILE_SIZE
from app.core.database import async_session
from app.ml.distributed_ops import DistributedCompute, FFNJob
from app.ml.tiling import TileTask
from app.ml.trainer import trainer
from app.models.token import TxType
from app.services.token_service import credit_tokens


class WorkerConnection:
    __slots__ = (
        "ws", "user_id", "ready", "current_task_id", "current_task_type",
        "tasks_completed", "tokens_earned", "cached_weights_version",
    )

    def __init__(self, ws: WebSocket, user_id: int):
        self.ws = ws
        self.user_id = user_id
        self.ready = False
        self.current_task_id: str | None = None
        self.current_task_type: str | None = None  # "tile" | "ffn_forward"
        self.tasks_completed = 0
        self.tokens_earned = 0
        self.cached_weights_version: str | None = None


def _compute_weights_version(model) -> str:
    """Hash first parameter to detect model changes."""
    first_param = next(model.parameters())
    data = first_param.data[:8, :8].cpu().numpy().tobytes()
    return hashlib.md5(data).hexdigest()[:12]


class ComputeManager:
    def __init__(self):
        self.workers: dict[WebSocket, WorkerConnection] = {}
        self.is_training = False
        self._training_task: asyncio.Task | None = None
        self.canary_results: dict[str, np.ndarray] = {}
        self.distributed = DistributedCompute()
        self._weights_version = ""
        self._swarm = None

    @property
    def swarm(self):
        """Lazy access to the SwarmInference singleton from inference module."""
        if self._swarm is None:
            try:
                from app.ml.inference import get_swarm
                self._swarm = get_swarm()
            except Exception:
                pass
        return self._swarm

    @property
    def worker_count(self) -> int:
        return len(self.workers)

    def _update_weights_version(self):
        v = _compute_weights_version(trainer.model)
        if v != self._weights_version:
            self._weights_version = v
            self.distributed.set_weights_version(v)

    async def connect(self, ws: WebSocket, user_id: int):
        self.workers[ws] = WorkerConnection(ws, user_id)
        self.distributed.set_worker_count(self.worker_count)
        self._update_weights_version()
        if not self.is_training and self.worker_count >= 1:
            self.start_training()

    def disconnect(self, ws: WebSocket):
        self.workers.pop(ws, None)
        self.distributed.set_worker_count(self.worker_count)
        if self.worker_count == 0:
            self.stop_training()

    async def handle_message(self, ws: WebSocket, user_id: int, data: dict):
        msg_type = data.get("type")
        worker = self.workers.get(ws)
        if not worker:
            return

        if msg_type == "ready":
            worker.ready = True
            await self._dispatch_work(worker)

        elif msg_type == "result":
            await self._handle_result(worker, data)
            worker.current_task_id = None
            worker.current_task_type = None
            worker.ready = True
            await self._dispatch_work(worker)

        elif msg_type == "need_weights":
            await self._send_weights(worker, data.get("layer_idx", 0))

    async def _handle_result(self, worker: WorkerConnection, data: dict):
        task_id = data.get("task_id", "")
        task_type = data.get("task_type", "tile")
        compute_time_ms = data.get("compute_time_ms", 0)

        is_verified = True

        if task_type == "ffn_forward":
            # FFN result — decode and forward to the job
            try:
                output_b64 = data.get("output", "")
                output_bytes = base64.b64decode(output_b64)
                output = np.frombuffer(output_bytes, dtype=np.float32).copy()
                self.distributed.submit_ffn_result(task_id, output)
            except Exception as e:
                print(f"FFN result decode error: {e}")
                is_verified = False

        elif task_type == "tile":
            # Tile result — verify canary or forward to job
            if task_id in self.canary_results:
                try:
                    c_b64 = data.get("c_tile", "")
                    c_bytes = base64.b64decode(c_b64)
                    c_tile = np.frombuffer(c_bytes, dtype=np.float32).reshape(TILE_SIZE, TILE_SIZE)
                    expected = self.canary_results.pop(task_id)
                    if not np.allclose(c_tile, expected, atol=1e-2):
                        is_verified = False
                except Exception:
                    is_verified = False
            else:
                # Real tile result — forward to distributed system
                try:
                    c_b64 = data.get("c_tile", "")
                    c_bytes = base64.b64decode(c_b64)
                    c_tile = np.frombuffer(c_bytes, dtype=np.float32).reshape(TILE_SIZE, TILE_SIZE)
                    self.distributed.submit_tile_result(task_id, c_tile)
                except Exception as e:
                    print(f"Tile result decode error: {e}")

        if is_verified:
            # FFN tasks are worth more (more compute)
            base_reward = max(1, int(compute_time_ms / 2))
            tokens = base_reward * 3 if task_type == "ffn_forward" else base_reward
            worker.tasks_completed += 1
            worker.tokens_earned += tokens

            try:
                async with async_session() as db:
                    await credit_tokens(
                        db, worker.user_id, tokens,
                        tx_type=TxType.COMPUTE_REWARD,
                        reference_id=task_id,
                    )
                    await db.commit()
            except Exception as e:
                print(f"Token credit error: {e}")

            await worker.ws.send_json({
                "type": "credited",
                "tokens_earned": tokens,
                "total_earned": worker.tokens_earned,
                "tasks_completed": worker.tasks_completed,
            })

    async def _dispatch_work(self, worker: WorkerConnection):
        """Send the next piece of real work to a ready worker."""
        if not worker.ready:
            return

        # Priority 1: FFN jobs (latency-sensitive inference work)
        ffn_job = self.distributed.get_next_ffn_job()
        if ffn_job:
            await self._send_ffn_task(worker, ffn_job)
            return

        # Priority 2: Real tile tasks from the queue
        tile = self.distributed.get_next_tile()
        if tile:
            await self._send_real_tile(worker, tile)
            return

        # Priority 3: Synthetic tile from model weights (warmup / keeps workers busy)
        await self._send_synthetic_tile(worker)

    async def _send_ffn_task(self, worker: WorkerConnection, job: FFNJob):
        """Send a full SwiGLU FFN computation to a browser worker."""
        worker.ready = False
        worker.current_task_id = job.job_id
        worker.current_task_type = "ffn_forward"

        msg: dict = {
            "type": "task",
            "task_type": "ffn_forward",
            "task_id": job.job_id,
            "activations": base64.b64encode(job.activations.astype(np.float32).tobytes()).decode(),
            "layer_idx": job.layer_idx,
            "d_model": job.d_model,
            "d_ff": job.d_ff,
            "seq_len": job.input_shape[0],
            "weights_version": job.weights_version,
        }

        # Include weights if browser doesn't have them cached
        if worker.cached_weights_version != job.weights_version:
            msg["weights"] = {
                "gate": base64.b64encode(job.gate_w.T.astype(np.float32).tobytes()).decode(),
                "up": base64.b64encode(job.up_w.T.astype(np.float32).tobytes()).decode(),
                "down": base64.b64encode(job.down_w.T.astype(np.float32).tobytes()).decode(),
            }
            worker.cached_weights_version = job.weights_version

        await worker.ws.send_json(msg)

    async def _send_weights(self, worker: WorkerConnection, layer_idx: int):
        """Send model weights for a specific layer when browser requests them."""
        model = trainer.model
        blocks = list(model.blocks)
        if layer_idx >= len(blocks):
            return

        ff = blocks[layer_idx].ff
        gate_w = ff.gate.weight.detach().cpu().numpy()
        up_w = ff.up.weight.detach().cpu().numpy()
        down_w = ff.down.weight.detach().cpu().numpy()

        await worker.ws.send_json({
            "type": "weights",
            "layer_idx": layer_idx,
            "weights_version": self._weights_version,
            "gate": base64.b64encode(gate_w.T.astype(np.float32).tobytes()).decode(),
            "up": base64.b64encode(up_w.T.astype(np.float32).tobytes()).decode(),
            "down": base64.b64encode(down_w.T.astype(np.float32).tobytes()).decode(),
            "d_model": gate_w.shape[1],
            "d_ff": gate_w.shape[0],
        })
        worker.cached_weights_version = self._weights_version

    async def _send_real_tile(self, worker: WorkerConnection, tile: TileTask):
        """Send a real tile task from the distributed queue."""
        worker.ready = False
        worker.current_task_id = tile.task_id
        worker.current_task_type = "tile"

        a_b64 = base64.b64encode(tile.a_tile.tobytes()).decode()
        b_b64 = base64.b64encode(tile.b_tile.tobytes()).decode()

        await worker.ws.send_json({
            "type": "task",
            "task_type": "tile",
            "task_id": tile.task_id,
            "a_tile": a_b64,
            "b_tile": b_b64,
            "tile_size": TILE_SIZE,
            "position": {"i": tile.i, "j": tile.j, "k": tile.k},
            "meta": tile.meta,
        })

    async def _send_synthetic_tile(self, worker: WorkerConnection):
        """Generate a synthetic tile from real model weights (warmup work)."""
        task_id = str(uuid.uuid4())[:12]

        model = trainer.model
        weight_choices = []
        for name, param in model.named_parameters():
            if param.dim() == 2 and param.shape[0] >= TILE_SIZE and param.shape[1] >= TILE_SIZE:
                weight_choices.append((name, param.detach().cpu().numpy()))
        for name, param in trainer.vision_encoder.named_parameters():
            if param.dim() == 2 and param.shape[0] >= TILE_SIZE and param.shape[1] >= TILE_SIZE:
                weight_choices.append((f"vision.{name}", param.detach().cpu().numpy()))

        if not weight_choices:
            a_tile = np.random.randn(TILE_SIZE, TILE_SIZE).astype(np.float32) * 0.1
            b_tile = np.random.randn(TILE_SIZE, TILE_SIZE).astype(np.float32) * 0.1
        else:
            name, weight = random.choice(weight_choices)
            r = random.randint(0, max(0, weight.shape[0] - TILE_SIZE))
            c = random.randint(0, max(0, weight.shape[1] - TILE_SIZE))
            a_tile = weight[r:r + TILE_SIZE, c:c + TILE_SIZE].copy()
            b_tile = np.random.randn(TILE_SIZE, TILE_SIZE).astype(np.float32) * 0.02

            if a_tile.shape != (TILE_SIZE, TILE_SIZE):
                padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                padded[:a_tile.shape[0], :a_tile.shape[1]] = a_tile
                a_tile = padded

        # Canary check: 1 in 10 synthetic tasks
        is_canary = random.random() < 0.1
        if is_canary:
            canary_a = np.eye(TILE_SIZE, dtype=np.float32)
            self.canary_results[task_id] = b_tile.copy()
            a_tile = canary_a

        a_b64 = base64.b64encode(a_tile.tobytes()).decode()
        b_b64 = base64.b64encode(b_tile.tobytes()).decode()

        worker.ready = False
        worker.current_task_id = task_id
        worker.current_task_type = "tile"

        await worker.ws.send_json({
            "type": "task",
            "task_type": "tile",
            "task_id": task_id,
            "a_tile": a_b64,
            "b_tile": b_b64,
            "tile_size": TILE_SIZE,
            "position": {"i": 0, "j": 0, "k": 0},
            "meta": {"step": trainer.step, "layer": 0, "op": "warmup"},
        })

    def start_training(self):
        if self.is_training:
            return
        self.is_training = True
        self._training_task = asyncio.create_task(self._training_loop())

    async def _training_loop(self):
        while self.is_training and self.worker_count > 0:
            try:
                loss = await trainer.run_training_step(None)
                self._update_weights_version()
                print(f"Step {trainer.step}: loss={loss:.4f}")
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Training error: {e}")
                await asyncio.sleep(2.0)

            # Periodic cleanup
            if trainer.step % 10 == 0:
                self.distributed.cleanup_stale_jobs()

        self.is_training = False

    def stop_training(self):
        self.is_training = False
        if self._training_task:
            self._training_task.cancel()


manager = ComputeManager()
