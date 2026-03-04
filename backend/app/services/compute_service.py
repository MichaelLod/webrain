import asyncio
import base64
import random
import time

import numpy as np
from fastapi import WebSocket

from app.core.config import TILE_SIZE
from app.ml.tiling import TileTask, decompose_matmul, assemble_tiles
from app.ml.trainer import trainer


class WorkerConnection:
    __slots__ = ("ws", "user_id", "ready", "current_task", "tasks_completed")

    def __init__(self, ws: WebSocket, user_id: int):
        self.ws = ws
        self.user_id = user_id
        self.ready = False
        self.current_task: TileTask | None = None
        self.tasks_completed = 0


class ComputeManager:
    def __init__(self):
        self.workers: dict[WebSocket, WorkerConnection] = {}
        self.pending_tasks: asyncio.Queue[TileTask] = asyncio.Queue()
        self.completed_tiles: dict[str, dict[tuple[int, int, int], np.ndarray]] = {}
        self.task_callbacks: dict[str, asyncio.Event] = {}
        self.is_training = False
        self._training_task: asyncio.Task | None = None
        # Verification: canary tasks
        self.canary_results: dict[str, np.ndarray] = {}
        # Track user results for token crediting
        self.task_user_map: dict[str, int] = {}

    @property
    def worker_count(self) -> int:
        return len(self.workers)

    async def connect(self, ws: WebSocket, user_id: int):
        self.workers[ws] = WorkerConnection(ws, user_id)
        if not self.is_training and self.worker_count >= 1:
            self.start_training()

    def disconnect(self, ws: WebSocket):
        worker = self.workers.pop(ws, None)
        if worker and worker.current_task:
            # Re-queue the task
            self.pending_tasks.put_nowait(worker.current_task)

    async def handle_message(self, ws: WebSocket, user_id: int, data: dict):
        msg_type = data.get("type")
        worker = self.workers.get(ws)
        if not worker:
            return

        if msg_type == "ready":
            worker.ready = True
            await self._dispatch_to_worker(worker)

        elif msg_type == "result":
            task_id = data["task_id"]
            c_tile_b64 = data["c_tile"]
            compute_time_ms = data.get("compute_time_ms", 0)

            c_tile_bytes = base64.b64decode(c_tile_b64)
            c_tile = np.frombuffer(c_tile_bytes, dtype=np.float32).reshape(
                TILE_SIZE, TILE_SIZE
            )

            # Verify canary if applicable
            is_verified = True
            if task_id in self.canary_results:
                expected = self.canary_results.pop(task_id)
                if not np.allclose(c_tile, expected, atol=1e-3):
                    is_verified = False

            if is_verified:
                # Store result
                task = worker.current_task
                if task:
                    op_key = f"s{task.meta['step']}_l{task.meta['layer']}_{task.meta['op']}"
                    if op_key not in self.completed_tiles:
                        self.completed_tiles[op_key] = {}
                    self.completed_tiles[op_key][(task.i, task.j, task.k)] = c_tile

                    if op_key in self.task_callbacks:
                        self.task_callbacks[op_key].set()

                # Credit tokens
                tokens = max(1, int(compute_time_ms / 2))
                await ws.send_json({"type": "credited", "tokens_earned": tokens})

            worker.current_task = None
            worker.tasks_completed += 1
            worker.ready = True
            await self._dispatch_to_worker(worker)

    async def _dispatch_to_worker(self, worker: WorkerConnection):
        if not worker.ready or self.pending_tasks.empty():
            return

        task = await self.pending_tasks.get()
        worker.ready = False
        worker.current_task = task

        # Encode tiles as base64
        a_b64 = base64.b64encode(task.a_tile.tobytes()).decode()
        b_b64 = base64.b64encode(task.b_tile.tobytes()).decode()

        is_canary = random.random() < 0.1
        if is_canary:
            # Replace with identity multiplication for verification
            canary_a = np.eye(TILE_SIZE, dtype=np.float32)
            canary_b = task.b_tile.copy()
            self.canary_results[task.task_id] = canary_b  # I @ B = B
            a_b64 = base64.b64encode(canary_a.tobytes()).decode()
            b_b64 = base64.b64encode(canary_b.tobytes()).decode()

        await worker.ws.send_json(
            {
                "type": "task",
                "task_id": task.task_id,
                "a_tile": a_b64,
                "b_tile": b_b64,
                "tile_size": TILE_SIZE,
                "position": {"i": task.i, "j": task.j, "k": task.k},
                "meta": task.meta,
            }
        )

    async def dispatch_matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        step: int,
        layer: int,
        op_name: str,
    ) -> np.ndarray:
        """Decompose, dispatch to workers, wait for all tiles, assemble."""
        tasks = decompose_matmul(a, b, step, layer, op_name, TILE_SIZE)
        op_key = f"s{step}_l{layer}_{op_name}"
        self.completed_tiles[op_key] = {}
        event = asyncio.Event()
        self.task_callbacks[op_key] = event

        expected_tiles = set()
        for task in tasks:
            expected_tiles.add((task.i, task.j, task.k))
            await self.pending_tasks.put(task)

        # Dispatch to all ready workers
        for worker in self.workers.values():
            if worker.ready:
                await self._dispatch_to_worker(worker)

        # Wait for all tiles (with timeout)
        while set(self.completed_tiles[op_key].keys()) != expected_tiles:
            event.clear()
            try:
                await asyncio.wait_for(event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                break

        M, _ = a.shape
        _, N = b.shape
        result = assemble_tiles(self.completed_tiles.pop(op_key, {}), M, N, TILE_SIZE)
        self.task_callbacks.pop(op_key, None)
        return result

    def start_training(self):
        if self.is_training:
            return
        self.is_training = True
        self._training_task = asyncio.create_task(self._training_loop())

    async def _training_loop(self):
        """Main training loop - runs training steps as long as workers are connected."""
        while self.is_training and self.worker_count > 0:
            try:
                loss = await trainer.run_training_step(self.dispatch_matmul)
                print(f"Step {trainer.step}: loss={loss:.4f}")
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Training error: {e}")
                await asyncio.sleep(1.0)

        self.is_training = False

    def stop_training(self):
        self.is_training = False
        if self._training_task:
            self._training_task.cancel()


manager = ComputeManager()
