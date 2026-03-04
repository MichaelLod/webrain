"""Bridge between PyTorch model operations and browser-distributed compute.

Provides:
- DistributedCompute: async task queue for dispatching matmul/FFN work to browsers
- FFNJob: a full SwiGLU FFN layer dispatched as a single unit to one browser
- TileJob: a matmul decomposed into tiles dispatched across multiple browsers
"""

import asyncio
import logging
import time
import uuid

import numpy as np

from app.core.config import TILE_SIZE
from app.ml.tiling import TileTask, decompose_matmul, assemble_tiles

log = logging.getLogger(__name__)

FFN_TIMEOUT_S = 10.0
TILE_TIMEOUT_S = 30.0


class FFNJob:
    """A full SwiGLU FFN layer to be computed by a single browser worker.

    The browser receives activations + cached weights, computes:
        output = down(silu(gate(x)) * up(x))
    and returns the result.
    """

    __slots__ = (
        "job_id", "layer_idx", "activations", "gate_w", "up_w", "down_w",
        "input_shape", "d_model", "d_ff", "weights_version",
        "done_event", "result", "created_at",
    )

    def __init__(
        self,
        layer_idx: int,
        activations: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
        weights_version: str,
    ):
        self.job_id = f"ffn_{uuid.uuid4().hex[:10]}"
        self.layer_idx = layer_idx
        self.activations = activations  # [M, D] float32, already flattened from [B,T,D]
        self.input_shape = activations.shape
        self.d_model = gate_w.shape[1]
        self.d_ff = gate_w.shape[0]
        # Weights stored as [D_out, D_in] (PyTorch convention)
        # Browser needs them transposed for matmul: [D_in, D_out]
        self.gate_w = gate_w
        self.up_w = up_w
        self.down_w = down_w
        self.weights_version = weights_version
        self.done_event = asyncio.Event()
        self.result: np.ndarray | None = None
        self.created_at = time.monotonic()

    def submit_result(self, output: np.ndarray):
        self.result = output
        self.done_event.set()


class TileJob:
    """A matmul A@B decomposed into tiles distributed across multiple workers."""

    __slots__ = (
        "job_id", "M", "N", "K", "tiles", "pending_tasks", "completed_tiles",
        "done_event", "result", "created_at",
    )

    def __init__(self, job_id: str, a: np.ndarray, b: np.ndarray, step: int, layer: int, op_name: str):
        self.job_id = job_id
        self.M, self.K = a.shape
        _, self.N = b.shape
        self.tiles = decompose_matmul(a, b, step, layer, op_name, TILE_SIZE)
        self.pending_tasks: dict[str, TileTask] = {t.task_id: t for t in self.tiles}
        self.completed_tiles: dict[tuple[int, int, int], np.ndarray] = {}
        self.done_event = asyncio.Event()
        self.result: np.ndarray | None = None
        self.created_at = time.monotonic()

    @property
    def is_complete(self) -> bool:
        return len(self.completed_tiles) == len(self.tiles)

    def submit_tile_result(self, task_id: str, c_tile: np.ndarray):
        task = self.pending_tasks.pop(task_id, None)
        if task:
            self.completed_tiles[(task.i, task.j, task.k)] = c_tile
            if self.is_complete:
                self.result = assemble_tiles(self.completed_tiles, self.M, self.N, TILE_SIZE)
                self.done_event.set()


class DistributedCompute:
    """Manages the queue of real compute work dispatched to browser workers.

    Two types of jobs:
    1. FFNJob — full SwiGLU FFN layer (sent to one worker, best for inference)
    2. TileJob — matmul decomposed into tiles (spread across workers, best for large matmuls)
    """

    def __init__(self):
        self.ffn_queue: asyncio.Queue[FFNJob] = asyncio.Queue()
        self.tile_queue: asyncio.Queue[TileTask] = asyncio.Queue()
        self.ffn_jobs: dict[str, FFNJob] = {}
        self.tile_jobs: dict[str, TileJob] = {}
        self.task_to_tile_job: dict[str, str] = {}
        self._weights_version: str = ""
        self._worker_count = 0

    @property
    def has_pending_work(self) -> bool:
        return not self.ffn_queue.empty() or not self.tile_queue.empty()

    @property
    def has_workers(self) -> bool:
        return self._worker_count > 0

    def set_worker_count(self, count: int):
        self._worker_count = count

    def set_weights_version(self, version: str):
        self._weights_version = version

    async def submit_ffn(
        self,
        layer_idx: int,
        activations: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> np.ndarray | None:
        """Submit an FFN layer for browser computation. Returns result or None on timeout."""
        if not self.has_workers:
            return None

        job = FFNJob(layer_idx, activations, gate_w, up_w, down_w, self._weights_version)
        self.ffn_jobs[job.job_id] = job
        await self.ffn_queue.put(job)

        try:
            await asyncio.wait_for(job.done_event.wait(), timeout=FFN_TIMEOUT_S)
            return job.result
        except asyncio.TimeoutError:
            log.warning("FFN job %s timed out after %.1fs", job.job_id, FFN_TIMEOUT_S)
            return None
        finally:
            self.ffn_jobs.pop(job.job_id, None)

    async def submit_matmul(
        self, a: np.ndarray, b: np.ndarray, step: int = 0, layer: int = 0, op_name: str = "fwd",
    ) -> np.ndarray | None:
        """Submit a tiled matmul. Returns result or None on timeout."""
        if not self.has_workers:
            return None

        job_id = f"matmul_{uuid.uuid4().hex[:10]}"
        job = TileJob(job_id, a, b, step, layer, op_name)
        self.tile_jobs[job_id] = job

        for task in job.tiles:
            self.task_to_tile_job[task.task_id] = job_id
            await self.tile_queue.put(task)

        try:
            await asyncio.wait_for(job.done_event.wait(), timeout=TILE_TIMEOUT_S)
            return job.result
        except asyncio.TimeoutError:
            log.warning("Tile job %s timed out, computing remaining locally", job_id)
            for task_id, task in job.pending_tasks.items():
                c = task.a_tile @ task.b_tile
                job.completed_tiles[(task.i, task.j, task.k)] = c
            return assemble_tiles(job.completed_tiles, job.M, job.N, TILE_SIZE)
        finally:
            self.tile_jobs.pop(job_id, None)
            for task in job.tiles:
                self.task_to_tile_job.pop(task.task_id, None)

    def get_next_ffn_job(self) -> FFNJob | None:
        """Get the next pending FFN job for a worker. Non-blocking."""
        try:
            return self.ffn_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_next_tile(self) -> TileTask | None:
        """Get the next pending tile task for a worker. Non-blocking."""
        try:
            return self.tile_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def submit_ffn_result(self, job_id: str, output: np.ndarray):
        """Called when a browser returns FFN computation result."""
        job = self.ffn_jobs.get(job_id)
        if job:
            job.submit_result(output)

    def submit_tile_result(self, task_id: str, c_tile: np.ndarray):
        """Called when a browser returns a tile result."""
        job_id = self.task_to_tile_job.get(task_id)
        if job_id and job_id in self.tile_jobs:
            self.tile_jobs[job_id].submit_tile_result(task_id, c_tile)

    def cleanup_stale_jobs(self, max_age_s: float = 60.0):
        """Remove jobs that have been pending too long."""
        now = time.monotonic()
        stale_ffn = [jid for jid, j in self.ffn_jobs.items() if now - j.created_at > max_age_s]
        for jid in stale_ffn:
            job = self.ffn_jobs.pop(jid, None)
            if job and not job.done_event.is_set():
                job.done_event.set()

        stale_tile = [jid for jid, j in self.tile_jobs.items() if now - j.created_at > max_age_s]
        for jid in stale_tile:
            job = self.tile_jobs.pop(jid, None)
            if job and not job.done_event.is_set():
                job.done_event.set()
