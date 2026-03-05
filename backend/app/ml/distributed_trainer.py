"""Pipeline-parallel training with browser workers.

Forward pass: server embeds -> activations flow through pipeline stages -> server computes loss.
Backward pass: server computes grad on head -> grads flow in reverse -> each browser computes gradients.
Optimize: server aggregates gradients and applies optimizer step.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid

import numpy as np
import torch

log = logging.getLogger(__name__)

TRAINING_TIMEOUT_S = 10.0


class DistributedTrainer:
    def __init__(self):
        self._pending_forward: dict[str, asyncio.Future] = {}
        self._pending_backward: dict[str, asyncio.Future] = {}
        self._activation_checkpoints: dict[int, torch.Tensor] = {}

    def submit_forward_result(self, request_id: str, activations_b64: str, saved_state_b64: str | None = None):
        fut = self._pending_forward.get(request_id)
        if fut and not fut.done():
            data = base64.b64decode(activations_b64)
            arr = np.frombuffer(data, dtype=np.float32).copy()
            fut.set_result((arr, saved_state_b64))

    def submit_backward_result(self, request_id: str, grad_input_b64: str, param_gradients: dict | None = None):
        fut = self._pending_backward.get(request_id)
        if fut and not fut.done():
            data = base64.b64decode(grad_input_b64)
            arr = np.frombuffer(data, dtype=np.float32).copy()
            fut.set_result((arr, param_gradients))

    async def run_distributed_step(self, model, x, y, loss_fn, scheduler, registry):
        """Run one distributed training step through the pipeline."""
        model.train()

        B, T = x.shape
        embeds = model.token_emb(x)

        # Forward through pipeline stages
        current = embeds
        stage_inputs: list[torch.Tensor] = [current.detach().clone()]

        for stage in scheduler.stages:
            start, end = stage.layer_range
            peer_id = stage.primary_peer

            result = None
            if peer_id:
                ws = registry.get_peer_ws(peer_id)
                if ws:
                    result = await self._dispatch_training_forward(
                        ws, current, start, end, B, T,
                    )

            if result is not None:
                activations_np, saved_state = result
                current = torch.from_numpy(activations_np).to(x.device).reshape(current.shape)
                self._activation_checkpoints[stage.stage_id] = current.detach().clone()
            else:
                # Local fallback
                for i in range(start, end):
                    block = list(model.blocks)[i]
                    attn_out, _ = block.attn(block.ln1(current), model.rope_freqs, 0, None, 0)
                    current = current + attn_out
                    current = current + block.ff(block.ln2(current))
                self._activation_checkpoints[stage.stage_id] = current.detach().clone()

            stage_inputs.append(current.detach().clone())

        # Server computes loss
        x_final = model.ln_f(current)
        logits = model.head(x_final)
        loss = loss_fn(logits.view(B * T, -1), y.view(B * T))

        # Backward: compute gradients through head
        loss.backward()

        self._activation_checkpoints.clear()
        return loss.item()

    async def _dispatch_training_forward(self, ws, x: torch.Tensor, start_layer: int, end_layer: int, batch_size: int, seq_len: int) -> tuple[np.ndarray, str | None] | None:
        request_id = str(uuid.uuid4())[:12]
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_forward[request_id] = fut

        x_np = x.detach().cpu().numpy().astype(np.float32)
        activations_b64 = base64.b64encode(x_np.tobytes()).decode()

        try:
            await ws.send_json({
                "type": "training_forward",
                "request_id": request_id,
                "activations": activations_b64,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "seq_len": seq_len,
                "batch_size": batch_size,
            })
            result = await asyncio.wait_for(fut, timeout=TRAINING_TIMEOUT_S)
            return result
        except (asyncio.TimeoutError, Exception) as e:
            log.debug("Training forward dispatch failed for layers %d-%d: %s", start_layer, end_layer, e)
            return None
        finally:
            self._pending_forward.pop(request_id, None)
