"""Pipeline-parallel inference engine.

Distributes transformer layers across browser peers. Server handles
embedding and output head; peers handle assigned layer ranges.
Falls back to local computation when peers are unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid

import numpy as np
import torch

log = logging.getLogger(__name__)

PIPELINE_TIMEOUT_S = 5.0


class PipelineInference:
    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}

    def submit_result(self, request_id: str, activations_b64: str):
        fut = self._pending.get(request_id)
        if fut and not fut.done():
            data = base64.b64decode(activations_b64)
            arr = np.frombuffer(data, dtype=np.float32).copy()
            fut.set_result(arr)

    async def forward_pipeline(self, model, idx, scheduler, registry, kv_caches=None, start_pos=0, image_embeds=None):
        B, T = idx.shape
        x = model.token_emb(idx)

        n_vision = 0
        if image_embeds is not None:
            n_vision = image_embeds.shape[1]
            x = torch.cat([image_embeds, x], dim=1)

        if not scheduler.is_active:
            return self._compute_all_locally(model, x, n_vision, kv_caches, start_pos)

        new_kv_caches = []
        current_layer = 0

        for stage in scheduler.stages:
            start, end = stage.layer_range
            peer_id = stage.primary_peer

            # Try dispatching to the primary peer
            result = None
            if peer_id:
                ws = registry.get_peer_ws(peer_id)
                if ws:
                    result = await self._dispatch_to_peer(
                        ws, x, start, end, start_pos, n_vision,
                    )

            # Try backup peers
            if result is None:
                for bp in stage.backup_peers:
                    ws = registry.get_peer_ws(bp)
                    if ws:
                        result = await self._dispatch_to_peer(
                            ws, x, start, end, start_pos, n_vision,
                        )
                        if result is not None:
                            break

            # Fallback: compute locally
            if result is not None:
                x = torch.from_numpy(result).to(x.device).reshape(x.shape)
                # Placeholder KV caches for pipeline-computed layers
                for _ in range(start, end):
                    new_kv_caches.append(None)
            else:
                x, layer_kvs = self._compute_layers_locally(
                    model, x, start, end, n_vision, kv_caches, start_pos,
                )
                new_kv_caches.extend(layer_kvs)

            current_layer = end

        x = model.ln_f(x)
        logits = model.head(x)
        if n_vision > 0:
            logits = logits[:, n_vision:, :]
        return logits, new_kv_caches

    async def _dispatch_to_peer(self, ws, x: torch.Tensor, start_layer: int, end_layer: int, start_pos: int, n_vision: int) -> np.ndarray | None:
        request_id = str(uuid.uuid4())[:12]
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[request_id] = fut

        x_np = x.detach().cpu().numpy().astype(np.float32)
        activations_b64 = base64.b64encode(x_np.tobytes()).decode()

        try:
            await ws.send_json({
                "type": "pipeline_forward",
                "request_id": request_id,
                "activations": activations_b64,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "seq_len": x.shape[1],
                "d_model": x.shape[2] if x.dim() == 3 else x.shape[-1],
                "start_pos": start_pos,
            })
            result = await asyncio.wait_for(fut, timeout=PIPELINE_TIMEOUT_S)
            return result
        except (asyncio.TimeoutError, Exception) as e:
            log.debug("Pipeline dispatch failed for layers %d-%d: %s", start_layer, end_layer, e)
            return None
        finally:
            self._pending.pop(request_id, None)

    def _compute_layers_locally(self, model, x, start, end, n_vision, kv_caches, start_pos):
        new_kvs = []
        blocks = list(model.blocks)
        for i in range(start, end):
            block = blocks[i]
            cache = kv_caches[i] if kv_caches and i < len(kv_caches) else None
            attn_out, new_kv = block.attn(block.ln1(x), model.rope_freqs, n_vision, cache, start_pos)
            x = x + attn_out
            new_kvs.append(new_kv)
            x = x + block.ff(block.ln2(x))
        return x, new_kvs

    def _compute_all_locally(self, model, x, n_vision, kv_caches, start_pos):
        new_kv_caches = []
        for i, block in enumerate(model.blocks):
            cache = kv_caches[i] if kv_caches and i < len(kv_caches) else None
            attn_out, new_kv = block.attn(block.ln1(x), model.rope_freqs, n_vision, cache, start_pos)
            x = x + attn_out
            new_kv_caches.append(new_kv)
            x = x + block.ff(block.ln2(x))
        x = model.ln_f(x)
        logits = model.head(x)
        if n_vision > 0:
            logits = logits[:, n_vision:, :]
        return logits, new_kv_caches
