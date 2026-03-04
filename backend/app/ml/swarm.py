"""Swarm Intelligence: Expert decomposition for dynamic model quality.

Splits each SwiGLU FFN layer's d_ff dimension into N expert slices.
The server always computes 1 expert locally (25% baseline).
Each active browser worker computes additional expert slices.
With all experts active, outputs sum to the exact full FFN output.

Key insight: SwiGLU(x) = down(silu(gate(x)) * up(x))
Since down is linear, slicing along d_ff and summing is algebraically
equivalent to the full computation.
"""

import asyncio
import logging

import numpy as np
import torch

from app.ml.distributed_ops import FFNJob

log = logging.getLogger(__name__)

N_EXPERTS = 4
EXPERT_TIMEOUT_S = 2.0


class ExpertSlices:
    """Precomputed weight slices for one transformer block's FFN."""

    __slots__ = ("gate", "up", "down")

    def __init__(self, gate: list[torch.Tensor], up: list[torch.Tensor], down: list[torch.Tensor]):
        self.gate = gate  # N_EXPERTS tensors of [slice_size, d_model]
        self.up = up      # N_EXPERTS tensors of [slice_size, d_model]
        self.down = down   # N_EXPERTS tensors of [d_model, slice_size]


class SwarmInference:
    """Expert decomposition engine for swarm-based inference.

    Slices each FFN layer into N_EXPERTS along d_ff. Server computes
    expert 0 locally; experts 1..N-1 are dispatched to browser workers.
    """

    def __init__(self):
        self._slices: list[ExpertSlices] = []
        self._slice_size = 0
        self._active_experts = 1  # server-only baseline
        self._weights_version = ""

    @property
    def collective_intelligence(self) -> float:
        return self._active_experts / N_EXPERTS

    @property
    def active_experts(self) -> int:
        return self._active_experts

    def slice_weights(self, model):
        """Precompute expert slices from model FFN weights."""
        self._slices = []
        for block in model.blocks:
            ff = block.ff
            gate_w = ff.gate.weight.data  # [d_ff, d_model]
            up_w = ff.up.weight.data      # [d_ff, d_model]
            down_w = ff.down.weight.data  # [d_model, d_ff]

            d_ff = gate_w.shape[0]
            self._slice_size = d_ff // N_EXPERTS

            gate_slices = list(gate_w.split(self._slice_size, dim=0))
            up_slices = list(up_w.split(self._slice_size, dim=0))
            down_slices = list(down_w.split(self._slice_size, dim=1))

            self._slices.append(ExpertSlices(gate_slices, up_slices, down_slices))

        log.info(
            "Sliced %d layers into %d experts (slice_size=%d)",
            len(self._slices), N_EXPERTS, self._slice_size,
        )

    def compute_expert_locally(self, layer_idx: int, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Run one expert slice on the server. Returns [B, T, d_model]."""
        slices = self._slices[layer_idx]
        gate_w = slices.gate[expert_idx]  # [slice_size, d_model]
        up_w = slices.up[expert_idx]      # [slice_size, d_model]
        down_w = slices.down[expert_idx]  # [d_model, slice_size]

        gate_out = torch.nn.functional.silu(x @ gate_w.T)
        up_out = x @ up_w.T
        hidden = gate_out * up_out
        return hidden @ down_w.T

    def create_expert_ffn_job(
        self,
        layer_idx: int,
        expert_idx: int,
        x_np: np.ndarray,
        weights_version: str,
    ) -> FFNJob:
        """Create an FFNJob with sliced weights for browser dispatch."""
        slices = self._slices[layer_idx]
        gate_w = slices.gate[expert_idx].cpu().numpy()
        up_w = slices.up[expert_idx].cpu().numpy()
        down_w = slices.down[expert_idx].cpu().numpy()

        return FFNJob(layer_idx, x_np, gate_w, up_w, down_w, weights_version)

    async def forward_swarm(self, model, idx, distributed=None, kv_caches=None, start_pos=0, image_embeds=None):
        """Layer-by-layer forward pass with expert decomposition.

        Attention: always local.
        FFN: server always computes expert 0 (25% baseline). During prefill,
        experts 1-3 are dispatched to browser workers if available. Outputs
        are summed — with all 4 experts, result is identical to full FFN.
        """
        B, T = idx.shape
        x = model.token_emb(idx)

        n_vision = 0
        if image_embeds is not None:
            n_vision = image_embeds.shape[1]
            x = torch.cat([image_embeds, x], dim=1)

        has_workers = distributed is not None and distributed.has_workers
        dispatch_to_browsers = has_workers and T > 4

        new_kv_caches = []
        experts_this_pass = 1  # at minimum, expert 0

        for i, block in enumerate(model.blocks):
            cache = kv_caches[i] if kv_caches else None

            # Attention — always local
            attn_out, new_kv = block.attn(block.ln1(x), model.rope_freqs, n_vision, cache, start_pos)
            x = x + attn_out
            new_kv_caches.append(new_kv)

            # FFN — expert decomposition
            ffn_input = block.ln2(x)

            if i < len(self._slices):
                # Expert 0: always computed locally
                ffn_output = self.compute_expert_locally(i, 0, ffn_input)

                # Experts 1-3: dispatch to browsers during prefill
                if dispatch_to_browsers:
                    ffn_np = ffn_input.detach().cpu().numpy().reshape(
                        -1, ffn_input.shape[-1],
                    ).astype(np.float32)
                    jobs: list[FFNJob] = []
                    for k in range(1, N_EXPERTS):
                        job = self.create_expert_ffn_job(
                            i, k, ffn_np, distributed._weights_version,
                        )
                        distributed.ffn_jobs[job.job_id] = job
                        await distributed.ffn_queue.put(job)
                        jobs.append(job)

                    layer_experts = 1
                    for job in jobs:
                        try:
                            await asyncio.wait_for(
                                job.done_event.wait(), timeout=EXPERT_TIMEOUT_S,
                            )
                            if job.result is not None:
                                result_t = torch.from_numpy(
                                    job.result,
                                ).to(x.device).reshape(ffn_input.shape)
                                ffn_output = ffn_output + result_t
                                layer_experts += 1
                        except asyncio.TimeoutError:
                            pass
                        finally:
                            distributed.ffn_jobs.pop(job.job_id, None)

                    experts_this_pass = max(experts_this_pass, layer_experts)
            else:
                # Slices not ready — fall back to full local FFN
                ffn_output = block.ff(ffn_input)

            x = x + ffn_output

        # Update active experts metric
        if has_workers:
            self._active_experts = min(1 + distributed._worker_count, N_EXPERTS)
        else:
            self._active_experts = 1

        x = model.ln_f(x)
        logits = model.head(x)

        if n_vision > 0:
            logits = logits[:, n_vision:, :]

        return logits, new_kv_caches
