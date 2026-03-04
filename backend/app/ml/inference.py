import logging
import os
from typing import AsyncGenerator, Generator

import torch

from app.ml.model import WeBrainGPT, ModelConfig, VisionEncoder
from app.ml.swarm import SwarmInference
from app.ml.tokenizer import BPETokenizer, CharTokenizer, get_tokenizer

log = logging.getLogger(__name__)

_model: WeBrainGPT | None = None
_vision_encoder: VisionEncoder | None = None
_tokenizer: BPETokenizer | CharTokenizer | None = None
_swarm: SwarmInference | None = None

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

CONFIG_VERSION = 2


def _get_tokenizer() -> BPETokenizer | CharTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = get_tokenizer()
    return _tokenizer


def get_model() -> WeBrainGPT:
    global _model
    if _model is None:
        cfg = ModelConfig()
        _model = WeBrainGPT(cfg)
        _model.eval()
        checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            version = state.get("config_version", 1)
            if version == CONFIG_VERSION:
                _model.load_state_dict(state["model"])
                log.info("Loaded model checkpoint (version %d)", version)
            else:
                log.warning("Checkpoint version %d != current %d, using fresh model", version, CONFIG_VERSION)
    return _model


def get_vision_encoder() -> VisionEncoder:
    global _vision_encoder
    if _vision_encoder is None:
        cfg = ModelConfig()
        _vision_encoder = VisionEncoder(cfg)
        _vision_encoder.eval()
        checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            version = state.get("config_version", 1)
            if version == CONFIG_VERSION and "vision_encoder" in state:
                _vision_encoder.load_state_dict(state["vision_encoder"])
    return _vision_encoder


def _get_distributed():
    """Get the distributed compute instance from compute_service, if available."""
    try:
        from app.services.compute_service import manager
        return manager.distributed
    except Exception:
        return None


def get_swarm() -> SwarmInference:
    """Lazy-init the swarm inference singleton, slicing weights from the model."""
    global _swarm
    if _swarm is None:
        model = get_model()
        _swarm = SwarmInference()
        _swarm.slice_weights(model)
    return _swarm


async def _forward_hybrid(model, idx, distributed, kv_caches=None, start_pos=0, image_embeds=None):
    """Layer-by-layer forward pass with optional browser FFN offloading.

    During prefill (T > 1), FFN layers are dispatched to browser workers.
    During decode (T = 1), everything runs locally (not worth the round-trip).
    """
    B, T = idx.shape
    x = model.token_emb(idx)

    n_vision = 0
    if image_embeds is not None:
        n_vision = image_embeds.shape[1]
        x = torch.cat([image_embeds, x], dim=1)

    use_distributed = distributed is not None and distributed.has_workers and T > 4

    new_kv_caches = []
    for i, block in enumerate(model.blocks):
        cache = kv_caches[i] if kv_caches else None

        # Attention — always local (needs KV cache, sequential)
        attn_out, new_kv = block.attn(block.ln1(x), model.rope_freqs, n_vision, cache, start_pos)
        x = x + attn_out
        new_kv_caches.append(new_kv)

        # FFN — dispatch to browser during prefill
        ffn_input = block.ln2(x)

        if use_distributed:
            import numpy as np
            ffn_np = ffn_input.detach().cpu().numpy().reshape(-1, ffn_input.shape[-1]).astype(np.float32)
            gate_w = block.ff.gate.weight.detach().cpu().numpy()
            up_w = block.ff.up.weight.detach().cpu().numpy()
            down_w = block.ff.down.weight.detach().cpu().numpy()

            result = await distributed.submit_ffn(i, ffn_np, gate_w, up_w, down_w)
            if result is not None:
                ffn_output = torch.from_numpy(result).to(x.device).reshape(ffn_input.shape)
            else:
                ffn_output = block.ff(ffn_input)
        else:
            ffn_output = block.ff(ffn_input)

        x = x + ffn_output

    x = model.ln_f(x)
    logits = model.head(x)

    if n_vision > 0:
        logits = logits[:, n_vision:, :]

    return logits, new_kv_caches


def generate_text(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    image: torch.Tensor | None = None,
) -> Generator[str, None, None]:
    """Synchronous text generation — fully local, no browser offloading."""
    model = get_model()
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]

    cfg = model.cfg
    image_embeds = None
    if image is not None:
        vision_enc = get_vision_encoder()
        with torch.no_grad():
            image_embeds = vision_enc(image.unsqueeze(0))
        n_vision = image_embeds.shape[1]
        max_text_len = cfg.max_seq_len - n_vision
    else:
        max_text_len = cfg.max_seq_len

    tokens = tokens[-max_text_len:]
    idx = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        logits, kv_caches = model(idx, image_embeds=image_embeds)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        yield tokenizer.decode([next_token.item()])

        seq_len = idx.shape[1] + (image_embeds.shape[1] if image_embeds is not None else 0)
        for _ in range(max_tokens - 1):
            logits, kv_caches = model(next_token, kv_caches=kv_caches, start_pos=seq_len)
            seq_len += 1
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            yield tokenizer.decode([next_token.item()])


async def generate_text_hybrid(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    image: torch.Tensor | None = None,
) -> AsyncGenerator[str, None]:
    """Async text generation — offloads FFN prefill to browser workers.

    Prefill phase: attention runs locally, FFN layers dispatched to browsers.
    Decode phase: fully local (single-token FFN isn't worth the network round-trip).
    Falls back to fully local if no workers are available.
    """
    model = get_model()
    tokenizer = _get_tokenizer()
    distributed = _get_distributed()
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]

    cfg = model.cfg
    image_embeds = None
    if image is not None:
        vision_enc = get_vision_encoder()
        with torch.no_grad():
            image_embeds = vision_enc(image.unsqueeze(0))
        n_vision = image_embeds.shape[1]
        max_text_len = cfg.max_seq_len - n_vision
    else:
        max_text_len = cfg.max_seq_len

    tokens = tokens[-max_text_len:]
    idx = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        # Prefill — distributed FFN when workers available
        logits, kv_caches = await _forward_hybrid(
            model, idx, distributed, image_embeds=image_embeds,
        )
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        yield tokenizer.decode([next_token.item()])

        # Decode — always local (single-token, fast)
        seq_len = idx.shape[1] + (image_embeds.shape[1] if image_embeds is not None else 0)
        for _ in range(max_tokens - 1):
            logits, kv_caches = model(next_token, kv_caches=kv_caches, start_pos=seq_len)
            seq_len += 1
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            yield tokenizer.decode([next_token.item()])


async def generate_text_swarm(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    image: torch.Tensor | None = None,
) -> AsyncGenerator[str, None]:
    """Swarm text generation — expert-decomposed FFN with browser workers.

    Prefill: server computes expert 0 locally, dispatches experts 1-3 to browsers.
    Decode: server computes expert 0 only (single-token, fast baseline).
    Quality scales with active browser workers (25% to 100%).
    """
    model = get_model()
    tokenizer = _get_tokenizer()
    distributed = _get_distributed()
    swarm = get_swarm()
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]

    cfg = model.cfg
    image_embeds = None
    if image is not None:
        vision_enc = get_vision_encoder()
        with torch.no_grad():
            image_embeds = vision_enc(image.unsqueeze(0))
        n_vision = image_embeds.shape[1]
        max_text_len = cfg.max_seq_len - n_vision
    else:
        max_text_len = cfg.max_seq_len

    tokens = tokens[-max_text_len:]
    idx = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        # Prefill — expert-decomposed FFN with browser dispatch
        logits, kv_caches = await swarm.forward_swarm(
            model, idx, distributed, image_embeds=image_embeds,
        )
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        yield tokenizer.decode([next_token.item()])

        # Decode — expert 0 only via swarm (T=1, no browser dispatch)
        seq_len = idx.shape[1] + (image_embeds.shape[1] if image_embeds is not None else 0)
        for _ in range(max_tokens - 1):
            logits, kv_caches = await swarm.forward_swarm(
                model, next_token, distributed,
                kv_caches=kv_caches, start_pos=seq_len,
            )
            seq_len += 1
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            yield tokenizer.decode([next_token.item()])
