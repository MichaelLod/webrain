import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA: fewer KV heads than query heads
    d_model: int = 512
    d_ff: int = 1376  # SwiGLU: ~8/3 * d_model, rounded to 32
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0


# Legacy config alias for checkpoint migration
class TinyGPTConfig:
    vocab_size: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor as complex exponentials."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # xq, xk: [B, n_heads, T, d_head]
    B, H, T, D = xq.shape
    xq_complex = torch.view_as_complex(xq.float().reshape(B, H, T, D // 2, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(B, H, T, D // 2, 2))

    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]

    xq_out = torch.view_as_real(xq_complex * freqs).reshape(B, H, T, D)
    xk_out = torch.view_as_real(xk_complex * freqs).reshape(B, H, T, D)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # how many query heads per KV head

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * self.d_head, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        n_vision_tokens: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE only to text tokens (skip vision prefix)
        if n_vision_tokens > 0 and T > n_vision_tokens:
            text_q = q[:, :, n_vision_tokens:, :]
            text_k = k[:, :, n_vision_tokens:, :]
            # Use positions starting from 0 for text tokens
            text_len = T - n_vision_tokens
            text_freqs = rope_freqs[:text_len]
            text_q_rope, text_k_rope = apply_rope(
                text_q[:, :self.n_kv_heads, :, :] if self.n_kv_heads < self.n_heads else text_q,
                text_k,
                text_freqs.unsqueeze(0).unsqueeze(0) if text_freqs.dim() == 1 else text_freqs,
            )
            # For GQA, apply rope to all query heads
            q_text_all = q[:, :, n_vision_tokens:, :]
            q_text_complex = torch.view_as_complex(q_text_all.float().reshape(B, self.n_heads, text_len, self.d_head // 2, 2))
            freqs_for_q = rope_freqs[:text_len].unsqueeze(0).unsqueeze(0)
            q_text_rope = torch.view_as_real(q_text_complex * freqs_for_q).reshape(B, self.n_heads, text_len, self.d_head).type_as(q)
            q = torch.cat([q[:, :, :n_vision_tokens, :], q_text_rope], dim=2)
            k = torch.cat([k[:, :, :n_vision_tokens, :], text_k_rope], dim=2)
        elif n_vision_tokens == 0:
            # Pure text: apply RoPE to all positions
            pos_freqs = rope_freqs[start_pos:start_pos + T]
            q, k = apply_rope(q, k, pos_freqs)

        # KV cache for incremental decoding
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        S = k.shape[2]  # total sequence length including cache
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask for text tokens
        if T > 1:  # not single-token generation
            mask = torch.zeros(T, S, device=x.device, dtype=torch.bool)
            if n_vision_tokens > 0 and n_vision_tokens < T:
                text_len = T - n_vision_tokens
                text_mask = torch.triu(
                    torch.ones(text_len, text_len, device=x.device), diagonal=1
                ).bool()
                mask[n_vision_tokens:, n_vision_tokens:] = text_mask
            elif n_vision_tokens == 0:
                causal = torch.triu(torch.ones(T, S, device=x.device), diagonal=S - T + 1).bool()
                mask = causal
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out), new_kv_cache


class SwiGLUFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ff = SwiGLUFeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        n_vision_tokens: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln1(x), rope_freqs, n_vision_tokens, kv_cache, start_pos)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, image_size: int = 224, patch_size: int = 16):
        super().__init__()
        self.n_patches = (image_size // patch_size) ** 2  # 196
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        return x + self.pos_emb


class VisionEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        d = self.cfg.d_model
        self.patch_emb = PatchEmbedding(d)
        # Vision encoder uses standard attention (no GQA, no RoPE — it has learned pos embeddings)
        self.blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.ModuleDict({
                "ln1": RMSNorm(d),
                "attn_qkv": nn.Linear(d, 3 * d, bias=False),
                "attn_out": nn.Linear(d, d, bias=False),
                "ln2": RMSNorm(d),
                "ff": SwiGLUFeedForward(self.cfg),
            })
            self.blocks.append(block)
        self.ln = RMSNorm(d)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"VisionEncoder: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_model // n_heads
        x = self.patch_emb(images)
        B, T, C = x.shape
        for block in self.blocks:
            residual = x
            h = block["ln1"](x)
            qkv = block["attn_qkv"](h).reshape(B, T, 3, n_heads, d_head).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, T, C)
            x = residual + block["attn_out"](out)
            x = x + block["ff"](block["ln2"](x))
        return self.ln(x)

    def get_weight_matrices(self) -> dict[str, torch.Tensor]:
        matrices = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                matrices[name] = param.data
        return matrices


class WeBrainGPT(nn.Module):
    """45M-parameter modern transformer with RoPE, GQA, SwiGLU, and RMSNorm."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        self.token_emb = nn.Embedding(c.vocab_size, c.d_model)
        # No positional embedding — using RoPE instead
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.ln_f = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(c.d_model // c.n_heads, c.max_seq_len, c.rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"WeBrainGPT: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        B, T = idx.shape
        tok_emb = self.token_emb(idx)

        n_vision_tokens = 0
        if image_embeds is not None:
            n_vision_tokens = image_embeds.shape[1]
            x = torch.cat([image_embeds, tok_emb], dim=1)
        else:
            x = tok_emb

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, self.rope_freqs, n_vision_tokens, cache, start_pos)
            new_kv_caches.append(new_kv)

        x = self.ln_f(x)
        logits = self.head(x)

        if n_vision_tokens > 0:
            logits = logits[:, n_vision_tokens:, :]

        return logits, new_kv_caches

    def get_weight_matrices(self) -> dict[str, torch.Tensor]:
        matrices = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                matrices[name] = param.data
        return matrices


# Alias for backward compatibility — trainer and other code can use this
TinyGPT = WeBrainGPT
