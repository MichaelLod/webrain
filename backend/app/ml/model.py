import math

import torch
import torch.nn as nn


class TinyGPTConfig:
    vocab_size: int = 256  # character-level
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, n_vision_tokens: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Vision tokens are fully visible; text tokens remain causal
        mask = torch.zeros(T, T, device=x.device, dtype=torch.bool)
        if n_vision_tokens < T:
            text_mask = torch.triu(
                torch.ones(T - n_vision_tokens, T - n_vision_tokens, device=x.device),
                diagonal=1,
            ).bool()
            mask[n_vision_tokens:, n_vision_tokens:] = text_mask
        attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(torch.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor, n_vision_tokens: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), n_vision_tokens=n_vision_tokens)
        x = x + self.ff(self.ln2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: TinyGPTConfig, image_size: int = 224, patch_size: int = 16):
        super().__init__()
        self.n_patches = (image_size // patch_size) ** 2  # 196
        self.proj = nn.Conv2d(3, cfg.d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches + 1, cfg.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        B = x.shape[0]
        patches = self.proj(x)  # [B, d_model, 14, 14]
        patches = patches.flatten(2).transpose(1, 2)  # [B, 196, d_model]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)  # [B, 197, d_model]
        return x + self.pos_emb


class VisionEncoder(nn.Module):
    def __init__(self, cfg: TinyGPTConfig | None = None):
        super().__init__()
        self.cfg = cfg or TinyGPTConfig()
        self.patch_emb = PatchEmbedding(self.cfg)
        self.blocks = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(2)])
        self.ln = nn.LayerNorm(self.cfg.d_model)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"VisionEncoder: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 224, 224] -> [B, 197, d_model]
        x = self.patch_emb(images)
        for block in self.blocks:
            x = block(x)
        return self.ln(x)

    def get_weight_matrices(self) -> dict[str, torch.Tensor]:
        matrices = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                matrices[name] = param.data
        return matrices


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig | None = None):
        super().__init__()
        self.cfg = cfg or TinyGPTConfig()
        c = self.cfg

        self.token_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.max_seq_len, c.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyGPT: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor, image_embeds: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.shape
        tok_emb = self.token_emb(idx)

        n_vision_tokens = 0
        if image_embeds is not None:
            n_vision_tokens = image_embeds.shape[1]
            total_len = n_vision_tokens + T
            pos_emb = self.pos_emb(torch.arange(total_len, device=idx.device))
            x = torch.cat([image_embeds, tok_emb], dim=1) + pos_emb
        else:
            pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
            x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x, n_vision_tokens=n_vision_tokens)

        x = self.ln_f(x)
        logits = self.head(x)

        if n_vision_tokens > 0:
            logits = logits[:, n_vision_tokens:, :]

        return logits

    def get_weight_matrices(self) -> dict[str, torch.Tensor]:
        """Extract all weight matrices for tiled distribution."""
        matrices = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                matrices[name] = param.data
        return matrices
