"""Sharded checkpoint format — one file per layer group.

Enables progressive download: browsers fetch only the layers they need
rather than the entire model.
"""

from __future__ import annotations

import json
import logging
import os

import torch

log = logging.getLogger(__name__)


class ShardedCheckpoint:
    @staticmethod
    def save_sharded(model, path: str, layers_per_shard: int = 3) -> dict:
        os.makedirs(path, exist_ok=True)

        manifest = {
            "format": "webrain_sharded_v1",
            "layers_per_shard": layers_per_shard,
            "n_layers": len(list(model.blocks)),
            "shards": [],
        }

        # Save embedding shard
        embed_state = {
            "token_emb.weight": model.token_emb.weight.data.half().cpu(),
        }
        embed_path = os.path.join(path, "embed.pt")
        torch.save(embed_state, embed_path)
        manifest["shards"].append({
            "name": "embed.pt",
            "type": "embedding",
            "layers": None,
            "size_bytes": os.path.getsize(embed_path),
            "dtype": "float16",
        })

        # Save layer shards
        blocks = list(model.blocks)
        for start in range(0, len(blocks), layers_per_shard):
            end = min(start + layers_per_shard, len(blocks))
            shard_state = {}
            for i in range(start, end):
                prefix = f"blocks.{i}."
                block = blocks[i]
                for name, param in block.named_parameters():
                    shard_state[prefix + name] = param.data.half().cpu()
                for name, buf in block.named_buffers():
                    shard_state[prefix + name] = buf.half().cpu()

            shard_name = f"layers_{start}_{end - 1}.pt"
            shard_path = os.path.join(path, shard_name)
            torch.save(shard_state, shard_path)
            manifest["shards"].append({
                "name": shard_name,
                "type": "layers",
                "layers": [start, end],
                "size_bytes": os.path.getsize(shard_path),
                "dtype": "float16",
            })

        # Save head shard (ln_f + output head)
        head_state = {
            "ln_f.weight": model.ln_f.weight.data.half().cpu(),
            "head.weight": model.head.weight.data.half().cpu(),
        }
        head_path = os.path.join(path, "head.pt")
        torch.save(head_state, head_path)
        manifest["shards"].append({
            "name": "head.pt",
            "type": "head",
            "layers": None,
            "size_bytes": os.path.getsize(head_path),
            "dtype": "float16",
        })

        # Save manifest
        manifest_path = os.path.join(path, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        total_bytes = sum(s["size_bytes"] for s in manifest["shards"])
        log.info(
            "Saved sharded checkpoint: %d shards, %.1f MB total",
            len(manifest["shards"]),
            total_bytes / 1024 / 1024,
        )
        return manifest

    @staticmethod
    def load_shard(path: str, shard_name: str) -> dict[str, torch.Tensor]:
        shard_path = os.path.join(path, shard_name)
        state = torch.load(shard_path, map_location="cpu", weights_only=True)
        # Convert fp16 back to fp32
        return {k: v.float() for k, v in state.items()}

    @staticmethod
    def load_manifest(path: str) -> dict | None:
        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.exists(manifest_path):
            return None
        with open(manifest_path) as f:
            return json.load(f)
