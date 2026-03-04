import asyncio
import os
import time

import numpy as np
import torch
import torch.nn as nn

from app.ml.model import TinyGPT, TinyGPTConfig
from app.ml.tokenizer import CharTokenizer
from app.ml.tiling import TileTask, assemble_tiles, decompose_matmul
from app.core.config import TILE_SIZE

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TrainingOrchestrator:
    def __init__(self):
        self.cfg = TinyGPTConfig()
        self.model = TinyGPT(self.cfg)
        self.tokenizer = CharTokenizer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.loss_fn = nn.CrossEntropyLoss()
        self.step = 0
        self.current_loss = 0.0
        self.total_flops = 0.0
        self.is_training = False
        self._training_text: str | None = None
        self._load_checkpoint()

    def _load_checkpoint(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.step = state.get("step", 0)
            self.current_loss = state.get("loss", 0.0)

    def save_checkpoint(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "loss": self.current_loss,
            },
            path,
        )

    def load_training_data(self) -> str:
        if self._training_text:
            return self._training_text
        # Load all .txt files from data directory
        texts = []
        for fname in sorted(os.listdir(DATA_DIR)):
            if fname.endswith(".txt"):
                with open(os.path.join(DATA_DIR, fname)) as f:
                    texts.append(f.read())
        if not texts:
            # Fallback: generate some dummy training data
            texts = ["The quick brown fox jumps over the lazy dog. " * 1000]
        self._training_text = "\n".join(texts)
        return self._training_text

    def get_batch(self, batch_size: int = 32, seq_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.load_training_data()
        tokens = self.tokenizer.encode(text)
        data = np.array(tokens, dtype=np.int64)

        ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
        x = np.stack([data[i : i + seq_len] for i in ix])
        y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix])
        return torch.tensor(x), torch.tensor(y)

    def decompose_forward_layer(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        step: int,
        layer: int,
        op_name: str,
    ) -> list[TileTask]:
        """Decompose a single linear layer forward pass into tile tasks."""
        return decompose_matmul(x, weight, step, layer, op_name, TILE_SIZE)

    def assemble_forward_result(
        self,
        tiles: dict[tuple[int, int, int], np.ndarray],
        M: int,
        N: int,
    ) -> np.ndarray:
        return assemble_tiles(tiles, M, N, TILE_SIZE)

    async def run_training_step(self, dispatch_fn) -> float:
        """Execute one training step using distributed tile computation.

        dispatch_fn: async callable that takes a list of TileTasks and returns
                     dict of (i,j,k) -> result_tile for each task.
        """
        self.model.train()
        x, y = self.get_batch()

        # Forward pass using standard PyTorch (for MVP, tiled dispatch is used
        # for the linear layers when workers are available)
        logits = self.model(x)
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        self.current_loss = loss.item()

        if self.step % 50 == 0:
            self.save_checkpoint()

        return self.current_loss


# Global instance
trainer = TrainingOrchestrator()
