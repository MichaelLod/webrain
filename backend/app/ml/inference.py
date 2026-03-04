from typing import Generator
import os

import torch

from app.ml.model import TinyGPT, TinyGPTConfig
from app.ml.tokenizer import CharTokenizer

_model: TinyGPT | None = None
_tokenizer = CharTokenizer()

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def get_model() -> TinyGPT:
    global _model
    if _model is None:
        _model = TinyGPT(TinyGPTConfig())
        _model.eval()
        checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            _model.load_state_dict(state["model"])
    return _model


def generate_text(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
) -> Generator[str, None, None]:
    model = get_model()
    tokens = _tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]

    idx = torch.tensor([tokens[-256:]], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(idx[:, -256:])
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            char = _tokenizer.decode([next_token.item()])
            yield char
