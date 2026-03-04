from typing import Generator
import os

import torch

from app.ml.model import TinyGPT, TinyGPTConfig, VisionEncoder
from app.ml.tokenizer import CharTokenizer

_model: TinyGPT | None = None
_vision_encoder: VisionEncoder | None = None
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


def get_vision_encoder() -> VisionEncoder:
    global _vision_encoder
    if _vision_encoder is None:
        _vision_encoder = VisionEncoder(TinyGPTConfig())
        _vision_encoder.eval()
        checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "vision_encoder" in state:
                _vision_encoder.load_state_dict(state["vision_encoder"])
    return _vision_encoder


def generate_text(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    image: torch.Tensor | None = None,
) -> Generator[str, None, None]:
    model = get_model()
    tokens = _tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]

    image_embeds = None
    if image is not None:
        vision_enc = get_vision_encoder()
        with torch.no_grad():
            image_embeds = vision_enc(image.unsqueeze(0))  # [1, 197, d_model]
        n_vision = image_embeds.shape[1]
        max_text_len = 512 - n_vision
    else:
        max_text_len = 512

    idx = torch.tensor([tokens[-max_text_len:]], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(idx[:, -max_text_len:], image_embeds=image_embeds)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            char = _tokenizer.decode([next_token.item()])
            yield char
