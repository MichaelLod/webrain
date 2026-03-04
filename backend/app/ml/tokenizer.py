import numpy as np


class CharTokenizer:
    """Simple character-level tokenizer (256 possible byte values)."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def encode_batch(self, text: str, seq_len: int, batch_size: int) -> np.ndarray:
        """Encode text and return random batches of shape [batch_size, seq_len]."""
        encoded = self.encode(text)
        if len(encoded) < seq_len + 1:
            raise ValueError("Text too short for requested sequence length")

        batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        for i in range(batch_size):
            start = np.random.randint(0, len(encoded) - seq_len)
            batch[i] = encoded[start : start + seq_len]
        return batch

    @property
    def vocab_size(self) -> int:
        return 256
