import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TOKENIZER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.model")


class CharTokenizer:
    """Legacy character-level tokenizer (256 possible byte values). Kept for checkpoint migration."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def encode_batch(self, text: str, seq_len: int, batch_size: int) -> np.ndarray:
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


class BPETokenizer:
    """BPE tokenizer wrapping sentencepiece with 8192 vocab."""

    def __init__(self, model_path: str | None = None):
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor()
        self._model_path = model_path or TOKENIZER_MODEL_PATH
        self._loaded = False
        if os.path.exists(self._model_path):
            self._sp.Load(self._model_path)
            self._loaded = True
            log.info("Loaded BPE tokenizer from %s (vocab_size=%d)", self._model_path, self._sp.GetPieceSize())

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def train(self, corpus: str | list[str], vocab_size: int = 8192, model_prefix: str | None = None):
        """Train a new sentencepiece BPE model from a corpus."""
        import sentencepiece as spm
        import tempfile

        if model_prefix is None:
            model_prefix = self._model_path.replace(".model", "")

        if isinstance(corpus, list):
            corpus = "\n".join(corpus)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(corpus)
            tmp_path = f.name

        try:
            spm.SentencePieceTrainer.Train(
                input=tmp_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                num_threads=os.cpu_count() or 4,
                train_extremely_large_corpus=False,
            )
            self._sp.Load(f"{model_prefix}.model")
            self._loaded = True
            log.info("Trained BPE tokenizer: vocab_size=%d", self._sp.GetPieceSize())
        finally:
            os.unlink(tmp_path)

    def encode(self, text: str) -> list[int]:
        if not self._loaded:
            raise RuntimeError("Tokenizer not loaded. Call train() or load() first.")
        return self._sp.Encode(text)

    def decode(self, tokens: list[int]) -> str:
        if not self._loaded:
            raise RuntimeError("Tokenizer not loaded. Call train() or load() first.")
        return self._sp.Decode(tokens)

    def encode_batch(self, text: str, seq_len: int, batch_size: int) -> np.ndarray:
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
        if self._loaded:
            return self._sp.GetPieceSize()
        return 8192

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    def save(self, path: str):
        """Copy the trained model to a new path."""
        if not self._loaded:
            raise RuntimeError("No model to save.")
        src = self._model_path
        if not os.path.exists(src):
            raise FileNotFoundError(f"Model file not found: {src}")
        import shutil
        shutil.copy2(src, path)

    def load(self, path: str):
        """Load a sentencepiece model from path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self._sp.Load(path)
        self._model_path = path
        self._loaded = True


def get_tokenizer() -> BPETokenizer | CharTokenizer:
    """Return BPE tokenizer if model exists, else fall back to CharTokenizer."""
    if os.path.exists(TOKENIZER_MODEL_PATH):
        return BPETokenizer(TOKENIZER_MODEL_PATH)
    log.warning("BPE tokenizer model not found at %s, falling back to CharTokenizer", TOKENIZER_MODEL_PATH)
    return CharTokenizer()
