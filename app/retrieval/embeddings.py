"""Embedding provider abstractions with deterministic and Sentence Transformer backends."""

from hashlib import sha256
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Interface implemented by local, hosted, and test embedding providers."""

    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


class HashEmbeddingProvider:
    """Small deterministic fallback used by tests and offline development."""

    model_name = "hash-fallback-v1"

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = sha256(text.casefold().encode("utf-8")).digest()
            values = [digest[index % len(digest)] / 255 for index in range(self.dimensions)]
            norm = sum(value * value for value in values) ** 0.5 or 1
            vectors.append([value / norm for value in values])
        return vectors


class SentenceTransformerProvider:
    """Optional embedding provider backed by sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformer support requires the optional 'rag' dependencies"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts, normalize_embeddings=True).tolist()
