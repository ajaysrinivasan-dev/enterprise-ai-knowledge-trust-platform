"""Deterministic vector store used when ChromaDB is not configured."""

import math
from collections.abc import Sequence

from app.retrieval.models import DocumentChunk, RetrievalResult


class InMemoryVectorStore:
    """Persist vectors for the process lifetime and support cosine retrieval."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[DocumentChunk, list[float]]] = {}

    def upsert(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal lengths")
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._records[chunk.chunk_id] = (chunk, list(embedding))

    def delete_document(self, document_id: str) -> None:
        self._records = {
            key: value
            for key, value in self._records.items()
            if value[0].document_id != document_id
        }

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        tenant_id: str | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_norm = math.sqrt(sum(value * value for value in query_embedding)) or 1
        results: list[RetrievalResult] = []
        for chunk, embedding in self._records.values():
            if tenant_id is not None and tenant_id != "*" and chunk.tenant_id != tenant_id:
                continue
            denominator = query_norm * (math.sqrt(sum(value * value for value in embedding)) or 1)
            score = (
                sum(left * right for left, right in zip(query_embedding, embedding, strict=True))
                / denominator
            )
            if score >= min_score:
                results.append(RetrievalResult(chunk, score))
        return sorted(results, key=lambda result: (-result.score, result.chunk.chunk_id))[:top_k]
