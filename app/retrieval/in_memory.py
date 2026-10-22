"""Dependency-free retrieval backend used for tests and local fallback."""

import re

from app.retrieval.models import DocumentChunk, RetrievalResult


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.casefold()))


class InMemoryRetriever:
    """Retrieve chunks using normalized query-term coverage."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self._chunks = list(chunks or [])

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks, replacing existing chunks with the same identifier."""

        by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        by_id.update({chunk.chunk_id: chunk for chunk in chunks})
        self._chunks = list(by_id.values())

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Return highest-overlap chunks, excluding chunks with no query terms."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = _terms(query)
        if not query_terms:
            return []
        scored = []
        for chunk in self._chunks:
            score = len(query_terms & _terms(chunk.text)) / len(query_terms)
            if score > 0:
                scored.append(RetrievalResult(chunk=chunk, score=score))
        return sorted(scored, key=lambda result: (-result.score, result.chunk.chunk_id))[:top_k]

