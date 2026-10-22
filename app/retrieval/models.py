"""Retrieval domain models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable piece of sanitized document text."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """A chunk and its deterministic lexical relevance score."""

    chunk: DocumentChunk
    score: float

