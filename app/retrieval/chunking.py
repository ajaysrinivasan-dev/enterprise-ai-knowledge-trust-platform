"""Deterministic word-window chunking."""

import re

from app.retrieval.models import DocumentChunk


def chunk_text(
    text: str,
    *,
    document_id: str,
    source_name: str,
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[DocumentChunk]:
    """Split text into bounded word windows while preserving source metadata."""

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = re.findall(r"\S+", text)
    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size]
        if not window:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}:{index}",
                document_id=document_id,
                source_name=source_name,
                text=" ".join(window),
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks

