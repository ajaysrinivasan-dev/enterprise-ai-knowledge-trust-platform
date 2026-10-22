"""Deterministic retrieval tests."""

import pytest
from app.retrieval.chunking import chunk_text
from app.retrieval.in_memory import InMemoryRetriever


def test_chunking_preserves_source_identity_and_overlap() -> None:
    chunks = chunk_text(
        "one two three four five six",
        document_id="doc-1",
        source_name="policy.txt",
        chunk_size=4,
        overlap=1,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["doc-1:0", "doc-1:1"]
    assert chunks[0].source_name == "policy.txt"
    assert chunks[1].text.startswith("four")


def test_retrieval_returns_real_chunks_in_deterministic_order() -> None:
    chunks = chunk_text(
        "retention policy is seven years. access requires approval.",
        document_id="doc-1",
        source_name="policy.txt",
        chunk_size=20,
        overlap=0,
    )
    retriever = InMemoryRetriever(chunks)

    results = retriever.search("What is the retention policy?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "doc-1:0"
    assert results[0].score > 0


def test_chunking_and_search_validate_limits() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", document_id="d", source_name="x", chunk_size=2, overlap=2)
    with pytest.raises(ValueError):
        InMemoryRetriever().search("query", top_k=0)
