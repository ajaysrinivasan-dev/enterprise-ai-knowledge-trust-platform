"""RAG provider, indexing, retrieval, and API integration tests."""

from app.api.rag_runtime import _documents
from app.main import app
from app.retrieval.chunking import chunk_text
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.memory_store import InMemoryVectorStore
from app.retrieval.service import RetrievalService
from fastapi.testclient import TestClient


def test_hash_embeddings_are_deterministic_and_indexing_redacts_pii() -> None:
    provider = HashEmbeddingProvider()
    assert provider.embed(["same"]) == provider.embed(["same"])

    service = RetrievalService(provider, InMemoryVectorStore())
    from app.ingestion.models import ParsedDocument

    indexed = service.index(
        ParsedDocument(
            "policy.txt",
            ".txt",
            "Contact owner@example.com. Retention is seven years.",
        ),
        "doc-1",
    )
    results = service.retrieve("retention", top_k=2, min_score=0.0)

    assert indexed.pii_counts == {"EMAIL": 1}
    assert "owner@example.com" not in indexed.text
    assert results[0].chunk.document_id == "doc-1"


def test_in_memory_vector_store_supports_upsert_and_deletion() -> None:
    store = InMemoryVectorStore()
    chunks = chunk_text("alpha beta", document_id="doc", source_name="a.txt")
    store.upsert(chunks, [[1.0] * 64])
    assert store.search([1.0] * 64, top_k=1, min_score=0.5)
    store.delete_document("doc")
    assert store.search([1.0] * 64, top_k=1) == []


def test_document_retrieval_and_question_api_use_real_citations() -> None:
    _documents.clear()
    client = TestClient(app)
    upload = client.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", b"Retention policy is seven years.", "text/plain")},
    )
    assert upload.status_code == 200
    question = client.post(
        "/api/v1/questions",
        json={"question": "What is the retention policy?", "min_score": 0},
    )
    assert question.status_code == 200
    payload = question.json()
    assert payload["abstained"] is False
    assert payload["citations"][0]["chunk_id"] == payload["evidence"][0]["chunk_id"]


def test_question_api_abstains_when_no_evidence_exists() -> None:
    _documents.clear()
    response = TestClient(app).post(
        "/api/v1/questions",
        json={"question": "Unknown topic", "min_score": 0.9},
    )
    assert response.status_code == 200
    assert response.json()["abstained"] is True
