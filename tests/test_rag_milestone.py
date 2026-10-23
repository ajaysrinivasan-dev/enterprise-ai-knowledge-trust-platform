"""RAG provider, indexing, retrieval, and API integration tests."""

import pytest
from app.api.rag_runtime import _documents
from app.main import app
from app.retrieval.chunking import chunk_text
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.memory_store import InMemoryVectorStore
from app.retrieval.models import DocumentChunk, RetrievalResult
from app.retrieval.service import RetrievalService
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolate_process_local_rag_runtime(monkeypatch):
    """Keep API tests independent of any local persistent database."""

    import app.api.rag_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "_service",
        RetrievalService(HashEmbeddingProvider(), InMemoryVectorStore()),
    )
    _documents.clear()


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


def test_question_api_uses_one_retrieval_set_for_answer_and_evidence(monkeypatch) -> None:
    evidence = [
        RetrievalResult(
            DocumentChunk("doc-1:0", "doc-1", "policy.txt", "Retention is seven years."),
            1.0,
        )
    ]
    calls = []

    def fake_retrieve(question, top_k, min_score):
        calls.append((question, top_k, min_score))
        return evidence

    def fake_answer(question, top_k, min_score, *, evidence):
        assert evidence is evidence_set
        return type(
            "Answer",
            (),
            {
                "answer": "Retention is seven years.",
                "abstained": False,
                "citations": (),
                "reason": None,
            },
        )()

    evidence_set = evidence
    import app.api.rag_runtime as runtime

    monkeypatch.setattr(runtime, "retrieve", fake_retrieve)
    monkeypatch.setattr(runtime, "answer", fake_answer)
    response = TestClient(app).post(
        "/api/v1/questions",
        json={"question": "What is the retention policy?", "min_score": 0},
    )

    assert response.status_code == 200
    assert len(calls) == 1


def test_retrieval_service_uses_explicit_hash_and_memory_settings() -> None:
    from app.api.rag_runtime import build_retrieval_service
    from app.core.config import Settings

    service = build_retrieval_service(
        Settings(embedding_backend="hash", vector_backend="memory")
    )

    assert isinstance(service.embeddings, HashEmbeddingProvider)
    assert isinstance(service.store, InMemoryVectorStore)


def test_retrieval_service_wires_configured_optional_providers(monkeypatch) -> None:
    import app.api.rag_runtime as runtime
    from app.core.config import Settings

    calls = {}

    class FakeEmbeddings:
        def __init__(self, model_name):
            calls["model_name"] = model_name

    class FakeStore:
        def __init__(self, path):
            calls["chroma_path"] = path

    monkeypatch.setattr(runtime, "SentenceTransformerProvider", FakeEmbeddings)
    monkeypatch.setattr(runtime, "ChromaVectorStore", FakeStore)
    settings = Settings(
        embedding_backend="sentence-transformers",
        embedding_model="local-approved-model",
        vector_backend="chroma",
        chroma_path="temporary-test-chroma",
    )

    service = runtime.build_retrieval_service(settings)

    assert isinstance(service.embeddings, FakeEmbeddings)
    assert isinstance(service.store, FakeStore)
    assert calls == {
        "model_name": "local-approved-model",
        "chroma_path": "temporary-test-chroma",
    }



def test_sqlite_retrieval_and_document_listing_survive_store_recreation(
    tmp_path, monkeypatch
) -> None:
    import app.api.rag_runtime as runtime
    from app.api.rag_runtime import build_retrieval_service, index_document, list_documents
    from app.core.config import Settings
    from app.ingestion.models import ParsedDocument

    database = tmp_path / "eaktp.sqlite3"
    settings = Settings(vector_backend="sqlite", sqlite_vector_path=str(database))
    monkeypatch.setattr(runtime, "_service", build_retrieval_service(settings))

    indexed = index_document(
        ParsedDocument(
            "retention.txt",
            ".txt",
            "Contact owner@example.com. The retention policy is seven years.",
        )
    )
    assert indexed["pii_counts"] == {"EMAIL": 1}

    # A fresh store instance represents a process restart against the same database.
    restarted_service = build_retrieval_service(settings)
    monkeypatch.setattr(runtime, "_service", restarted_service)

    documents = list_documents()
    assert documents == [indexed]
    assert "owner@example.com" not in str(documents)
    results = runtime.retrieve("retention policy", top_k=5, min_score=0.0)
    assert results
    assert results[0].chunk.document_id == indexed["document_id"]
    assert all("owner@example.com" not in result.chunk.text for result in results)

    restarted_service.store.delete_document(str(indexed["document_id"]))
    assert list_documents() == []
