"""Process-local RAG runtime for local development and deterministic API tests."""

from uuid import uuid4

from app.core.config import get_settings
from app.generation.models import GroundedAnswer
from app.generation.providers import OllamaProvider
from app.generation.service import generate_grounded_answer
from app.ingestion.models import ParsedDocument
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.memory_store import InMemoryVectorStore
from app.retrieval.service import RetrievalService

_service = RetrievalService(HashEmbeddingProvider(), InMemoryVectorStore())
_documents: dict[str, dict[str, object]] = {}


def index_document(document: ParsedDocument) -> dict[str, object]:
    """Index one sanitized document and retain safe metadata."""

    document_id = str(uuid4())
    indexed = _service.index(document, document_id)
    _documents[document_id] = {
        "document_id": indexed.document_id,
        "source_name": indexed.source_name,
        "pii_counts": indexed.pii_counts,
        "chunk_count": indexed.chunk_count,
    }
    return _documents[document_id]


def list_documents() -> list[dict[str, object]]:
    """List safe document metadata."""

    return list(_documents.values())


def retrieve(question: str, top_k: int, min_score: float):
    """Retrieve evidence from the process-local index."""

    return _service.retrieve(question, top_k=top_k, min_score=min_score)


def answer(question: str, top_k: int, min_score: float) -> GroundedAnswer:
    """Return retrieval-only grounded evidence until an LLM is configured."""

    results = retrieve(question, top_k, min_score)
    settings = get_settings()
    if settings.ollama_model:
        generator = OllamaProvider(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.ollama_timeout_seconds,
        ).generate
    else:
        def generator(_question: str, evidence: list[str]) -> str:
            return "Relevant evidence:\n" + "\n".join(f"- {item}" for item in evidence)

    return generate_grounded_answer(question, results, generator, minimum_score=min_score)
