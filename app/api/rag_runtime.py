"""RAG runtime with durable SQLite retrieval by default and an explicit memory fallback."""

from uuid import uuid4

from app.core.config import Settings, get_settings
from app.generation.models import GroundedAnswer
from app.generation.providers import OllamaProvider
from app.generation.service import generate_grounded_answer
from app.ingestion.models import ParsedDocument
from app.retrieval.embeddings import HashEmbeddingProvider, SentenceTransformerProvider
from app.retrieval.memory_store import InMemoryVectorStore
from app.retrieval.service import RetrievalService
from app.retrieval.sqlite_store import SQLiteVectorStore
from app.retrieval.vector_store import ChromaVectorStore


def build_retrieval_service(settings: Settings | None = None) -> RetrievalService:
    """Build the explicitly configured retrieval providers without silent fallbacks."""

    settings = settings or get_settings()
    # Validate the store first so a missing optional vector dependency fails before
    # a sentence-transformer model could be initialized or downloaded.
    if settings.vector_backend == "memory":
        store = InMemoryVectorStore()
    elif settings.vector_backend == "sqlite":
        store = SQLiteVectorStore(settings.sqlite_vector_path)
    elif settings.vector_backend == "chroma":
        store = ChromaVectorStore(settings.chroma_path)
    else:  # Defensive guard for custom settings objects.
        raise ValueError(f"unsupported vector backend: {settings.vector_backend}")

    if settings.embedding_backend == "hash":
        embeddings = HashEmbeddingProvider()
    elif settings.embedding_backend == "sentence-transformers":
        embeddings = SentenceTransformerProvider(settings.embedding_model)
    else:  # Defensive guard for custom settings objects.
        raise ValueError(f"unsupported embedding backend: {settings.embedding_backend}")
    return RetrievalService(embeddings, store)


_service = build_retrieval_service()
_documents: dict[str, dict[str, object]] = {}


def index_document(document: ParsedDocument) -> dict[str, object]:
    """Index one sanitized document and retain safe metadata."""

    document_id = str(uuid4())
    indexed = _service.index(document, document_id)
    metadata: dict[str, object] = {
        "document_id": indexed.document_id,
        "source_name": indexed.source_name,
        "pii_counts": indexed.pii_counts,
        "chunk_count": indexed.chunk_count,
    }
    persist_metadata = getattr(_service.store, "upsert_document_metadata", None)
    if callable(persist_metadata):
        persist_metadata(metadata)
    _documents[document_id] = metadata
    return metadata


def list_documents() -> list[dict[str, object]]:
    """List safe metadata from durable storage when the configured backend supports it."""

    persistent_listing = getattr(_service.store, "list_documents", None)
    if callable(persistent_listing):
        return persistent_listing()
    return list(_documents.values())


def retrieve(question: str, top_k: int, min_score: float):
    """Retrieve evidence from the process-local index."""

    return _service.retrieve(question, top_k=top_k, min_score=min_score)


def answer(
    question: str,
    top_k: int,
    min_score: float,
    *,
    evidence=None,
) -> GroundedAnswer:
    """Return retrieval-only grounded evidence until an LLM is configured."""

    results = retrieve(question, top_k, min_score) if evidence is None else evidence
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
