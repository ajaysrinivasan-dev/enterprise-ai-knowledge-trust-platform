"""Privacy-aware indexing and retrieval orchestration."""

from app.ingestion.models import ParsedDocument
from app.privacy.service import redact_pii
from app.retrieval.chunking import chunk_text
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.models import IndexedDocument, RetrievalResult


class RetrievalService:
    """Redact, chunk, embed, and retrieve document evidence."""

    def __init__(self, embeddings: EmbeddingProvider, store: object) -> None:
        self.embeddings = embeddings
        self.store = store

    def index(self, document: ParsedDocument, document_id: str) -> IndexedDocument:
        """Index sanitized content and return only safe metadata."""

        redaction = redact_pii(document.text)
        chunks = chunk_text(
            redaction.text,
            document_id=document_id,
            source_name=document.filename,
        )
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        self.store.upsert(chunks, vectors)
        return IndexedDocument(
            document_id=document_id,
            source_name=document.filename,
            text=redaction.text,
            pii_counts=redaction.detection.counts,
            chunk_count=len(chunks),
        )

    def retrieve(
        self, question: str, *, top_k: int = 5, min_score: float = 0.0
    ) -> list[RetrievalResult]:
        """Embed a query and retrieve evidence from the configured store."""

        vectors = self.embeddings.embed([question])
        return self.store.search(vectors[0], top_k=top_k, min_score=min_score)
