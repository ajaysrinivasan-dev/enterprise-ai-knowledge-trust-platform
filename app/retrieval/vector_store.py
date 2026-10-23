"""Persistent vector storage with an optional ChromaDB backend."""

from collections.abc import Sequence
from typing import Any

from app.retrieval.models import DocumentChunk, RetrievalResult


class ChromaVectorStore:
    """Store chunks in a persistent Chroma collection."""

    def __init__(self, path: str, collection_name: str = "eaktp_documents") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB support requires the optional 'rag' dependencies") from exc
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Upsert chunks and safe source metadata without storing raw PII."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal lengths")
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source_name": chunk.source_name,
                    "page_number": chunk.page_number or 0,
                    "tenant_id": chunk.tenant_id or "__legacy_unowned__",
                }
                for chunk in chunks
            ],
        )

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to one document."""

        self._collection.delete(where={"document_id": document_id})

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        tenant_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Query Chroma and convert distances to bounded similarity scores."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_args: dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if tenant_id is not None and tenant_id != "*":
            query_args["where"] = {"tenant_id": tenant_id}
        response: dict[str, Any] = self._collection.query(**query_args)
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        results: list[RetrievalResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            score = 1 / (1 + float(distance))
            if score < min_score:
                continue
            results.append(
                RetrievalResult(
                    chunk=DocumentChunk(
                        chunk_id=str(metadata.get("chunk_id", "")),
                        document_id=str(metadata["document_id"]),
                        source_name=str(metadata["source_name"]),
                        text=str(text),
                        page_number=int(metadata.get("page_number", 0)) or None,
                        tenant_id=metadata.get("tenant_id"),
                    ),
                    score=score,
                )
            )
        return results
