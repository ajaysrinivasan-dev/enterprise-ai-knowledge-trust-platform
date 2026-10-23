"""SQLite-backed persistent vector store and safe document metadata registry."""

import json
import math
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from app.retrieval.models import DocumentChunk, RetrievalResult


class SQLiteVectorStore:
    """Persist sanitized chunks, embeddings, and safe document metadata in SQLite."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER,
                    embedding_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
                    ON document_chunks(document_id);
                CREATE TABLE IF NOT EXISTS indexed_documents (
                    document_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    pii_counts_json TEXT NOT NULL DEFAULT '{}',
                    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def upsert(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Persist chunks and vectors; only sanitized chunk text reaches this store."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal lengths")
        if not chunks:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO document_chunks
                    (chunk_id, document_id, source_name, text, page_number, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    source_name=excluded.source_name,
                    text=excluded.text,
                    page_number=excluded.page_number,
                    embedding_json=excluded.embedding_json
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source_name,
                        chunk.text,
                        chunk.page_number,
                        json.dumps(list(vector), separators=(",", ":")),
                    )
                    for chunk, vector in zip(chunks, embeddings, strict=True)
                ],
            )
            by_document: dict[str, tuple[str, int]] = {}
            for chunk in chunks:
                source, count = by_document.get(chunk.document_id, (chunk.source_name, 0))
                by_document[chunk.document_id] = (source, count + 1)
            for document_id, (source_name, chunk_count) in by_document.items():
                connection.execute(
                    """
                    INSERT INTO indexed_documents (document_id, source_name, chunk_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        source_name=excluded.source_name,
                        chunk_count=(
                            SELECT COUNT(*) FROM document_chunks
                            WHERE document_id=excluded.document_id
                        )
                    """,
                    (document_id, source_name, chunk_count),
                )

    def upsert_document_metadata(self, metadata: dict[str, object]) -> None:
        """Persist metadata that contains no document body or raw PII."""
        document_id = str(metadata["document_id"])
        source_name = str(metadata["source_name"])
        pii_counts = metadata.get("pii_counts", {})
        chunk_count = int(metadata["chunk_count"])
        with self._connect() as connection:
            chunk_row = connection.execute(
                "SELECT COUNT(*) AS n FROM document_chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            actual_count = int(chunk_row["n"])
            if actual_count != chunk_count:
                raise RuntimeError("document metadata does not match persisted indexed chunks")
            connection.execute(
                """
                INSERT INTO indexed_documents
                    (document_id, source_name, pii_counts_json, chunk_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_name=excluded.source_name,
                    pii_counts_json=excluded.pii_counts_json,
                    chunk_count=excluded.chunk_count
                """,
                (document_id, source_name, json.dumps(pii_counts, sort_keys=True), chunk_count),
            )

    def list_documents(self) -> list[dict[str, object]]:
        """Return safe document metadata only, ordered deterministically."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.source_name, d.pii_counts_json, d.chunk_count
                FROM indexed_documents AS d
                WHERE d.chunk_count > 0
                  AND d.chunk_count = (
                      SELECT COUNT(*) FROM document_chunks AS c
                      WHERE c.document_id = d.document_id
                  )
                ORDER BY d.created_at, d.document_id
                """
            ).fetchall()
        return [
            {
                "document_id": row["document_id"],
                "source_name": row["source_name"],
                "pii_counts": json.loads(row["pii_counts_json"]),
                "chunk_count": int(row["chunk_count"]),
            }
            for row in rows
        ]

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks and metadata for a document."""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
            )
            connection.execute(
                "DELETE FROM indexed_documents WHERE document_id = ?", (document_id,)
            )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Search persisted vectors using cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = list(query_embedding)
        query_norm = math.sqrt(sum(value * value for value in query)) or 1.0
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, source_name, text, page_number, embedding_json
                FROM document_chunks
                """
            ).fetchall()
        results: list[RetrievalResult] = []
        for row in rows:
            embedding = json.loads(row["embedding_json"])
            if len(embedding) != len(query):
                continue
            embedding_norm = math.sqrt(sum(value * value for value in embedding)) or 1.0
            denominator = query_norm * embedding_norm
            dot_product = sum(
                left * right for left, right in zip(query, embedding, strict=True)
            )
            score = dot_product / denominator
            if score >= min_score:
                results.append(
                    RetrievalResult(
                        DocumentChunk(
                            chunk_id=row["chunk_id"],
                            document_id=row["document_id"],
                            source_name=row["source_name"],
                            text=row["text"],
                            page_number=row["page_number"],
                        ),
                        score,
                    )
                )
        return sorted(results, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]
