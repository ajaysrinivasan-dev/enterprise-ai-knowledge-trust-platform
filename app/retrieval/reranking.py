"""Optional cross-encoder reranking."""

from app.retrieval.models import RetrievalResult


class CrossEncoderReranker:
    """Rerank retrieval results when sentence-transformers is installed."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("reranking requires the optional 'rag' dependencies") from exc
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Return results sorted by the cross-encoder score."""

        if not results:
            return []
        scores = self._model.predict([(query, result.chunk.text) for result in results])
        reranked = [
            RetrievalResult(result.chunk, float(score))
            for result, score in zip(results, scores, strict=True)
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)[:top_k]
