"""Generation response models."""

from dataclasses import dataclass

from app.retrieval.models import RetrievalResult


@dataclass(frozen=True)
class Citation:
    """A citation that must refer to a retrieved chunk."""

    chunk_id: str
    source_name: str
    excerpt: str


@dataclass(frozen=True)
class GroundedAnswer:
    """An answer and its evidence-backed citations."""

    answer: str
    citations: tuple[Citation, ...]
    abstained: bool
    reason: str | None = None


def citations_from_results(results: list[RetrievalResult]) -> tuple[Citation, ...]:
    """Build citations only from actual retrieval results."""

    return tuple(
        Citation(
            chunk_id=result.chunk.chunk_id,
            source_name=result.chunk.source_name,
            excerpt=result.chunk.text,
        )
        for result in results
    )


def retrieval_only_answer(
    question: str, results: list[RetrievalResult], *, minimum_score: float = 0.25
) -> GroundedAnswer:
    """Return evidence or an explicit abstention when retrieval is insufficient."""

    usable = [result for result in results if result.score >= minimum_score]
    if not usable:
        return GroundedAnswer(
            answer="I could not find sufficient evidence in the indexed documents.",
            citations=(),
            abstained=True,
            reason=f"No retrieved evidence met the score threshold for: {question}",
        )
    answer = "Relevant evidence:\n" + "\n".join(
        f"- {result.chunk.text}" for result in usable
    )
    return GroundedAnswer(
        answer=answer,
        citations=citations_from_results(usable),
        abstained=False,
    )


def validate_citations(answer: GroundedAnswer, results: list[RetrievalResult]) -> None:
    """Reject citations that do not correspond to the current retrieval set."""

    available = {result.chunk.chunk_id: result.chunk for result in results}
    for citation in answer.citations:
        chunk = available.get(citation.chunk_id)
        if chunk is None or chunk.source_name != citation.source_name:
            raise ValueError(f"citation does not reference retrieved chunk: {citation.chunk_id}")

