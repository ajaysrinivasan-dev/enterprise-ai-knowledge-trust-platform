"""Grounded answer orchestration."""

from collections.abc import Callable

from app.generation.models import GroundedAnswer, citations_from_results
from app.retrieval.models import RetrievalResult


def generate_grounded_answer(
    question: str,
    results: list[RetrievalResult],
    generator: Callable[[str, list[str]], str],
    *,
    minimum_score: float = 0.25,
) -> GroundedAnswer:
    """Generate only from sufficient retrieved evidence and attach real citations."""

    usable = [result for result in results if result.score >= minimum_score]
    if not usable:
        return GroundedAnswer(
            answer="I could not find sufficient evidence in the indexed documents.",
            citations=(),
            abstained=True,
            reason="No retrieved evidence met the configured similarity threshold.",
        )
    answer = generator(question, [result.chunk.text for result in usable])
    if not answer.strip():
        raise RuntimeError("generation provider returned an empty answer")
    return GroundedAnswer(answer, citations_from_results(usable), False)
