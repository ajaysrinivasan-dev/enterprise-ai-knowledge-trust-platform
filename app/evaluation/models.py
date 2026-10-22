"""Evaluation data structures."""

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class EvaluationCase:
    """A synthetic or user-provided evaluation example."""

    case_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    expected_answerable: bool
    reference_answer: str | None = None


@dataclass(frozen=True)
class EvaluationRun:
    """Persistable evaluation result metadata."""

    run_id: str
    dataset_version: str
    created_at: datetime
    metrics: dict[str, float | None]
    errors: tuple[str, ...] = ()

    @classmethod
    def now(
        cls,
        run_id: str,
        dataset_version: str,
        metrics: dict[str, float | None],
        errors: tuple[str, ...] = (),
    ) -> "EvaluationRun":
        return cls(run_id, dataset_version, datetime.now(UTC), metrics, errors)
