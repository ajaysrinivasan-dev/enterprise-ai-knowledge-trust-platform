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
class ExampleOutcome:
    """Auditable result for one evaluation case."""

    case_id: str
    status: str
    retrieved_ids: tuple[str, ...] = ()
    answer: str | None = None
    citation_ids: tuple[str, ...] = ()
    claim_support_rate: float | None = None
    unsupported_claim_rate: float | None = None
    latency_ms: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Validated execution settings recorded with every run."""

    top_k: int = 5
    min_score: float = 0.25
    pipeline_id: str = "offline-deterministic"
    evaluator_id: str = "deterministic"


@dataclass(frozen=True)
class EvaluationRun:
    """Persistable evaluation result metadata."""

    run_id: str
    dataset_version: str
    created_at: datetime
    metrics: dict[str, float | None]
    dataset_fingerprint: str = ""
    dataset_status: str = "synthetic_smoke"
    errors: tuple[str, ...] = ()
    config: EvaluationConfig = EvaluationConfig()
    outcomes: tuple[ExampleOutcome, ...] = ()
    duration_ms: float = 0.0

    @classmethod
    def now(
        cls,
        run_id: str,
        dataset_version: str,
        metrics: dict[str, float | None],
        errors: tuple[str, ...] = (),
    ) -> "EvaluationRun":
        return cls(
            run_id,
            dataset_version,
            datetime.now(UTC),
            metrics,
            errors=errors,
        )

    @property
    def compatible_key(self) -> tuple[str, str, str, str]:
        """Runs are comparable only when their dataset and pipeline match."""

        return (
            self.dataset_version,
            self.dataset_fingerprint,
            self.dataset_status,
            self.config.pipeline_id,
        )
