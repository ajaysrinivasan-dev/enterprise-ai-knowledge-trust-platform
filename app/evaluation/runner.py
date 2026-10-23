"""Reproducible offline evaluation runner."""

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from app.evaluation.dataset import dataset_fingerprint
from app.evaluation.metrics import (
    answer_correctness,
    citation_relevance,
    citation_validity,
    mean_or_none,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.models import EvaluationCase, EvaluationConfig, EvaluationRun, ExampleOutcome


class EvaluationRunner:
    """Execute a deterministic pipeline and persist actual outcomes."""

    def __init__(
        self,
        retrieve: Callable[[str, int, float], list],
        answer: Callable[[str, list], object],
        verify: Callable[[str, list], object],
        *,
        store,
    ) -> None:
        self.retrieve = retrieve
        self.answer = answer
        self.verify = verify
        self.store = store

    def run(
        self,
        cases: list[EvaluationCase],
        *,
        dataset_version: str,
        config: EvaluationConfig | None = None,
        run_id: str | None = None,
    ) -> EvaluationRun:
        """Run every case, recording failures rather than converting them to successes."""

        if not cases:
            raise ValueError("evaluation dataset is empty")
        fingerprint = dataset_fingerprint(cases)
        execution = config or EvaluationConfig()
        started = perf_counter()
        outcomes: list[ExampleOutcome] = []
        errors: list[str] = []
        for case in cases:
            case_started = perf_counter()
            try:
                results = self.retrieve(case.question, execution.top_k, execution.min_score)
                retrieved_ids = tuple(result.chunk.chunk_id for result in results)
                generated = self.answer(case.question, results)
                verified = self.verify(generated.answer, results)
                claims = verified.claims
                supported = sum(claim.status.value == "supported" for claim in claims)
                support_rate = supported / len(claims) if claims else None
                citation_ids = tuple(citation.chunk_id for citation in generated.citations)
                outcomes.append(
                    ExampleOutcome(
                        case_id=case.case_id,
                        status="completed",
                        retrieved_ids=retrieved_ids,
                        answer=generated.answer,
                        citation_ids=citation_ids,
                        claim_support_rate=support_rate,
                        unsupported_claim_rate=(
                            1 - support_rate if support_rate is not None else None
                        ),
                        latency_ms=(perf_counter() - case_started) * 1000,
                    )
                )
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                message = f"{case.case_id}: {exc}"
                errors.append(message)
                outcomes.append(
                    ExampleOutcome(
                        case_id=case.case_id,
                        status="failed",
                        latency_ms=(perf_counter() - case_started) * 1000,
                        error=str(exc),
                    )
                )
        completed = [outcome for outcome in outcomes if outcome.status == "completed"]
        recalls = [
            recall_at_k(outcome.retrieved_ids, case.relevant_chunk_ids, execution.top_k)
            for outcome, case in zip(outcomes, cases, strict=True)
            if outcome.status == "completed"
        ]
        precisions = [
            precision_at_k(outcome.retrieved_ids, case.relevant_chunk_ids, execution.top_k)
            for outcome, case in zip(outcomes, cases, strict=True)
            if outcome.status == "completed"
        ]
        correctness = [
            answer_correctness(outcome.answer, case.reference_answer)
            for outcome, case in zip(outcomes, cases, strict=True)
            if outcome.status == "completed"
        ]
        citation_validities = [
            citation_validity(outcome.citation_ids, outcome.retrieved_ids)
            for outcome in completed
        ]
        citation_relevances = [
            citation_relevance(outcome.citation_ids, case.relevant_chunk_ids)
            for outcome, case in zip(outcomes, cases, strict=True)
            if outcome.status == "completed"
        ]
        metrics = {
            "recall_at_k": mean_or_none(recalls),
            "precision_at_k": mean_or_none(precisions),
            "answer_correctness": mean_or_none(correctness),
            "citation_validity": mean_or_none(citation_validities),
            "citation_relevance": mean_or_none(citation_relevances),
            "claim_support_rate": mean_or_none(
                outcome.claim_support_rate for outcome in completed
            ),
            "unsupported_claim_rate": mean_or_none(
                outcome.unsupported_claim_rate for outcome in completed
            ),
            "completion_rate": len(completed) / len(outcomes),
            "failure_rate": len(errors) / len(outcomes),
            "mean_latency_ms": mean_or_none(outcome.latency_ms for outcome in completed),
        }
        run = EvaluationRun(
            run_id=run_id or str(uuid4()),
            dataset_version=dataset_version,
            dataset_fingerprint=fingerprint,
            dataset_status="synthetic_smoke",
            created_at=datetime.now(UTC),
            metrics=metrics,
            errors=tuple(errors),
            config=execution,
            outcomes=tuple(outcomes),
            duration_ms=(perf_counter() - started) * 1000,
        )
        self.store.save(run)
        return run
