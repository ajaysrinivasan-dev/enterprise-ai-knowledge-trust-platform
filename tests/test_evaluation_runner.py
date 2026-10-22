"""Evaluation runner and optional adapter tests."""

from dataclasses import dataclass

from app.evaluation.models import EvaluationCase, EvaluationConfig
from app.evaluation.optional import check_deepeval, check_ragas
from app.evaluation.runner import EvaluationRunner
from app.evaluation.storage import EvaluationStore
from app.retrieval.models import DocumentChunk, RetrievalResult
from app.verification.models import ClaimStatus, VerificationResult


@dataclass
class Answer:
    answer: str
    citations: tuple = ()


def test_runner_records_success_and_partial_failure(tmp_path) -> None:
    class Generator:
        def __call__(self, question, results):
            if question == "bad":
                raise RuntimeError("provider unavailable")
            return Answer("The policy is seven years.", ())

    def retrieve(question, _top_k, _min_score):
        return [] if question == "bad" else [
            RetrievalResult(DocumentChunk("doc-1:0", "doc-1", "policy.txt", "seven years"), 1)
        ]

    def verify(_answer, _results):
        return VerificationResult((), ClaimStatus.UNCERTAIN, "test")

    runner = EvaluationRunner(
        retrieve,
        Generator(),
        verify,
        store=EvaluationStore(str(tmp_path / "runs.db")),
    )
    cases = [
        EvaluationCase("ok", "policy", ("doc-1:0",), True, "The policy is seven years."),
        EvaluationCase("bad", "bad", (), False),
    ]
    run = runner.run(cases, dataset_version="synthetic_v1", config=EvaluationConfig())

    assert run.metrics["completion_rate"] == 0.5
    assert run.metrics["failure_rate"] == 0.5
    assert [outcome.status for outcome in run.outcomes] == ["completed", "failed"]
    assert len(EvaluationStore(str(tmp_path / "runs.db")).list_runs()) == 1


def test_optional_evaluators_fail_clearly_when_missing() -> None:
    for checker in (check_ragas, check_deepeval):
        try:
            result = checker()
        except RuntimeError as exc:
            assert "not installed" in str(exc)
        else:
            assert result == "available-but-not-executed"
