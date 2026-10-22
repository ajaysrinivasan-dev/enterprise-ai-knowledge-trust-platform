"""Approved evaluation runtime wiring existing RAG and verification services."""

from pathlib import Path

from app.api.rag_runtime import retrieve
from app.evaluation.dataset import load_jsonl
from app.evaluation.models import EvaluationConfig, EvaluationRun
from app.evaluation.runner import EvaluationRunner
from app.evaluation.storage import EvaluationStore
from app.retrieval.models import RetrievalResult
from app.verification.service import verify_answer

_DATASETS = {
    "synthetic_v1": Path("data/evaluation/synthetic_v1.jsonl"),
}


def approved_dataset_path(dataset_id: str) -> Path:
    """Resolve only named datasets from the project allowlist."""

    path = _DATASETS.get(dataset_id)
    if path is None:
        raise ValueError("dataset is not approved")
    return path


def evaluation_store(database_path: str) -> EvaluationStore:
    """Create the runtime store and its ignored runtime directory."""

    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    return EvaluationStore(database_path)


def run_evaluation(
    dataset_id: str,
    database_path: str,
    *,
    config: EvaluationConfig | None = None,
) -> EvaluationRun:
    """Execute the approved offline pipeline and persist the actual run."""

    cases = load_jsonl(approved_dataset_path(dataset_id))

    def answer_case(question: str, evidence: list[RetrievalResult]):
        from app.generation.service import generate_grounded_answer

        def retrieval_only(_question: str, texts: list[str]) -> str:
            return "Relevant evidence:\n" + "\n".join(f"- {text}" for text in texts)

        return generate_grounded_answer(question, evidence, retrieval_only, minimum_score=0)

    runner = EvaluationRunner(
        retrieve,
        answer_case,
        verify_answer,
        store=evaluation_store(database_path),
    )
    return runner.run(cases, dataset_version=dataset_id, config=config)


def get_run(run_id: str, database_path: str) -> EvaluationRun:
    """Retrieve a persisted run by ID."""

    matches = [run for run in evaluation_store(database_path).list_runs() if run.run_id == run_id]
    if not matches:
        raise KeyError("evaluation run not found")
    return matches[0]
