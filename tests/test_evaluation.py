"""Deterministic evaluation tests."""

from app.evaluation.dataset import load_jsonl
from app.evaluation.metrics import (
    binary_f1,
    citation_relevance,
    citation_validity,
    precision_at_k,
    recall_at_k,
    routing_accuracy,
)
from app.evaluation.models import EvaluationRun
from app.evaluation.storage import EvaluationStore


def test_metrics_are_deterministic_and_zero_safe() -> None:
    assert recall_at_k(["a", "b"], ["b"], 2) == 1.0
    assert precision_at_k(["a", "b"], ["b"], 1) == 0.0
    assert binary_f1([True, False], [True, True]) == 2 / 3
    assert routing_accuracy(["question"], ["question"]) == 1.0
    assert citation_validity(["a", "missing"], ["a"]) == 0.5
    assert citation_relevance(["a"], ["b"]) == 0.0


def test_dataset_and_run_storage(tmp_path) -> None:
    cases = load_jsonl("data/evaluation/synthetic_v1.jsonl")
    assert len(cases) == 2
    store = EvaluationStore(str(tmp_path / "evaluation.db"))
    run = EvaluationRun.now("run-1", "synthetic_v1", {"recall_at_1": 1.0})
    store.save(run)
    assert store.list_runs()[0].metrics["recall_at_1"] == 1.0
