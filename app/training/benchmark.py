"""Provider-neutral benchmark result schemas and baseline evaluation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.evaluation.metrics import routing_accuracy


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured benchmark output; no improvement is implied."""

    model_name: str
    accuracy: float
    latency_ms: float
    sample_count: int
    trained: bool
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    failure_rate: float
    dataset_fingerprint: str
    split: str
    dataset_status: str
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    adapter_id: str | None = None


def benchmark_router(
    model_name: str,
    predictor: Callable[[str], str],
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    dataset_fingerprint: str,
    split: str = "test",
    dataset_status: str,
    trained: bool = False,
) -> BenchmarkResult:
    """Measure an intent predictor on a held-out sequence."""

    if len(texts) != len(labels):
        raise ValueError("texts and labels must have equal lengths")
    if not dataset_fingerprint.strip():
        raise ValueError("dataset fingerprint is required")
    if split != "test":
        raise ValueError("benchmarks must use the held-out test split")
    if dataset_status not in {"synthetic_smoke", "approved_real"}:
        raise ValueError("dataset status must be synthetic_smoke or approved_real")
    if not texts:
        raise ValueError("benchmark split must not be empty")
    started = perf_counter()
    predictions = [predictor(text) for text in texts]
    elapsed = (perf_counter() - started) * 1000
    report = _per_class(predictions, labels)
    return BenchmarkResult(
        model_name=model_name,
        accuracy=routing_accuracy(predictions, labels),
        latency_ms=elapsed / len(texts) if texts else 0.0,
        sample_count=len(texts),
        trained=trained,
        per_class=report,
        confusion_matrix=_confusion(predictions, labels),
        failure_rate=0.0,
        dataset_fingerprint=dataset_fingerprint,
        split=split,
        dataset_status=dataset_status,
        macro_f1=sum(item["f1"] for item in report.values()) / len(report) if report else 0.0,
        weighted_f1=_weighted_f1(report, labels),
    )


def _per_class(predictions: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    """Calculate one-vs-rest precision, recall, and F1 for each observed label."""

    classes = sorted(set(predictions) | set(labels))
    report: dict[str, dict[str, float]] = {}
    for label in classes:
        pairs = list(zip(predictions, labels, strict=True))
        tp = sum(pred == label and actual == label for pred, actual in pairs)
        fp = sum(pred == label and actual != label for pred, actual in pairs)
        fn = sum(pred != label and actual == label for pred, actual in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        }
    return report


def _confusion(predictions: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """Return a JSON-safe confusion matrix keyed by expected then predicted label."""

    matrix: dict[str, dict[str, int]] = {}
    for predicted, actual in zip(predictions, labels, strict=True):
        matrix.setdefault(actual, {})
        matrix[actual][predicted] = matrix[actual].get(predicted, 0) + 1
    return matrix


def _weighted_f1(report: dict[str, dict[str, float]], labels: Sequence[str]) -> float:
    """Weight per-class F1 by observed support."""

    total = len(labels)
    if not total:
        return 0.0
    return sum(
        report[label]["f1"] * labels.count(label) for label in report
    ) / total


def benchmark_comparison(
    baseline: BenchmarkResult, fine_tuned: BenchmarkResult
) -> dict[str, Any]:
    """Compare measured results without implying improvement."""

    if baseline.sample_count != fine_tuned.sample_count:
        raise ValueError("benchmarks must use the same held-out sample count")
    if (
        baseline.dataset_fingerprint != fine_tuned.dataset_fingerprint
    ):
        raise ValueError("benchmarks must use the same held-out dataset")
    if baseline.split != fine_tuned.split:
        raise ValueError("benchmarks must use the same evaluation split")
    if baseline.dataset_status != fine_tuned.dataset_status:
        raise ValueError("benchmarks must use the same dataset status")
    return {
        "baseline": baseline,
        "fine_tuned": fine_tuned,
        "accuracy_delta": fine_tuned.accuracy - baseline.accuracy,
        "limitations": (
            "Small synthetic datasets do not establish production generalization."
            if baseline.sample_count < 100
            else None
        ),
    }
