"""Provider-neutral benchmark result schemas and baseline evaluation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from app.evaluation.metrics import routing_accuracy


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured benchmark output; no improvement is implied."""

    model_name: str
    accuracy: float
    latency_ms: float
    sample_count: int
    trained: bool


def benchmark_router(
    model_name: str,
    predictor: Callable[[str], str],
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    trained: bool = False,
) -> BenchmarkResult:
    """Measure an intent predictor on a held-out sequence."""

    if len(texts) != len(labels):
        raise ValueError("texts and labels must have equal lengths")
    started = perf_counter()
    predictions = [predictor(text) for text in texts]
    elapsed = (perf_counter() - started) * 1000
    return BenchmarkResult(
        model_name=model_name,
        accuracy=routing_accuracy(predictions, labels),
        latency_ms=elapsed / len(texts) if texts else 0.0,
        sample_count=len(texts),
        trained=trained,
    )
