"""Adapter benchmark safety tests."""

import pytest
from app.main import app
from app.training.benchmark import BenchmarkResult, benchmark_comparison
from fastapi.testclient import TestClient


def test_benchmark_rejects_unknown_adapter_without_model_loading() -> None:
    response = TestClient(app).post(
        "/api/v1/training/benchmarks",
        json={"adapter_id": "not-a-real-run"},
    )

    assert response.status_code == 422
    assert "approved" in response.json()["detail"]


def test_benchmark_comparison_rejects_different_held_out_datasets() -> None:
    baseline = BenchmarkResult(
        model_name="baseline",
        accuracy=1.0,
        latency_ms=1.0,
        sample_count=1,
        trained=False,
        per_class={},
        confusion_matrix={},
        failure_rate=0.0,
        dataset_fingerprint="one",
    )
    candidate = BenchmarkResult(
        model_name="candidate",
        accuracy=1.0,
        latency_ms=1.0,
        sample_count=1,
        trained=True,
        per_class={},
        confusion_matrix={},
        failure_rate=0.0,
        dataset_fingerprint="two",
    )

    with pytest.raises(ValueError, match="same held-out dataset"):
        benchmark_comparison(baseline, candidate)
