"""Adapter benchmark safety tests."""

import pytest
from app.api.evaluation_runtime import approved_dataset_path
from app.core.config import Settings
from app.main import create_app
from app.training.benchmark import BenchmarkResult, benchmark_comparison, benchmark_router
from app.training.dataset import (
    IntentSplits,
    dataset_fingerprint,
    load_intent_dataset,
    split_intent_dataset,
    validate_training_ready,
)
from fastapi.testclient import TestClient
from pydantic import SecretStr


def _admin_client(monkeypatch) -> TestClient:
    import app.main as main

    settings = Settings(api_key=SecretStr("benchmark-admin-" + "a" * 32), api_principals_json=None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return TestClient(
        create_app(), headers={"Authorization": "Bearer benchmark-admin-" + "a" * 32}
    )


def test_benchmark_rejects_unknown_adapter_without_model_loading(monkeypatch) -> None:
    response = _admin_client(monkeypatch).post(
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
        split="test",
        dataset_status="synthetic_smoke",
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
        split="test",
        dataset_status="synthetic_smoke",
    )

    with pytest.raises(ValueError, match="same held-out dataset"):
        benchmark_comparison(baseline, candidate)


def test_benchmark_requires_fingerprint_and_test_split() -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        from app.training.benchmark import benchmark_router

        benchmark_router(
            "model",
            lambda _text: "question",
            ["example"],
            ["question"],
            dataset_fingerprint="",
            dataset_status="synthetic_smoke",
        )

    with pytest.raises(ValueError, match="held-out test split"):
        from app.training.benchmark import benchmark_router

        benchmark_router(
            "model",
            lambda _text: "question",
            ["example"],
            ["question"],
            dataset_fingerprint="fingerprint",
            split="validation",
            dataset_status="approved_real",
        )


def test_benchmark_comparison_rejects_different_dataset_status() -> None:
    baseline = BenchmarkResult(
        model_name="baseline",
        accuracy=1.0,
        latency_ms=1.0,
        sample_count=1,
        trained=False,
        per_class={},
        confusion_matrix={},
        failure_rate=0.0,
        dataset_fingerprint="same",
        split="test",
        dataset_status="synthetic_smoke",
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
        dataset_fingerprint="same",
        split="test",
        dataset_status="approved_real",
    )

    with pytest.raises(ValueError, match="dataset status"):
        benchmark_comparison(baseline, candidate)


def test_synthetic_fixture_passes_dataset_and_evaluation_readiness(tmp_path) -> None:
    phrases = {
        "question": [
            "explain the retention rule",
            "clarify the access policy",
            "describe the review cycle",
        ],
        "search": [
            "locate the retention rule",
            "find the access policy",
            "search for the review cycle",
        ],
        "sql_report": ["report retention totals", "show access counts", "list review metrics"],
        "email_draft": [
            "draft a retention notice",
            "write an access update",
            "compose a review message",
        ],
        "verify": [
            "validate the retention claim",
            "check the access assertion",
            "verify the review statement",
        ],
    }
    path = tmp_path / "synthetic.jsonl"
    path.write_text(
        "\n".join(
            (
                f'{{"id":"{intent}-{index}","group_id":"{intent}-group-{index}",'
                f'"text":"{text}","intent":"{intent}"}}'
            )
            for intent, texts in phrases.items()
            for index, text in enumerate(texts)
        ),
        encoding="utf-8",
    )

    rows = load_intent_dataset(path)
    splits = split_intent_dataset(rows, seed=2026)
    validate_training_ready(splits)
    result = benchmark_router(
        "synthetic-baseline",
        lambda _text: "question",
        [row["text"] for row in splits.test],
        [row["intent"] for row in splits.test],
        dataset_fingerprint=dataset_fingerprint(list(splits.test)),
        dataset_status="synthetic_smoke",
    )

    assert result.dataset_fingerprint == dataset_fingerprint(list(splits.test))
    assert result.split == "test"
    assert result.dataset_status == "synthetic_smoke"
    assert result.sample_count == 5


def test_unapproved_evaluation_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="not approved"):
        approved_dataset_path("unapproved-fixture")


def test_training_readiness_rejects_group_leakage() -> None:
    rows = [
        {
            "id": f"synthetic-{intent}-{index}",
            "group_id": f"synthetic-{intent}-{index}",
            "text": text,
            "intent": intent,
        }
        for intent, texts in {
            "question": [
                "explain retirement eligibility",
                "clarify badge access requirements",
                "describe annual audit schedule",
            ],
            "search": [
                "locate vendor onboarding records",
                "find security incident summaries",
                "search archived contract clauses",
            ],
            "sql_report": [
                "aggregate quarterly revenue",
                "count unresolved support tickets",
                "summarize invoice aging",
            ],
            "email_draft": [
                "draft supplier renewal notice",
                "compose employee welcome message",
                "write customer escalation response",
            ],
            "verify": [
                "validate checksum against manifest",
                "check claim against source",
                "confirm approval signature",
            ],
        }.items()
        for index, text in enumerate(texts)
    ]
    splits = split_intent_dataset(rows, seed=7)
    leaked = splits.validation + (splits.train[0],)

    invalid = IntentSplits(
        train=splits.train,
        validation=leaked,
        test=splits.test,
        seed=splits.seed,
        fingerprint=splits.fingerprint,
    )
    with pytest.raises(ValueError, match="group leakage"):
        validate_training_ready(invalid)
