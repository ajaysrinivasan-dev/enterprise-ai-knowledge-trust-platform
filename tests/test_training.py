"""CPU-only training benchmark foundation tests."""

import pytest
from app.training.benchmark import benchmark_router
from app.training.config import TrainingConfig
from app.training.dataset import (
    class_distribution,
    load_intent_dataset,
    split_intent_dataset,
    validate_training_ready,
)
from app.training.preflight import collect_preflight


def test_intent_dataset_validates_and_benchmark_is_measured() -> None:
    rows = load_intent_dataset("data/training/intent_v1.jsonl")
    result = benchmark_router(
        "deterministic-baseline",
        lambda text: "question" if text.startswith("What") else "search",
        [rows[0]["text"], rows[1]["text"]],
        [rows[0]["intent"], rows[1]["intent"]],
    )
    assert len(rows) == 5
    assert result.sample_count == 2
    assert result.trained is False
    assert result.latency_ms >= 0
    assert result.macro_f1 >= 0
    assert result.weighted_f1 >= 0


def test_intent_dataset_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"text":"same","intent":"question"}\n{"text":"SAME","intent":"question"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_intent_dataset(path)


def test_split_is_reproducible_and_reports_tiny_class_limits() -> None:
    rows = load_intent_dataset("data/training/intent_v1.jsonl")
    first = split_intent_dataset(rows, seed=7)
    second = split_intent_dataset(rows, seed=7)

    assert first == second
    assert len(first.train) + len(first.validation) + len(first.test) == len(rows)
    assert class_distribution(rows)["question"] == 1
    assert first.warnings


def test_near_duplicate_split_and_invalid_config_are_rejected() -> None:
    rows = [
        {"text": "Find the policy", "intent": "search"},
        {"text": "find the policy", "intent": "search"},
        {"text": "Draft an email", "intent": "email_draft"},
    ]
    with pytest.raises(ValueError, match="near-identical"):
        split_intent_dataset(rows)
    with pytest.raises(ValueError, match="method"):
        TrainingConfig(method="full").validate()


def test_preflight_is_safe_and_serializable(tmp_path) -> None:
    report = collect_preflight(str(tmp_path / "artifacts"))

    assert report.python_version
    assert report.free_disk_gb > 0
    assert isinstance(report.as_dict()["optional_packages"], dict)


def test_smoke_fixture_is_not_ready_for_real_training() -> None:
    rows = load_intent_dataset("data/training/intent_v1.jsonl")
    splits = split_intent_dataset(rows)

    with pytest.raises(ValueError, match="validation"):
        validate_training_ready(splits)
