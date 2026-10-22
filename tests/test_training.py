"""CPU-only training benchmark foundation tests."""

import pytest
from app.training.benchmark import benchmark_router
from app.training.dataset import load_intent_dataset


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


def test_intent_dataset_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"text":"same","intent":"question"}\n{"text":"SAME","intent":"question"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_intent_dataset(path)
