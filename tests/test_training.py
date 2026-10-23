"""CPU-only training benchmark foundation tests."""

from types import SimpleNamespace

import pytest
from app.training.benchmark import benchmark_router
from app.training.config import TrainingConfig
from app.training.dataset import (
    class_distribution,
    load_intent_dataset,
    split_intent_dataset,
    validate_training_ready,
)
from app.training.lora import _encode_supervised, _target_modules
from app.training.preflight import collect_preflight


def test_intent_dataset_validates_and_benchmark_is_measured() -> None:
    rows = load_intent_dataset("data/training/intent_v1.jsonl")
    result = benchmark_router(
        "deterministic-baseline",
        lambda text: "question" if text.startswith("What") else "search",
        [rows[0]["text"], rows[1]["text"]],
        [rows[0]["intent"], rows[1]["intent"]],
        dataset_fingerprint="smoke-fingerprint",
        dataset_status="synthetic_smoke",
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


def test_annotated_schema_rejects_duplicate_ids_and_conflicting_groups(tmp_path) -> None:
    duplicate_ids = tmp_path / "duplicate_ids.jsonl"
    duplicate_ids.write_text(
        '{"id":"same","group_id":"g1","text":"one","intent":"question"}\n'
        '{"id":"same","group_id":"g2","text":"two","intent":"question"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="id"):
        load_intent_dataset(duplicate_ids)

    conflicting_groups = tmp_path / "conflicting_groups.jsonl"
    conflicting_groups.write_text(
        '{"id":"one","group_id":"g1","text":"one","intent":"question"}\n'
        '{"id":"two","group_id":"g1","text":"two","intent":"search"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting"):
        load_intent_dataset(conflicting_groups)


def test_split_keeps_annotation_groups_together() -> None:
    phrases = {
        "email_draft": ["compose a note", "prepare a message", "write an update"],
        "question": ["explain the rule", "what does this mean", "answer the policy query"],
        "search": ["locate the clause", "find the handbook section", "search the repository"],
        "sql_report": ["count the records", "group documents by type", "report the totals"],
        "verify": ["check the citation", "validate the claim", "review the answer"],
    }
    rows = [
        {
            "id": f"{intent}-{index}",
            "group_id": f"{intent}-group-{index}",
            "text": phrases[intent][index],
            "intent": intent,
        }
        for intent in sorted({"question", "search", "sql_report", "email_draft", "verify"})
        for index in range(3)
    ]
    splits = split_intent_dataset(rows, seed=3)
    locations = {}
    for name, partition in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        for row in partition:
            locations.setdefault(row["group_id"], set()).add(name)
    assert all(len(names) == 1 for names in locations.values())
    validate_training_ready(splits)


def test_training_readiness_rejects_missing_class_coverage() -> None:
    phrases = {
        "email_draft": ["compose a note", "prepare a message", "write an update", "draft a reply"],
        "question": [
            "explain the rule",
            "what does this mean",
            "answer the policy query",
            "clarify the process",
        ],
        "search": [
            "locate the clause",
            "find the handbook section",
            "search the repository",
            "look up the procedure",
        ],
        "sql_report": [
            "count the records",
            "group documents by type",
            "report the totals",
            "show the database summary",
        ],
    }
    rows = [
        {"text": phrases[intent][index], "intent": intent}
        for intent in sorted({"question", "search", "sql_report", "email_draft"})
        for index in range(4)
    ]
    splits = split_intent_dataset(rows, seed=5)
    with pytest.raises(ValueError, match="missing classes"):
        validate_training_ready(splits)


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


def test_lora_target_selection_supports_common_attention_shapes() -> None:
    gpt2_like = SimpleNamespace(
        named_modules=lambda: [("", object()), ("transformer.c_attn", object())]
    )
    llama_like = SimpleNamespace(
        named_modules=lambda: [
            ("model.q_proj", object()),
            ("model.k_proj", object()),
            ("model.v_proj", object()),
            ("model.o_proj", object()),
        ]
    )

    assert _target_modules(gpt2_like) == ["c_attn"]
    assert _target_modules(llama_like) == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_supervised_encoding_masks_prompt_tokens() -> None:
    class Tokenizer:
        pad_token_id = 0

        def __call__(self, value, **kwargs):
            del kwargs
            tokens = list(range(1, len(value.split()) + 1))
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    encoded = _encode_supervised(
        Tokenizer(),
        ({"text": "retention policy", "intent": "question"},),
        max_length=20,
    )

    labels = encoded["labels"][0]
    assert labels[-1] != -100
    assert all(label == -100 for label in labels[:-1])
