"""Validation for small intent-routing datasets."""

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

VALID_INTENTS = {"question", "search", "sql_report", "email_draft", "verify"}


def load_intent_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load JSONL rows with optional annotation IDs for legacy fixtures."""

    rows: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    seen_ids: set[str] = set()
    group_intents: dict[str, str] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError
            text = payload["text"].strip() if isinstance(payload["text"], str) else ""
            intent = payload["intent"].strip() if isinstance(payload["intent"], str) else ""
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid intent example on line {line_number}") from exc
        if not text or intent not in VALID_INTENTS:
            raise ValueError(f"invalid intent example on line {line_number}")
        row: dict[str, str] = {"text": text, "intent": intent}
        for field in ("id", "group_id"):
            value = payload.get(field)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"invalid intent example on line {line_number}")
                row[field] = value.strip()
        if "id" in row and row["id"] in seen_ids:
            raise ValueError(f"duplicate intent example id on line {line_number}")
        group_id = row.get("group_id", row.get("id", text.casefold()))
        previous_intent = group_intents.get(group_id)
        if previous_intent is not None and previous_intent != intent:
            raise ValueError(f"group has conflicting intents on line {line_number}")
        normalized = text.casefold()
        if normalized in seen_texts:
            raise ValueError(f"duplicate intent example on line {line_number}")
        seen_texts.add(normalized)
        if "id" in row:
            seen_ids.add(row["id"])
        group_intents[group_id] = intent
        rows.append(row)
    if not rows:
        raise ValueError("intent dataset is empty")
    return rows


@dataclass(frozen=True)
class IntentSplits:
    """Reproducible train/validation/test partition."""

    train: tuple[dict[str, str], ...]
    validation: tuple[dict[str, str], ...]
    test: tuple[dict[str, str], ...]
    seed: int
    fingerprint: str
    warnings: tuple[str, ...] = ()


def validate_training_ready(splits: IntentSplits) -> None:
    """Reject partitions that cannot support a valid train/validation/test run."""

    missing = [
        name
        for name, rows in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
        if not rows
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"dataset is not ready for training: required split(s) are empty: {joined}; "
            "provide a larger approved dataset without duplicating examples"
        )
    if len(splits.train) < 2:
        raise ValueError(
            "dataset is not ready for training: at least two training examples are required"
        )
    for split_name, rows in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        missing = sorted(VALID_INTENTS - {row["intent"] for row in rows})
        if missing:
            raise ValueError(
                f"dataset is not ready for training: {split_name} is missing classes "
                f"{', '.join(missing)}"
            )
    split_groups = {
        name: {_group_id(row) for row in rows}
        for name, rows in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
    }
    if split_groups["train"] & split_groups["validation"]:
        raise ValueError(
            "dataset is not ready for training: group leakage between train and validation"
        )
    if split_groups["train"] & split_groups["test"]:
        raise ValueError(
            "dataset is not ready for training: group leakage between train and test"
        )
    if split_groups["validation"] & split_groups["test"]:
        raise ValueError(
            "dataset is not ready for training: group leakage between validation and test"
        )


def _group_id(row: dict[str, str]) -> str:
    """Return an explicit group or isolate legacy rows by stable text."""

    return row.get("group_id", row.get("id", row["text"].casefold().strip()))


def dataset_fingerprint(rows: list[dict[str, str]]) -> str:
    """Hash normalized examples without logging their contents."""

    payload = "\n".join(
        f"{row['intent']}\t{row['text'].casefold().strip()}" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_intent_dataset(
    rows: list[dict[str, str]],
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> IntentSplits:
    """Split without cross-partition duplicates and report small-class limitations."""

    if not rows:
        raise ValueError("intent dataset is empty")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    normalized = [row["text"].casefold().strip() for row in rows]
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1 :]:
            if SequenceMatcher(None, left, right).ratio() >= 0.92:
                raise ValueError("near-identical intent examples cannot be split safely")
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        group_id = _group_id(row)
        grouped.setdefault(row["intent"], {}).setdefault(group_id, []).append(row)
    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    warnings: list[str] = []
    for intent, groups in sorted(grouped.items()):
        shuffled = list(groups.values())
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_fraction)) if len(shuffled) > 1 else 0
        validation_count = (
            max(1, round(len(shuffled) * validation_fraction))
            if len(shuffled) - test_count > 1
            else 0
        )
        if test_count + validation_count >= len(shuffled) and len(shuffled) > 2:
            validation_count = max(1, len(shuffled) - test_count - 1)
        if len(shuffled) < 3:
            warnings.append(f"class {intent} has fewer than three examples")
        test.extend(item for group in shuffled[:test_count] for item in group)
        validation.extend(
            item
            for group in shuffled[test_count : test_count + validation_count]
            for item in group
        )
        train.extend(item for group in shuffled[test_count + validation_count :] for item in group)
    if not test and len(train) > 1:
        fallback_group = _group_id(train[-1])
        fallback = [row for row in train if _group_id(row) == fallback_group]
        train = [row for row in train if _group_id(row) != fallback_group]
        test.extend(fallback)
        warnings.append("test split uses one singleton-class example")
    if not train or not test:
        raise ValueError("split must contain non-empty train and test partitions")
    if len(grouped) > 1 and not validation:
        warnings.append("validation split is empty because the dataset is too small")
    return IntentSplits(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        seed=seed,
        fingerprint=dataset_fingerprint(rows),
        warnings=tuple(warnings),
    )


def class_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    """Return counts without exposing example text."""

    return dict(Counter(row["intent"] for row in rows))
