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
    """Load JSONL rows and reject duplicate or leaking examples."""

    rows: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            text = str(payload["text"]).strip()
            intent = str(payload["intent"]).strip()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid intent example on line {line_number}") from exc
        if not text or intent not in VALID_INTENTS:
            raise ValueError(f"invalid intent example on line {line_number}")
        normalized = text.casefold()
        if normalized in seen_texts:
            raise ValueError(f"duplicate intent example on line {line_number}")
        seen_texts.add(normalized)
        rows.append({"text": text, "intent": intent})
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
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["intent"], []).append(row)
    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    warnings: list[str] = []
    for intent, examples in sorted(grouped.items()):
        shuffled = list(examples)
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_fraction)) if len(shuffled) > 1 else 0
        validation_count = (
            max(1, round(len(shuffled) * validation_fraction))
            if len(shuffled) - test_count > 1
            else 0
        )
        if len(shuffled) < 3:
            warnings.append(f"class {intent} has fewer than three examples")
        test.extend(shuffled[:test_count])
        validation.extend(shuffled[test_count : test_count + validation_count])
        train.extend(shuffled[test_count + validation_count :])
    if not test and len(train) > 1:
        test.append(train.pop())
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
