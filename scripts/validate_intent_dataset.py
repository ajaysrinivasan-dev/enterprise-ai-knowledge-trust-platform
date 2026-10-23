"""Read-only CLI validation for an annotated intent JSONL dataset."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.training.dataset import (
    VALID_INTENTS,
    load_intent_dataset,
    split_intent_dataset,
    validate_training_ready,
)


def validate_dataset(path: str | Path) -> dict[str, Any]:
    """Return a JSON-safe report without including dataset text."""

    report: dict[str, Any] = {
        "ready": False,
        "example_count": 0,
        "class_counts": {intent: 0 for intent in sorted(VALID_INTENTS)},
        "split_counts": {"train": 0, "validation": 0, "test": 0},
        "warnings": [],
        "errors": [],
    }
    try:
        rows = load_intent_dataset(path)
    except (OSError, ValueError):
        report["errors"].append("dataset could not be loaded or failed validation")
        return report

    report["example_count"] = len(rows)
    report["class_counts"] = {
        intent: count for intent, count in sorted(Counter(row["intent"] for row in rows).items())
    }
    try:
        splits = split_intent_dataset(rows)
        report["split_counts"] = {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        }
        report["warnings"] = list(splits.warnings)
        report["fingerprint"] = splits.fingerprint
        try:
            validate_training_ready(splits)
        except ValueError as exc:
            report["errors"].append(str(exc))
        else:
            report["ready"] = True
    except ValueError as exc:
        report["errors"].append(str(exc))
    return report


def main() -> int:
    """Validate one explicitly supplied input file and print a redacted report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="annotated intent JSONL file to validate")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.input_file), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
