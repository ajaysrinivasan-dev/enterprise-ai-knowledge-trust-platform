"""Versioned JSONL evaluation dataset loading."""

import json
from pathlib import Path

from app.evaluation.models import EvaluationCase


def load_jsonl(path: str | Path) -> list[EvaluationCase]:
    """Load and validate a versioned synthetic evaluation fixture."""

    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not str(payload["question"]).strip():
                raise ValueError("question is empty")
            cases.append(
                EvaluationCase(
                    case_id=str(payload["case_id"]),
                    question=str(payload["question"]),
                    relevant_chunk_ids=tuple(str(item) for item in payload["relevant_chunk_ids"]),
                    expected_answerable=bool(payload["expected_answerable"]),
                    reference_answer=payload.get("reference_answer"),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation case on line {line_number}") from exc
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases
