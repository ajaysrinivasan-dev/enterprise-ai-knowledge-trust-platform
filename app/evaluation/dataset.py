"""Versioned JSONL evaluation dataset loading."""

import hashlib
import json
from pathlib import Path

from app.evaluation.models import EvaluationCase


def load_jsonl(path: str | Path) -> list[EvaluationCase]:
    """Load and validate a versioned synthetic evaluation fixture."""

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            case_id = str(payload["case_id"]).strip()
            if not case_id or case_id in seen_ids:
                raise ValueError("case_id must be unique and non-empty")
            question = str(payload["question"]).strip()
            relevant_ids = payload["relevant_chunk_ids"]
            if not question or not isinstance(relevant_ids, list):
                raise ValueError("question is empty")
            cases.append(
                EvaluationCase(
                    case_id=case_id,
                    question=question,
                    relevant_chunk_ids=tuple(str(item) for item in relevant_ids),
                    expected_answerable=bool(payload["expected_answerable"]),
                    reference_answer=payload.get("reference_answer"),
                )
            )
            seen_ids.add(case_id)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation case on line {line_number}") from exc
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


def dataset_fingerprint(cases: list[EvaluationCase]) -> str:
    """Return a stable fingerprint without exposing evaluation text."""

    payload = "\n".join(
        f"{case.case_id}\t{case.question}\t{','.join(case.relevant_chunk_ids)}"
        for case in cases
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
