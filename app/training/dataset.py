"""Validation for small intent-routing datasets."""

import json
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
