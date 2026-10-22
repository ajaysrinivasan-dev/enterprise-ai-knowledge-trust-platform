"""Deterministic PII detection with an optional Presidio integration point."""

import re
from collections.abc import Iterable

from app.privacy.models import (
    PiiDetectionResult,
    PiiEntity,
    PiiEntityType,
    RedactionResult,
)

_PATTERNS: tuple[tuple[PiiEntityType, re.Pattern[str], float], ...] = (
    (
        PiiEntityType.EMAIL,
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        0.99,
    ),
    (
        PiiEntityType.PHONE,
        re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)"),
        0.88,
    ),
    (
        PiiEntityType.SSN,
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        0.97,
    ),
    (
        PiiEntityType.NAME,
        re.compile(r"(?i)(?<=\bname:\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"),
        0.75,
    ),
    (
        PiiEntityType.ADDRESS,
        re.compile(r"(?i)(?<=\baddress:\s)[^\r\n,]+(?:,\s*[^\r\n]+)?"),
        0.72,
    ),
)


def _non_overlapping(entities: Iterable[PiiEntity]) -> tuple[PiiEntity, ...]:
    """Keep the highest-confidence entity when patterns overlap."""

    selected: list[PiiEntity] = []
    for entity in sorted(entities, key=lambda item: (item.start, -item.confidence)):
        if selected and entity.start < selected[-1].end:
            continue
        selected.append(entity)
    return tuple(selected)


def detect_pii(text: str) -> PiiDetectionResult:
    """Detect common structured PII without logging or returning raw values elsewhere."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    entities: list[PiiEntity] = []
    for entity_type, pattern, confidence in _PATTERNS:
        entities.extend(
            PiiEntity(
                entity_type=entity_type,
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=confidence,
            )
            for match in pattern.finditer(text)
        )
    return PiiDetectionResult(entities=_non_overlapping(entities))


def redact_pii(text: str, detection: PiiDetectionResult | None = None) -> RedactionResult:
    """Replace detected values with stable, document-scoped placeholders."""

    result = detection or detect_pii(text)
    counters: dict[PiiEntityType, int] = {}
    redacted = text
    for entity in reversed(result.entities):
        counters[entity.entity_type] = counters.get(entity.entity_type, 0) + 1
        placeholder = f"[{entity.entity_type.value}_{counters[entity.entity_type]}]"
        redacted = redacted[: entity.start] + placeholder + redacted[entity.end :]
    return RedactionResult(text=redacted, detection=result)
