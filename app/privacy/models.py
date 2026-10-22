"""Typed privacy-processing results."""

from dataclasses import dataclass
from enum import StrEnum


class PiiEntityType(StrEnum):
    """Supported normalized PII categories."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    NAME = "NAME"
    ADDRESS = "ADDRESS"


@dataclass(frozen=True)
class PiiEntity:
    """A detected entity using offsets from the original text."""

    entity_type: PiiEntityType
    start: int
    end: int
    text: str
    confidence: float


@dataclass(frozen=True)
class PiiDetectionResult:
    """All detected entities and aggregate counts."""

    entities: tuple[PiiEntity, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Return counts without exposing entity text."""

        counts: dict[str, int] = {}
        for entity in self.entities:
            counts[entity.entity_type.value] = counts.get(entity.entity_type.value, 0) + 1
        return counts


@dataclass(frozen=True)
class RedactionResult:
    """Sanitized text plus safe detection metadata."""

    text: str
    detection: PiiDetectionResult

