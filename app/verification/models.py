"""Verification result models."""

from dataclasses import dataclass
from enum import StrEnum


class ClaimStatus(StrEnum):
    """Supported claim classifications."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ClaimResult:
    """A claim classification linked to evidence identifiers."""

    claim: str
    status: ClaimStatus
    evidence_ids: tuple[str, ...]
    confidence: float
    explanation: str


@dataclass(frozen=True)
class VerificationResult:
    """Claim results and an aggregate answer status."""

    claims: tuple[ClaimResult, ...]
    status: ClaimStatus
    method: str


@dataclass(frozen=True)
class JudgeDecision:
    """Validated decision returned by an optional entailment/judge provider."""

    status: ClaimStatus
    confidence: float
    explanation: str
    evidence_ids: tuple[str, ...]
