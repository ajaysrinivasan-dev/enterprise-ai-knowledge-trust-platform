"""Typed workflow state and public execution results."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.generation.models import GroundedAnswer
from app.retrieval.models import RetrievalResult
from app.verification.models import VerificationResult


class Intent(StrEnum):
    """Supported workflow routes."""

    QUESTION = "question"
    SEARCH = "search"
    SQL_REPORT = "sql_report"
    EMAIL_DRAFT = "email_draft"
    VERIFY = "verify"


@dataclass
class AgentState:
    """Mutable state passed through explicit workflow nodes."""

    request: dict[str, Any]
    intent: Intent | None = None
    evidence: list[RetrievalResult] = field(default_factory=list)
    answer: GroundedAnswer | None = None
    verification: VerificationResult | None = None
    result: Any = None
    approval_required: bool = False
    approved: bool = False
    error: str | None = None
    retries: int = 0
    trace: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentResult:
    """Safe workflow output without raw request or document logging."""

    intent: Intent | None
    result: Any
    approval_required: bool
    trace: tuple[str, ...]
    error: str | None
