"""Versioned API routes for the platform foundation."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/api/v1")


class EvidenceInput(BaseModel):
    """Evidence supplied to deterministic verification."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    score: float = 1.0


class VerificationRequest(BaseModel):
    """Request for claim-level answer verification."""

    answer: str
    evidence: list[EvidenceInput] = Field(default_factory=list)


class ClaimResponse(BaseModel):
    """Public claim verification result."""

    claim: str
    status: str
    evidence_ids: list[str]
    confidence: float
    explanation: str


class VerificationResponse(BaseModel):
    """Public verification response."""

    status: str
    method: str
    claims: list[ClaimResponse]


class HealthResponse(BaseModel):
    """Public health response without internal paths or secrets."""

    status: str
    service: str
    environment: str
    timestamp: datetime


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the application process is serving requests."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )


@router.get("/ready", response_model=HealthResponse, tags=["system"])
def readiness(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report foundation readiness; dependency checks will be added by later phases."""

    return HealthResponse(
        status="ready",
        service=settings.app_name,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )


@router.post("/verify", response_model=VerificationResponse, tags=["verification"])
def verify(request: VerificationRequest) -> VerificationResponse:
    """Verify answer claims against caller-provided retrieved evidence."""

    from app.retrieval.models import DocumentChunk, RetrievalResult
    from app.verification.service import verify_answer

    evidence = [
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                source_name=item.source_name,
                text=item.text,
            ),
            score=item.score,
        )
        for item in request.evidence
    ]
    result = verify_answer(request.answer, evidence)
    return VerificationResponse(
        status=result.status.value,
        method=result.method,
        claims=[
            ClaimResponse(
                claim=claim.claim,
                status=claim.status.value,
                evidence_ids=list(claim.evidence_ids),
                confidence=claim.confidence,
                explanation=claim.explanation,
            )
            for claim in result.claims
        ],
    )
