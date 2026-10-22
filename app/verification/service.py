"""Deterministic lexical claim verification for tests and local fallback."""

import re

from app.retrieval.models import RetrievalResult
from app.verification.models import ClaimResult, ClaimStatus, VerificationResult


def extract_claims(answer: str) -> list[str]:
    """Split prose into simple non-empty factual units."""

    return [part.strip(" -") for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip()]


def verify_answer(answer: str, evidence: list[RetrievalResult]) -> VerificationResult:
    """Classify claims using token coverage, never defaulting failed checks to support."""

    if not evidence:
        return VerificationResult((), ClaimStatus.UNCERTAIN, "deterministic-lexical")
    evidence_terms = {
        result.chunk.chunk_id: set(re.findall(r"[a-z0-9]{2,}", result.chunk.text.casefold()))
        for result in evidence
    }
    results: list[ClaimResult] = []
    for claim in extract_claims(answer):
        terms = set(re.findall(r"[a-z0-9]{2,}", claim.casefold()))
        matched = [
            chunk_id
            for chunk_id, chunk_terms in evidence_terms.items()
            if terms and len(terms & chunk_terms) / len(terms) >= 0.6
        ]
        if matched:
            results.append(
                ClaimResult(
                    claim=claim,
                    status=ClaimStatus.SUPPORTED,
                    evidence_ids=tuple(matched),
                    confidence=0.8,
                    explanation="Most claim terms occur in retrieved evidence.",
                )
            )
        else:
            results.append(
                ClaimResult(
                    claim=claim,
                    status=ClaimStatus.UNSUPPORTED,
                    evidence_ids=(),
                    confidence=0.2,
                    explanation="No retrieved evidence sufficiently establishes this claim.",
                )
            )
    status = (
        ClaimStatus.SUPPORTED
        if results and all(item.status is ClaimStatus.SUPPORTED for item in results)
        else ClaimStatus.UNSUPPORTED
    )
    return VerificationResult(tuple(results), status, "deterministic-lexical")

