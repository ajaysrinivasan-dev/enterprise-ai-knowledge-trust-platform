"""Deterministic lexical claim verification for tests and local fallback."""

import re
from collections.abc import Callable, Sequence

from app.retrieval.models import RetrievalResult
from app.verification.models import ClaimResult, ClaimStatus, JudgeDecision, VerificationResult


def extract_claims(answer: str) -> list[str]:
    """Split prose into simple non-empty factual units."""

    return [part.strip(" -") for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip()]


def verify_answer(answer: str, evidence: list[RetrievalResult]) -> VerificationResult:
    """Classify claims with a conservative deterministic fallback."""

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
        contradiction = [
            result.chunk.chunk_id
            for result in evidence
            if _contradicts(claim, result.chunk.text)
        ]
        if contradiction:
            results.append(
                ClaimResult(
                    claim=claim,
                    status=ClaimStatus.CONTRADICTED,
                    evidence_ids=tuple(contradiction),
                    confidence=0.75,
                    explanation="Retrieved evidence explicitly conflicts with this claim.",
                )
            )
        elif matched:
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
    statuses = {item.status for item in results}
    status = ClaimStatus.SUPPORTED if statuses == {ClaimStatus.SUPPORTED} else (
        ClaimStatus.CONTRADICTED
        if ClaimStatus.CONTRADICTED in statuses
        else ClaimStatus.UNSUPPORTED
    )
    return VerificationResult(tuple(results), status, "deterministic-lexical")


def _contradicts(claim: str, evidence: str) -> bool:
    """Detect explicit negation or conflicting numeric facts conservatively."""

    claim_terms = set(re.findall(r"[a-z0-9]{2,}", claim.casefold()))
    evidence_terms = set(re.findall(r"[a-z0-9]{2,}", evidence.casefold()))
    if len(claim_terms & evidence_terms) < max(2, len(claim_terms) // 2):
        return False
    claim_negative = bool(re.search(r"\b(no|not|never|cannot|isn't|doesn't)\b", claim.casefold()))
    evidence_negative = bool(
        re.search(r"\b(no|not|never|cannot|isn't|doesn't)\b", evidence.casefold())
    )
    if claim_negative != evidence_negative:
        return True
    number_pattern = r"\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    claim_numbers = set(re.findall(number_pattern, claim.casefold()))
    evidence_numbers = set(re.findall(number_pattern, evidence.casefold()))
    return bool(claim_numbers and evidence_numbers and claim_numbers.isdisjoint(evidence_numbers))


def verify_with_judge(
    answer: str,
    evidence: Sequence[RetrievalResult],
    judge: Callable[[str, str], JudgeDecision],
) -> VerificationResult:
    """Use an optional judge while validating all statuses and evidence references."""

    if not evidence:
        return VerificationResult((), ClaimStatus.UNCERTAIN, "judge-unavailable-no-evidence")
    valid_ids = {result.chunk.chunk_id for result in evidence}
    results: list[ClaimResult] = []
    for claim in extract_claims(answer):
        try:
            decision = judge(claim, "\n".join(item.chunk.text for item in evidence))
            if (
                not isinstance(decision.status, ClaimStatus)
                or not 0 <= decision.confidence <= 1
                or any(item not in valid_ids for item in decision.evidence_ids)
            ):
                raise ValueError("judge returned invalid verification output")
            results.append(
                ClaimResult(
                    claim=claim,
                    status=decision.status,
                    evidence_ids=decision.evidence_ids,
                    confidence=decision.confidence,
                    explanation=decision.explanation,
                )
            )
        except (ValueError, RuntimeError, TypeError):
            results.append(
                ClaimResult(
                    claim=claim,
                    status=ClaimStatus.UNCERTAIN,
                    evidence_ids=(),
                    confidence=0.0,
                    explanation="Verification provider failed or returned invalid output.",
                )
            )
    statuses = {item.status for item in results}
    overall = ClaimStatus.SUPPORTED if statuses == {ClaimStatus.SUPPORTED} else (
        ClaimStatus.CONTRADICTED if ClaimStatus.CONTRADICTED in statuses else ClaimStatus.UNCERTAIN
    )
    return VerificationResult(tuple(results), overall, "replaceable-judge")
