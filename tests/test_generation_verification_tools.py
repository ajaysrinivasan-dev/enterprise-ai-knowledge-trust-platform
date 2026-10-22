"""Tests for grounded answers, verification, and controlled tools."""

import sqlite3

import pytest
from app.generation.models import (
    Citation,
    GroundedAnswer,
    retrieval_only_answer,
    validate_citations,
)
from app.retrieval.models import DocumentChunk, RetrievalResult
from app.tools.email import create_email_draft
from app.tools.sql import ReadOnlySqlTool
from app.verification.models import ClaimStatus
from app.verification.service import verify_answer


def _result(text: str, chunk_id: str = "doc:0") -> RetrievalResult:
    return RetrievalResult(DocumentChunk(chunk_id, "doc", "policy.txt", text), 1.0)


def test_grounded_answer_citations_are_bound_to_retrieved_chunks() -> None:
    results = [_result("Retention policy is seven years.")]

    answer = retrieval_only_answer("retention", results)
    validate_citations(answer, results)

    assert answer.abstained is False
    assert answer.citations[0].chunk_id == "doc:0"


def test_missing_evidence_abstains_and_invalid_citation_is_rejected() -> None:
    answer = retrieval_only_answer("unknown", [])
    assert answer.abstained is True
    with pytest.raises(ValueError):
        invalid = GroundedAnswer("x", (Citation("missing", "x", "x"),), False)
        validate_citations(invalid, [_result("x")])


def test_verifier_distinguishes_supported_and_unsupported_claims() -> None:
    result = verify_answer(
        "Retention policy is seven years. Benefits include free parking.",
        [_result("Retention policy is seven years.")],
    )

    assert result.claims[0].status is ClaimStatus.SUPPORTED
    assert result.claims[1].status is ClaimStatus.UNSUPPORTED
    assert result.status is ClaimStatus.UNSUPPORTED


def test_sql_tool_allows_named_report_only(tmp_path) -> None:
    database = tmp_path / "metadata.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE documents (media_type TEXT)")
    connection.execute("INSERT INTO documents VALUES (?)", (".txt",))
    connection.commit()
    connection.close()

    report = ReadOnlySqlTool(str(database)).run("document_counts")

    assert report.columns == ("media_type", "count")
    assert report.rows == ((".txt", 1),)
    with pytest.raises(ValueError):
        ReadOnlySqlTool(str(database)).run("DROP TABLE documents")


def test_email_tool_only_creates_approval_required_draft() -> None:
    draft = create_email_draft("owner@example.com", "Review", "Please review.")
    assert draft.requires_approval is True
