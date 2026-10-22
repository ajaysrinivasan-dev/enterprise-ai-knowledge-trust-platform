"""Deterministic verification and workflow tests."""

from app.agents.models import Intent
from app.agents.workflow import AgentWorkflow
from app.main import app
from app.retrieval.models import DocumentChunk, RetrievalResult
from app.verification.models import ClaimStatus, JudgeDecision
from app.verification.service import verify_answer, verify_with_judge
from fastapi.testclient import TestClient


def evidence(text: str, chunk_id: str = "d:0") -> list[RetrievalResult]:
    return [RetrievalResult(DocumentChunk(chunk_id, "d", "policy.txt", text), 1.0)]


def test_verifier_handles_supported_contradicted_unsupported_and_absent() -> None:
    supported = verify_answer("Retention is seven years.", evidence("Retention is seven years."))
    contradicted = verify_answer("Retention is seven years.", evidence("Retention is five years."))
    unsupported = verify_answer("Parking is free.", evidence("Retention is seven years."))
    absent = verify_answer("Retention is seven years.", [])

    assert supported.claims[0].status is ClaimStatus.SUPPORTED
    assert contradicted.claims[0].status is ClaimStatus.CONTRADICTED
    assert unsupported.claims[0].status is ClaimStatus.UNSUPPORTED
    assert absent.status is ClaimStatus.UNCERTAIN


def test_invalid_judge_output_is_uncertain_and_cannot_invent_evidence() -> None:
    result = verify_with_judge(
        "Retention is seven years.",
        evidence("Retention is seven years."),
        lambda _claim, _evidence: JudgeDecision(
            ClaimStatus.SUPPORTED, 1.5, "bad", ("missing",)
        ),
    )
    assert result.claims[0].status is ClaimStatus.UNCERTAIN
    assert result.claims[0].evidence_ids == ()


def test_workflow_routes_search_and_requires_approval_for_email() -> None:
    def retrieve(_question: str, _top_k: int, _min_score: float) -> list[RetrievalResult]:
        return evidence("Retention is seven years.")

    workflow = AgentWorkflow(retrieve)

    search = workflow.run({"intent": Intent.SEARCH, "question": "retention"})
    email = workflow.run(
        {
            "intent": Intent.EMAIL_DRAFT,
            "recipient": "owner@example.com",
            "subject": "Review",
            "body": "Please review.",
        }
    )

    assert search.error is None
    assert "retrieve" in search.trace
    assert email.approval_required is True
    assert email.error is None


def test_workflow_rejects_unsupported_route_and_prevents_send() -> None:
    workflow = AgentWorkflow(lambda _question, _top_k, _min_score: [])
    invalid = workflow.run({"intent": "shell"})
    send_attempt = workflow.run(
        {
            "intent": Intent.EMAIL_DRAFT,
            "recipient": "owner@example.com",
            "subject": "Review",
            "body": "Please review.",
        },
        approved=True,
    )
    assert invalid.error == "unsupported workflow intent"
    assert send_attempt.error == "email sending is not supported; approval cannot send email"


def test_agent_api_returns_trace_and_approval_state() -> None:
    response = TestClient(app).post(
        "/api/v1/agent",
        json={
            "intent": "email_draft",
            "recipient": "owner@example.com",
            "subject": "Review",
            "body": "Please review.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["approval_required"] is True
    assert "approval" in payload["trace"]
