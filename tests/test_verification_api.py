"""Verification API tests."""

from app.main import app
from fastapi.testclient import TestClient


def test_verify_api_returns_structured_claim_results() -> None:
    response = TestClient(app).post(
        "/api/v1/verify",
        json={
            "answer": "The retention policy is seven years.",
            "evidence": [
                {
                    "chunk_id": "doc:0",
                    "document_id": "doc",
                    "source_name": "policy.txt",
                    "text": "The retention policy is seven years.",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "supported"
    assert payload["claims"][0]["evidence_ids"] == ["doc:0"]
