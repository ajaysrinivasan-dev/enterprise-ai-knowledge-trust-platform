"""Verification API tests."""

from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient
from pydantic import SecretStr


def test_verify_api_returns_structured_claim_results(monkeypatch) -> None:
    import app.main as main

    settings = Settings(api_key=SecretStr("verify-admin-" + "a" * 32), api_principals_json=None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    response = TestClient(
        create_app(), headers={"Authorization": "Bearer verify-admin-" + "a" * 32}
    ).post(
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
