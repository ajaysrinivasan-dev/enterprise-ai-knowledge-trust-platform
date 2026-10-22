"""API foundation tests."""

from app.main import app
from fastapi.testclient import TestClient


def test_health_endpoint_returns_safe_service_status() -> None:
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["environment"] == "development"
    assert "timestamp" in payload
    assert "C:\\" not in response.text


def test_readiness_endpoint_returns_ready() -> None:
    response = TestClient(app).get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
