"""Training API safety tests."""

from app.main import app
from fastapi.testclient import TestClient


def test_training_api_requires_explicit_confirmation() -> None:
    response = TestClient(app).post(
        "/api/v1/training/runs",
        json={"dataset_id": "intent_v1", "confirm": False},
    )

    assert response.status_code == 400
    assert "confirmation" in response.json()["detail"]


def test_training_dataset_and_preflight_endpoints_are_available() -> None:
    client = TestClient(app)
    datasets = client.get("/api/v1/training/datasets")
    preflight = client.get("/api/v1/training/preflight")

    assert datasets.status_code == 200
    assert datasets.json()[0]["dataset_id"] == "intent_v1"
    assert preflight.status_code == 200
    assert "optional_packages" in preflight.json()


def test_training_api_rejects_smoke_fixture_before_background_job() -> None:
    response = TestClient(app).post(
        "/api/v1/training/runs",
        json={"dataset_id": "intent_v1", "confirm": True},
    )

    assert response.status_code == 422
    assert "validation" in response.json()["detail"]
