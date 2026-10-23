"""Training API safety tests."""

from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient
from pydantic import SecretStr


def _admin_client(monkeypatch) -> TestClient:
    import app.main as main

    settings = Settings(api_key=SecretStr("training-admin-" + "a" * 32), api_principals_json=None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return TestClient(
        create_app(), headers={"Authorization": "Bearer training-admin-" + "a" * 32}
    )


def test_training_api_requires_explicit_confirmation(monkeypatch) -> None:
    response = _admin_client(monkeypatch).post(
        "/api/v1/training/runs",
        json={"dataset_id": "intent_v1", "confirm": False},
    )

    assert response.status_code == 400
    assert "confirmation" in response.json()["detail"]


def test_training_dataset_and_preflight_endpoints_are_available(monkeypatch) -> None:
    client = _admin_client(monkeypatch)
    datasets = client.get("/api/v1/training/datasets")
    preflight = client.get("/api/v1/training/preflight")

    assert datasets.status_code == 200
    assert datasets.json()[0]["dataset_id"] == "intent_v1"
    assert preflight.status_code == 200
    assert "optional_packages" in preflight.json()


def test_training_api_rejects_smoke_fixture_before_background_job(monkeypatch) -> None:
    response = _admin_client(monkeypatch).post(
        "/api/v1/training/runs",
        json={"dataset_id": "intent_v1", "confirm": True},
    )

    assert response.status_code == 422
    assert "validation" in response.json()["detail"]
