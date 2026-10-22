"""Evaluation API tests with isolated runtime configuration."""

from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient


def test_evaluation_api_rejects_unknown_dataset(tmp_path, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "evaluation_database", str(tmp_path / "runs.db"))

    response = TestClient(app).post("/api/v1/evaluations", json={"dataset_id": "arbitrary"})

    assert response.status_code == 422
    assert "approved" in response.json()["detail"]


def test_evaluation_api_runs_and_lists_approved_dataset(tmp_path, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "evaluation_database", str(tmp_path / "runs.db"))
    client = TestClient(app)
    response = client.post("/api/v1/evaluations", json={"dataset_id": "synthetic_v1"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_version"] == "synthetic_v1"
    assert "failure_rate" in payload["metrics"]

    listed = client.get("/api/v1/evaluations")
    assert listed.status_code == 200
    assert listed.json()[0]["run_id"] == payload["run_id"]
