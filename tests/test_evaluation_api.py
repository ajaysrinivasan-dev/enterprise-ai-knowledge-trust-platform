"""Evaluation API tests with isolated runtime configuration."""

from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient
from pydantic import SecretStr


def _admin_client(tmp_path, monkeypatch) -> TestClient:
    import app.main as main

    settings = Settings(
        api_key=SecretStr("evaluation-admin-" + "a" * 32),
        api_principals_json=None,
        evaluation_database=str(tmp_path / "runs.db"),
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return TestClient(create_app())


def test_evaluation_api_rejects_unknown_dataset(tmp_path, monkeypatch) -> None:
    client = _admin_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/evaluations",
        json={"dataset_id": "arbitrary"},
        headers={"Authorization": "Bearer evaluation-admin-" + "a" * 32},
    )

    assert response.status_code == 422
    assert "approved" in response.json()["detail"]


def test_evaluation_api_runs_and_lists_approved_dataset(tmp_path, monkeypatch) -> None:
    client = _admin_client(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer evaluation-admin-" + "a" * 32}
    response = client.post(
        "/api/v1/evaluations",
        json={"dataset_id": "synthetic_v1"},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_version"] == "synthetic_v1"
    assert len(payload["dataset_fingerprint"]) == 64
    assert payload["dataset_status"] == "synthetic_smoke"
    assert "failure_rate" in payload["metrics"]

    listed = client.get("/api/v1/evaluations", headers=headers)
    assert listed.status_code == 200
    assert listed.json()[0]["run_id"] == payload["run_id"]
