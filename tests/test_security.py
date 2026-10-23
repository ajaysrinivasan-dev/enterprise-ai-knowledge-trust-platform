"""API key access-control and deployment configuration tests."""

import pytest
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError


def test_configured_api_key_protects_api_but_keeps_health_public(monkeypatch) -> None:
    import app.main as main

    settings = Settings(api_key=SecretStr("a" * 40))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(create_app())

    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/ready").status_code == 200
    unauthorized = client.get("/api/v1/documents")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert (
        client.get("/api/v1/documents", headers={"Authorization": f"Bearer {'a' * 40}"}).status_code
        == 200
    )
    assert (
        client.get("/api/v1/documents", headers={"Authorization": f"Bearer {'b' * 40}"}).status_code
        == 401
    )


def test_production_settings_require_a_long_api_key() -> None:
    with pytest.raises(ValidationError, match="EAKTP_API_KEY is required"):
        Settings(environment="production")

    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(environment="production", api_key=SecretStr("too-short"))

    settings = Settings(environment="production", api_key=SecretStr("s" * 40))
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "s" * 40


def test_development_settings_can_leave_api_key_unconfigured() -> None:
    assert Settings(environment="development").api_key is None


def test_blank_development_api_key_does_not_enable_authentication(monkeypatch) -> None:
    import app.main as main

    settings = Settings(environment="development", api_key=SecretStr(""))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    response = TestClient(create_app()).get("/api/v1/documents")
    assert response.status_code == 200


def test_reader_key_is_read_only_and_cannot_access_admin_routes(monkeypatch) -> None:
    import app.main as main

    settings = Settings(read_api_key=SecretStr("r" * 40))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(create_app())
    headers = {"Authorization": f"Bearer {'r' * 40}"}

    assert client.get("/api/v1/documents", headers=headers).status_code == 200
    assert (
        client.post(
            "/api/v1/verify", json={"answer": "hello", "evidence": []}, headers=headers
        ).status_code
        == 403
    )
    assert client.get("/api/v1/training/preflight", headers=headers).status_code == 403


def test_writer_key_can_write_but_not_admin_routes(monkeypatch) -> None:
    import app.main as main

    settings = Settings(write_api_key=SecretStr("w" * 40))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(create_app())
    headers = {"Authorization": f"Bearer {'w' * 40}"}

    assert (
        client.post(
            "/api/v1/verify", json={"answer": "hello", "evidence": []}, headers=headers
        ).status_code
        == 200
    )
    assert client.get("/api/v1/training/preflight", headers=headers).status_code == 403
    assert (
        client.post(
            "/api/v1/agent", json={"intent": "question", "question": "hello"}, headers=headers
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/evaluations", json={"dataset_id": "synthetic_v1"}, headers=headers
        ).status_code
        == 403
    )


def test_admin_key_can_access_restricted_routes(monkeypatch) -> None:
    import app.main as main

    settings = Settings(api_key=SecretStr("a" * 40))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(create_app())
    headers = {"Authorization": f"Bearer {'a' * 40}"}

    assert client.get("/api/v1/training/preflight", headers=headers).status_code == 200
    assert (
        client.post(
            "/api/v1/verify", json={"answer": "hello", "evidence": []}, headers=headers
        ).status_code
        == 200
    )


def test_invalid_role_key_is_unauthorized_when_role_keys_are_configured(monkeypatch) -> None:
    import app.main as main

    settings = Settings(
        read_api_key=SecretStr("r" * 40),
        write_api_key=SecretStr("w" * 40),
        api_key=SecretStr("a" * 40),
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    response = TestClient(create_app()).get(
        "/api/v1/documents", headers={"Authorization": "Bearer invalid"}
    )
    assert response.status_code == 401
