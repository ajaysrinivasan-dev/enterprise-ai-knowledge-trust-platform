"""Frontend credential selection and safe API error tests."""

import json

import httpx
import pytest
from app.core.config import Settings, get_settings
from fastapi.testclient import TestClient
from frontend.api_client import (
    ApiAuthenticationError,
    ApiAuthorizationError,
    ApiClient,
    ApiConnectionError,
)
from frontend.auth import (
    FrontendCredentialError,
    resolve_frontend_admin_api_key,
    resolve_frontend_api_key,
)
from pydantic import SecretStr


def test_matching_ui_credential_is_sent_to_the_backend(monkeypatch) -> None:
    import app.main as main

    key = "tenant-a-key-" + "a" * 32
    settings = Settings(
        api_principals_json=SecretStr(
            json.dumps(
                [
                    {
                        "key": key,
                        "principal_id": "alice",
                        "tenant_id": "tenant-a",
                        "role": "writer",
                    }
                ]
            )
        ),
        ui_api_key=SecretStr(key),
    )

    selected = resolve_frontend_api_key(settings)
    client = ApiClient(api_key=selected)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    application = main.create_app()
    application.dependency_overrides[get_settings] = lambda: settings
    response = TestClient(application).get("/api/v1/documents", headers=client.headers)

    assert selected == key
    assert client.headers == {"Authorization": f"Bearer {key}"}
    assert response.status_code == 200


def test_registry_requires_explicit_ui_credential() -> None:
    settings = Settings(
        api_principals_json=SecretStr(
            '[{"key":"tenant-a-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            '"principal_id":"alice","tenant_id":"tenant-a","role":"writer"}]'
        ),
        ui_api_key=None,
    )

    with pytest.raises(FrontendCredentialError, match="EAKTP_UI_API_KEY is required"):
        resolve_frontend_api_key(settings)


def test_registry_rejects_ui_credential_that_backend_will_not_accept() -> None:
    settings = Settings(
        api_principals_json=SecretStr(
            '[{"key":"tenant-a-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            '"principal_id":"alice","tenant_id":"tenant-a","role":"writer"}]'
        ),
        ui_api_key=SecretStr("wrong-key-" + "b" * 32),
    )

    with pytest.raises(FrontendCredentialError, match="does not match"):
        resolve_frontend_api_key(settings)


def test_admin_key_is_resolved_only_for_an_admin_principal() -> None:
    writer = "tenant-a-key-" + "a" * 32
    admin = "platform-admin-key-" + "c" * 32
    settings = Settings(
        api_principals_json=SecretStr(json.dumps([
            {"key": writer, "principal_id": "alice", "tenant_id": "tenant-a", "role": "writer"},
            {
                "key": admin,
                "principal_id": "platform-admin",
                "tenant_id": "admin-home",
                "role": "admin",
            },
        ])),
        ui_api_key=SecretStr(writer),
        admin_ui_api_key=SecretStr(admin),
    )
    assert resolve_frontend_api_key(settings) == writer
    assert resolve_frontend_admin_api_key(settings) == admin


def test_admin_resolver_rejects_writer_key() -> None:
    writer = "tenant-a-key-" + "a" * 32
    settings = Settings(
        api_principals_json=SecretStr(json.dumps([
            {"key": writer, "principal_id": "alice", "tenant_id": "tenant-a", "role": "writer"},
        ])),
        ui_api_key=SecretStr(writer),
        admin_ui_api_key=SecretStr(writer),
    )
    with pytest.raises(FrontendCredentialError, match="must match an administrator principal"):
        resolve_frontend_admin_api_key(settings)


def test_admin_resolver_requires_matching_legacy_admin_key() -> None:
    admin = "legacy-admin-key-" + "c" * 32
    settings = Settings(
        api_key=SecretStr(admin),
        api_principals_json=None,
        admin_ui_api_key=SecretStr(admin),
    )
    assert resolve_frontend_admin_api_key(settings) == admin
    wrong = Settings(
        api_key=SecretStr(admin),
        api_principals_json=None,
        admin_ui_api_key=SecretStr("wrong-" + "d" * 32),
    )
    with pytest.raises(FrontendCredentialError, match="must match the configured EAKTP_API_KEY"):
        resolve_frontend_admin_api_key(wrong)


def test_admin_only_client_uses_admin_header_but_documents_use_writer(monkeypatch) -> None:
    calls = []
    def fake_request(method, url, **kwargs):
        calls.append((url, kwargs.get("headers")))
        request = httpx.Request(method, url)
        return httpx.Response(200, request=request, json=[])
    monkeypatch.setattr(httpx, "request", fake_request)
    client = ApiClient(api_key="writer-key", admin_api_key="admin-key")
    client.list_documents()
    client.list_evaluations()
    client.training_preflight()
    assert calls[0][1] == {"Authorization": "Bearer writer-key"}
    assert calls[1][1] == {"Authorization": "Bearer admin-key"}
    assert calls[2][1] == {"Authorization": "Bearer admin-key"}


def test_admin_only_client_fails_early_when_admin_key_missing(monkeypatch) -> None:
    def unexpected_request(*args, **kwargs):
        pytest.fail("request should not be sent without an admin credential")
    monkeypatch.setattr(httpx, "request", unexpected_request)
    client = ApiClient(api_key="writer-key")
    with pytest.raises(ApiAuthorizationError, match="EAKTP_ADMIN_UI_API_KEY"):
        client.list_evaluations()


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, ApiAuthenticationError),
        (403, ApiAuthorizationError),
    ],
)
def test_api_client_distinguishes_authentication_and_authorization(
    monkeypatch, status_code, error_type
) -> None:
    request = httpx.Request("GET", "http://test/api/v1/documents")
    response = httpx.Response(status_code, request=request)
    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: response)

    with pytest.raises(error_type):
        ApiClient(api_key="key").list_documents()


def test_api_client_reports_connectivity_without_exposing_credentials(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "request", fail)

    with pytest.raises(ApiConnectionError, match="Could not reach FastAPI"):
        ApiClient(api_key="secret-that-must-not-appear").list_documents()


@pytest.mark.parametrize(
    ("path", "expected_message"),
    [
        (
            "/api/v1/evaluations",
            "Evaluation is administrator-only",
        ),
        (
            "/api/v1/training/preflight",
            "Training is administrator-only",
        ),
    ],
)
def test_admin_only_operations_explain_required_role(
    monkeypatch, path: str, expected_message: str
) -> None:
    request = httpx.Request("GET", f"http://test{path}")
    response = httpx.Response(
        403,
        request=request,
        json={"detail": "Administrator access required"},
    )
    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: response)

    client = ApiClient(api_key="a-secret-key-that-must-not-be-displayed")
    with pytest.raises(ApiAuthorizationError, match=expected_message) as exc_info:
        client._request("GET", path)

    assert "writer credential for documents" not in str(exc_info.value)
    assert "a-secret-key-that-must-not-be-displayed" not in str(exc_info.value)


def test_reader_write_denial_explains_writer_requirement(monkeypatch) -> None:
    request = httpx.Request("POST", "http://test/api/v1/documents")
    response = httpx.Response(
        403,
        request=request,
        json={"detail": "Reader access is read-only"},
    )
    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: response)

    with pytest.raises(ApiAuthorizationError, match="requires a writer or administrator"):
        ApiClient(api_key="key").upload_document("sample.txt", b"sample")
