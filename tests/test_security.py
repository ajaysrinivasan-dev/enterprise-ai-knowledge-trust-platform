"""API key access-control and deployment configuration tests."""

import pytest
from app.core.config import Settings, get_settings
from app.main import create_app
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError


def test_configured_api_key_protects_api_but_keeps_health_public(monkeypatch) -> None:
    import app.api.routes as routes
    import app.main as main

    settings = Settings(api_key=SecretStr("a" * 40))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
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
    with pytest.raises(ValidationError, match="Configure EAKTP_API_KEY or an admin principal"):
        Settings(environment="production")

    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(environment="production", api_key=SecretStr("too-short"))

    settings = Settings(environment="production", api_key=SecretStr("s" * 40))
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "s" * 40


def test_invalid_principal_configuration_fails_during_settings_creation() -> None:
    with pytest.raises(ValidationError, match="must contain valid JSON"):
        Settings(api_principals_json=SecretStr("{"))

    with pytest.raises(ValidationError, match="keys must be unique"):
        Settings(
            api_principals_json=SecretStr(
                f'[{{"key":"{"k" * 32}","principal_id":"a","tenant_id":"t","role":"reader"}},'
                f'{{"key":"{"k" * 32}","principal_id":"b","tenant_id":"t","role":"writer"}}]'
            )
        )

    with pytest.raises(ValidationError, match="cannot be assigned multiple roles"):
        legacy_key = "legacy-key-" + "w" * 32
        Settings(
            write_api_key=SecretStr(legacy_key),
            api_principals_json=SecretStr(
                f'[{{"key":"{legacy_key}","principal_id":"a","tenant_id":"t","role":"reader"}}]'
            ),
        )


def test_development_settings_can_leave_api_key_unconfigured() -> None:
    assert Settings(environment="development").api_key is None


def test_blank_development_api_key_does_not_enable_authentication(monkeypatch) -> None:
    import app.main as main

    settings = Settings(
        environment="development",
        api_key=SecretStr(""),
        api_principals_json=None,
    )
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
    assert client.get("/api/v1/evaluations", headers=headers).status_code == 403
    assert client.get("/api/v1/evaluations/example", headers=headers).status_code == 403


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
    assert client.get("/api/v1/evaluations", headers=headers).status_code != 403


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


def test_principal_registry_isolates_document_listing_and_retrieval(monkeypatch) -> None:
    import json

    import app.api.rag_runtime as runtime
    import app.main as main
    from app.retrieval.embeddings import HashEmbeddingProvider
    from app.retrieval.memory_store import InMemoryVectorStore
    from app.retrieval.service import RetrievalService

    key_a = "tenant-a-key-" + "a" * 32
    key_b = "tenant-b-key-" + "b" * 32
    admin_key = "admin-key-" + "c" * 32
    settings = Settings(
        api_key=SecretStr(admin_key),
        api_principals_json=SecretStr(
            json.dumps(
                [
                    {
                        "key": key_a,
                        "principal_id": "user-a",
                        "tenant_id": "tenant-a",
                        "role": "writer",
                    },
                    {
                        "key": key_b,
                        "principal_id": "user-b",
                        "tenant_id": "tenant-b",
                        "role": "writer",
                    },
                    {
                        "key": admin_key,
                        "principal_id": "admin-1",
                        "tenant_id": "ignored",
                        "role": "admin",
                    },
                ]
            )
        ),
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(
        runtime, "_service", RetrievalService(HashEmbeddingProvider(), InMemoryVectorStore())
    )
    runtime._documents.clear()
    client = TestClient(create_app())

    upload_a = client.post(
        "/api/v1/documents",
        files={"file": ("alpha.txt", b"Tenant alpha secret phrase ORCHID-ALPHA", "text/plain")},
        headers={"Authorization": f"Bearer {key_a}"},
    )
    upload_b = client.post(
        "/api/v1/documents",
        files={"file": ("beta.txt", b"Tenant beta secret phrase ORCHID-BETA", "text/plain")},
        headers={"Authorization": f"Bearer {key_b}"},
    )
    assert upload_a.status_code == upload_b.status_code == 200
    doc_a = upload_a.json()["document_id"]
    doc_b = upload_b.json()["document_id"]

    headers_a = {"Authorization": f"Bearer {key_a}"}
    headers_b = {"Authorization": f"Bearer {key_b}"}
    docs_a = client.get("/api/v1/documents", headers=headers_a).json()
    docs_b = client.get("/api/v1/documents", headers=headers_b).json()
    assert [item["document_id"] for item in docs_a] == [doc_a]
    assert [item["document_id"] for item in docs_b] == [doc_b]
    spoofed_scope = client.get(
        "/api/v1/documents",
        headers={**headers_a, "X-Tenant-ID": "tenant-b"},
    )
    assert [item["document_id"] for item in spoofed_scope.json()] == [doc_a]

    retrieve_a = client.post(
        "/api/v1/retrieve",
        json={"question": "ORCHID-BETA", "top_k": 20, "min_score": 0},
        headers=headers_a,
    )
    retrieve_b = client.post(
        "/api/v1/retrieve",
        json={"question": "ORCHID-ALPHA", "top_k": 20, "min_score": 0},
        headers=headers_b,
    )
    assert retrieve_a.status_code == retrieve_b.status_code == 200
    assert all(item["document_id"] != doc_b for item in retrieve_a.json())
    assert all(item["document_id"] != doc_a for item in retrieve_b.json())

    admin_headers = {"Authorization": f"Bearer {admin_key}"}
    admin_docs = client.get("/api/v1/documents", headers=admin_headers)
    assert {item["document_id"] for item in admin_docs.json()} == {doc_a, doc_b}
    admin_retrieval = client.post(
        "/api/v1/retrieve",
        json={"question": "ORCHID-BETA", "top_k": 20, "min_score": 0},
        headers=admin_headers,
    )
    assert {item["document_id"] for item in admin_retrieval.json()} == {doc_a, doc_b}

    assert client.delete(f"/api/v1/documents/{doc_b}", headers=headers_a).status_code == 404
    assert client.delete(f"/api/v1/documents/{doc_b}", headers=headers_b).status_code == 204
    assert client.get("/api/v1/documents", headers=headers_b).json() == []


def test_upload_rejects_oversized_content_without_reading_unbounded_body(monkeypatch) -> None:
    import app.api.routes as routes
    import app.main as main

    settings = Settings(
        api_principals_json=None,
        api_key=SecretStr("upload-admin-" + "a" * 32),
        max_upload_bytes=4,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(application)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("large.txt", b"12345", "text/plain")},
        headers={"Authorization": "Bearer upload-admin-" + "a" * 32},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "document exceeds the configured size limit"


def test_sqlite_migration_keeps_legacy_documents_unowned_and_hidden(tmp_path) -> None:
    import sqlite3

    from app.retrieval.sqlite_store import SQLiteVectorStore

    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE document_chunks (
                chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                source_name TEXT NOT NULL, text TEXT NOT NULL, page_number INTEGER,
                embedding_json TEXT NOT NULL
            );
            CREATE TABLE indexed_documents (
                document_id TEXT PRIMARY KEY, source_name TEXT NOT NULL,
                pii_counts_json TEXT NOT NULL DEFAULT '{}',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO document_chunks VALUES (
                'legacy:0', 'legacy', 'old.txt', 'legacy confidential text', NULL, '[1.0,0.0]'
            );
            INSERT INTO indexed_documents (document_id, source_name, chunk_count)
                VALUES ('legacy', 'old.txt', 1);
            """
        )

    store = SQLiteVectorStore(str(database))
    assert store.list_documents(tenant_id="tenant-a") == []
    assert [item["document_id"] for item in store.list_documents(tenant_id="*")] == ["legacy"]
    assert store.search([1.0, 0.0], tenant_id="tenant-a") == []
    assert store.search([1.0, 0.0], tenant_id="*")[0].chunk.tenant_id is None


def test_sqlite_store_filters_tenant_owned_chunks_and_catalog(tmp_path) -> None:
    from app.ingestion.models import ParsedDocument
    from app.retrieval.embeddings import HashEmbeddingProvider
    from app.retrieval.service import RetrievalService
    from app.retrieval.sqlite_store import SQLiteVectorStore

    store = SQLiteVectorStore(str(tmp_path / "tenant-store.sqlite3"))
    service = RetrievalService(HashEmbeddingProvider(), store)
    for document_id, tenant_id, filename, text in (
        ("a-doc", "tenant-a", "a.txt", "ALPHA-TENANT-CONTENT"),
        ("b-doc", "tenant-b", "b.txt", "BETA-TENANT-CONTENT"),
    ):
        indexed = service.index(ParsedDocument(filename, ".txt", text), document_id, tenant_id)
        store.upsert_document_metadata(
            {
                "document_id": indexed.document_id,
                "source_name": indexed.source_name,
                "pii_counts": indexed.pii_counts,
                "chunk_count": indexed.chunk_count,
                "tenant_id": tenant_id,
            }
        )

    assert [item["document_id"] for item in store.list_documents("tenant-a")] == ["a-doc"]
    assert [item["document_id"] for item in store.list_documents("tenant-b")] == ["b-doc"]
    results_a = service.retrieve(
        "ALPHA-TENANT-CONTENT", top_k=10, min_score=0, tenant_id="tenant-a"
    )
    assert results_a
    assert {item.chunk.document_id for item in results_a} == {"a-doc"}
    assert {
        item.chunk.document_id
        for item in service.retrieve(
            "ALPHA-TENANT-CONTENT", top_k=10, min_score=0, tenant_id="tenant-b"
        )
    } == {"b-doc"}


def test_production_settings_accept_registry_admin_without_legacy_admin_key() -> None:
    import json

    principal_key = "production-admin-" + "p" * 32
    settings = Settings(
        environment="production",
        api_principals_json=SecretStr(
            json.dumps(
                [
                    {
                        "key": principal_key,
                        "principal_id": "production-admin",
                        "tenant_id": "admin-home",
                        "role": "admin",
                    }
                ]
            )
        ),
    )
    assert settings.api_key is None
    assert settings.api_principals_json is not None


def test_principal_registry_rejects_collision_with_implicit_legacy_identity() -> None:
    import json

    with pytest.raises(ValidationError, match="unique across configured credentials"):
        Settings(
            read_api_key=SecretStr("legacy-reader-key-" + "r" * 32),
            api_principals_json=SecretStr(
                json.dumps(
                    [
                        {
                            "key": "separate-principal-key-" + "s" * 32,
                            "principal_id": "legacy-reader",
                            "tenant_id": "tenant-a",
                            "role": "reader",
                        }
                    ]
                )
            ),
        )


def test_api_rate_limit_returns_429_and_preserves_public_health(monkeypatch) -> None:
    import app.main as main

    settings = Settings(
        api_key=SecretStr("rate-limit-admin-" + "r" * 32),
        api_rate_limit_requests=2,
        api_rate_limit_window_seconds=60,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(create_app())

    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents").status_code == 401
    limited = client.get("/api/v1/documents")
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_declared_oversized_multipart_body_is_rejected_before_route_parsing(monkeypatch) -> None:
    import app.main as main

    settings = Settings(
        api_key=SecretStr("large-upload-admin-" + "u" * 32),
        max_upload_bytes=4,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/documents",
        files={"file": ("large.txt", b"x" * (70 * 1024), "text/plain")},
        headers={"Authorization": f"Bearer {'large-upload-admin-' + 'u' * 32}"},
    )
    assert response.status_code == 413
