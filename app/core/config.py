"""Validated application settings loaded from environment variables."""

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for local development and deployment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EAKTP_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Enterprise AI Knowledge & Trust Platform"
    environment: str = "development"
    log_level: str = "INFO"
    api_key: SecretStr | None = None
    read_api_key: SecretStr | None = None
    write_api_key: SecretStr | None = None
    ui_api_key: SecretStr | None = None
    api_principals_json: SecretStr | None = None
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_rate_limit_requests: int = Field(default=120, ge=1)
    api_rate_limit_window_seconds: int = Field(default=60, ge=1)
    cors_origins: list[str] = ["http://localhost:8501"]
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str | None = None
    ollama_timeout_seconds: float = Field(default=30.0, gt=0)
    embedding_backend: Literal["hash", "sentence-transformers"] = "hash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_backend: Literal["memory", "sqlite", "chroma"] = "sqlite"
    sqlite_vector_path: str = "data/runtime/eaktp.sqlite3"
    chroma_path: str = "data/runtime/chroma"
    evaluation_database: str = "data/runtime/evaluation.db"
    training_output_dir: str = "artifacts/training"

    @model_validator(mode="after")
    def require_api_key_outside_development(self) -> "Settings":
        """Prevent production-like deployments from starting without API protection."""

        if self.environment.strip().lower() not in {"development", "dev", "test"}:
            admin_key = self.api_key.get_secret_value().strip() if self.api_key else ""
            registry_has_admin = False
            if self.api_principals_json is not None:
                try:
                    principals = json.loads(self.api_principals_json.get_secret_value())
                    registry_has_admin = isinstance(principals, list) and any(
                        isinstance(item, dict)
                        and item.get("role") == "admin"
                        and isinstance(item.get("key"), str)
                        and item["key"].strip()
                        for item in principals
                    )
                except json.JSONDecodeError:
                    # The dedicated principal validator reports malformed JSON clearly.
                    pass
            if not admin_key and not registry_has_admin:
                raise ValueError(
                    "Configure EAKTP_API_KEY or an admin principal in "
                    "EAKTP_API_PRINCIPALS_JSON outside development"
                )
            if admin_key and len(admin_key) < 32:
                raise ValueError("EAKTP_API_KEY must contain at least 32 characters")
        return self

    @model_validator(mode="after")
    def validate_api_principals(self) -> "Settings":
        """Reject malformed or ambiguous principal credentials during startup."""

        if self.api_principals_json is None:
            return self
        try:
            raw_principals = json.loads(self.api_principals_json.get_secret_value())
        except json.JSONDecodeError as exc:
            raise ValueError("EAKTP_API_PRINCIPALS_JSON must contain valid JSON") from exc
        if not isinstance(raw_principals, list):
            raise ValueError("EAKTP_API_PRINCIPALS_JSON must be a JSON list")

        keys: set[str] = set()
        principal_ids: set[str] = set()
        configured_legacy_keys = {
            "reader": self.read_api_key,
            "writer": self.write_api_key,
            "admin": self.api_key,
        }
        legacy_assignments: dict[str, str] = {}
        for role, secret in configured_legacy_keys.items():
            if secret is None or not secret.get_secret_value().strip():
                continue
            value = secret.get_secret_value()
            previous_role = legacy_assignments.get(value)
            if previous_role is not None and previous_role != role:
                raise ValueError("a legacy credential cannot be assigned multiple roles")
            legacy_assignments[value] = role
        for item in raw_principals:
            if not isinstance(item, dict) or not all(
                isinstance(item.get(field), str) and item[field].strip()
                for field in ("key", "principal_id", "tenant_id", "role")
            ):
                raise ValueError(
                    "each API principal needs non-empty key, principal_id, tenant_id, and role"
                )
            key = item["key"]
            principal_id = item["principal_id"]
            tenant_id = item["tenant_id"]
            role = item["role"]
            if len(key) < 32:
                raise ValueError("API principal keys must contain at least 32 characters")
            if role not in {"reader", "writer", "admin"}:
                raise ValueError("API principal role must be reader, writer, or admin")
            if role != "admin" and tenant_id == "*":
                raise ValueError("only administrators may use the all-tenant scope")
            if key in keys:
                raise ValueError("API principal keys must be unique")
            if principal_id in principal_ids:
                raise ValueError("API principal IDs must be unique")
            legacy_role = legacy_assignments.get(key)
            if legacy_role is not None and legacy_role != role:
                raise ValueError("a credential cannot be assigned multiple roles")
            keys.add(key)
            principal_ids.add(principal_id)

        for role, secret in configured_legacy_keys.items():
            if secret is None or not secret.get_secret_value().strip():
                continue
            if secret.get_secret_value() not in keys and f"legacy-{role}" in principal_ids:
                raise ValueError("API principal IDs must be unique across configured credentials")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable-by-convention settings instance."""

    return Settings()
