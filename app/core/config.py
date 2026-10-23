"""Validated application settings loaded from environment variables."""

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
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
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
            if self.api_key is None or not self.api_key.get_secret_value().strip():
                raise ValueError(
                    "EAKTP_API_KEY is required when EAKTP_ENVIRONMENT is not development"
                )
            if len(self.api_key.get_secret_value()) < 32:
                raise ValueError("EAKTP_API_KEY must contain at least 32 characters")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable-by-convention settings instance."""

    return Settings()
