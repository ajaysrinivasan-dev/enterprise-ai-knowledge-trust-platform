"""Validated application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
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
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = ["http://localhost:8501"]
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable-by-convention settings instance."""

    return Settings()

