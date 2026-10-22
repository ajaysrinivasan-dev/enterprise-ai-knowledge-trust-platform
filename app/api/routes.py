"""Versioned API routes for the platform foundation."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/api/v1")


class HealthResponse(BaseModel):
    """Public health response without internal paths or secrets."""

    status: str
    service: str
    environment: str
    timestamp: datetime


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the application process is serving requests."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )


@router.get("/ready", response_model=HealthResponse, tags=["system"])
def readiness(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report foundation readiness; dependency checks will be added by later phases."""

    return HealthResponse(
        status="ready",
        service=settings.app_name,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )
