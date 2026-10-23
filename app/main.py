"""FastAPI application entry point."""

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize process-wide services without hiding startup failures."""

    settings = get_settings()
    configure_logging(settings.log_level)
    structlog.get_logger(__name__).info("application_started", environment=settings.environment)
    yield


def create_app() -> FastAPI:
    """Build the application instance for the server and tests."""

    settings = get_settings()
    application = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def protect_api(request: Request, call_next):
        """Apply shared-key reader, writer, and administrator access levels."""

        path = request.url.path
        public_paths = {"/api/v1/health", "/api/v1/ready"}
        keys = {
            "reader": settings.read_api_key,
            "writer": settings.write_api_key,
            "admin": settings.api_key,
        }
        configured = {
            role: key.get_secret_value()
            for role, key in keys.items()
            if key is not None and key.get_secret_value().strip()
        }
        if path.startswith("/api/v1/") and path not in public_paths and configured:
            supplied = request.headers.get("authorization", "")
            role = next(
                (
                    candidate_role
                    for candidate_role, secret in configured.items()
                    if hmac.compare_digest(supplied, f"Bearer {secret}")
                ),
                None,
            )
            if role is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Valid bearer API key required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            request.state.eaktp_role = role
            if role == "reader" and request.method not in {"GET", "HEAD", "OPTIONS"}:
                return JSONResponse(
                    status_code=403, content={"detail": "Reader access is read-only"}
                )
            admin_only = (
                path.startswith("/api/v1/training/")
                or path == "/api/v1/agent"
                or (path == "/api/v1/evaluations" and request.method != "GET")
            )
            if admin_only and role != "admin":
                return JSONResponse(
                    status_code=403, content={"detail": "Administrator access required"}
                )
        return await call_next(request)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
