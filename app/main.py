"""FastAPI application entry point."""

import asyncio
import hmac
import json
import time
from collections import deque
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
    application.state.rate_limit_buckets: dict[str, deque[float]] = {}
    application.state.rate_limit_lock = asyncio.Lock()

    @application.middleware("http")
    async def protect_api(request: Request, call_next):
        """Apply shared-key reader, writer, and administrator access levels."""

        path = request.url.path
        public_paths = {"/api/v1/health", "/api/v1/ready"}
        security_log = structlog.get_logger("eaktp.security")
        if path.startswith("/api/v1/") and path not in public_paths:
            now = time.monotonic()
            client_host = request.client.host if request.client else "unknown"
            window = settings.api_rate_limit_window_seconds
            async with application.state.rate_limit_lock:
                buckets = application.state.rate_limit_buckets
                if len(buckets) > 10_000:
                    expired_hosts = [
                        host for host, events in buckets.items()
                        if not events or events[-1] <= now - window
                    ]
                    for host in expired_hosts:
                        buckets.pop(host, None)
                bucket = buckets.setdefault(client_host, deque())
                while bucket and bucket[0] <= now - window:
                    bucket.popleft()
                if len(bucket) >= settings.api_rate_limit_requests:
                    retry_after = max(1, int(bucket[0] + window - now + 0.999))
                    rate_limited = True
                else:
                    bucket.append(now)
                    rate_limited = False
            if rate_limited:
                security_log.warning(
                    "api_rate_limit_exceeded",
                    method=request.method,
                    path=path,
                    client_host=client_host,
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "API request rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
        if path == "/api/v1/documents" and request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=400, content={"detail": "Invalid Content-Length"}
                    )
                # Allow multipart boundaries/headers, while rejecting clearly oversized bodies
                # before Starlette's multipart parser consumes the request.
                if declared_length > settings.max_upload_bytes + 64 * 1024:
                    security_log.warning(
                        "api_upload_rejected",
                        reason="declared_body_too_large",
                        client_host=request.client.host if request.client else None,
                        declared_bytes=declared_length,
                    )
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "upload request exceeds the configured size limit"},
                    )
        principals: list[dict[str, str]] = []
        if settings.api_principals_json is not None:
            try:
                raw_principals = json.loads(settings.api_principals_json.get_secret_value())
                if not isinstance(raw_principals, list):
                    raise ValueError("principal configuration must be a JSON list")
                for item in raw_principals:
                    if not isinstance(item, dict) or not all(
                        isinstance(item.get(field), str) and item[field].strip()
                        for field in ("key", "principal_id", "tenant_id", "role")
                    ):
                        raise ValueError(
                            "each principal needs key, principal_id, tenant_id, and role"
                        )
                    if item["role"] not in {"reader", "writer", "admin"}:
                        raise ValueError("principal role must be reader, writer, or admin")
                    if item["role"] != "admin" and item["tenant_id"] == "*":
                        raise ValueError("only administrators may use the all-tenant scope")
                    if len(item["key"]) < 32:
                        raise ValueError("principal keys must contain at least 32 characters")
                    principals.append(
                        {
                            field: item[field]
                            for field in ("key", "principal_id", "tenant_id", "role")
                        }
                    )
            except (json.JSONDecodeError, ValueError) as exc:
                return JSONResponse(
                    status_code=500,
                    content={"detail": f"Invalid API principal configuration: {exc}"},
                )

        # Backward-compatible role keys share the explicit legacy tenant; use the
        # principal registry for separate tenants and identities.
        keys = {
            "reader": settings.read_api_key,
            "writer": settings.write_api_key,
            "admin": settings.api_key,
        }
        for candidate_role, key in keys.items():
            if key is None or not key.get_secret_value().strip():
                continue
            secret = key.get_secret_value()
            existing = next(
                (item for item in principals if hmac.compare_digest(item["key"], secret)),
                None,
            )
            if existing is not None:
                if existing["role"] != candidate_role:
                    return JSONResponse(
                        status_code=500,
                        content={"detail": "A credential cannot be assigned multiple roles"},
                    )
                continue
            principals.append(
                {
                    "key": secret,
                    "principal_id": f"legacy-{candidate_role}",
                    "tenant_id": "*" if candidate_role == "admin" else "default",
                    "role": candidate_role,
                }
            )
        if len({item["key"] for item in principals}) != len(principals):
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "API principal keys must be unique across configured credentials"
                },
            )
        if path.startswith("/api/v1/") and path not in public_paths and principals:
            supplied = request.headers.get("authorization", "")
            principal = next(
                (
                    item
                    for item in principals
                    if hmac.compare_digest(supplied, f"Bearer {item['key']}")
                ),
                None,
            )
            if principal is None:
                security_log.warning(
                    "api_authentication_failed",
                    method=request.method,
                    path=path,
                    client_host=request.client.host if request.client else None,
                )
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Valid bearer API key required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            role = principal["role"]
            request.state.eaktp_role = role
            request.state.eaktp_principal_id = principal["principal_id"]
            request.state.eaktp_home_tenant_id = principal["tenant_id"]
            request.state.eaktp_tenant_id = "*" if role == "admin" else principal["tenant_id"]
            if role == "reader" and request.method not in {"GET", "HEAD", "OPTIONS"}:
                security_log.warning(
                    "api_authorization_denied",
                    reason="reader_write_attempt",
                    principal_id=principal["principal_id"],
                    role=role,
                    method=request.method,
                    path=path,
                )
                return JSONResponse(
                    status_code=403, content={"detail": "Reader access is read-only"}
                )
            admin_only = (
                path.startswith("/api/v1/training/")
                or path == "/api/v1/agent"
                or path.startswith("/api/v1/evaluations")
            )
            if admin_only and role != "admin":
                security_log.warning(
                    "api_authorization_denied",
                    reason="administrator_required",
                    principal_id=principal["principal_id"],
                    role=role,
                    method=request.method,
                    path=path,
                )
                return JSONResponse(
                    status_code=403, content={"detail": "Administrator access required"}
                )
        response = await call_next(request)
        principal_id = getattr(request.state, "eaktp_principal_id", None)
        if principal_id is not None and path.startswith("/api/v1/") and path not in public_paths:
            security_log.info(
                "api_request_completed",
                principal_id=principal_id,
                role=getattr(request.state, "eaktp_role", "unknown"),
                tenant_id=getattr(request.state, "eaktp_home_tenant_id", None),
                method=request.method,
                path=path,
                status_code=response.status_code,
            )
        return response

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
