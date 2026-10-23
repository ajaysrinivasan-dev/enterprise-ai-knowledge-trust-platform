"""Frontend credential selection and validation."""

import hmac
import json

from app.core.config import Settings


class FrontendCredentialError(RuntimeError):
    """The frontend cannot identify a valid configured service credential."""


def _registry(settings: Settings) -> list[dict[str, object]] | None:
    """Read the same principal registry used by the API, without exposing keys."""
    if settings.api_principals_json is None:
        return None
    try:
        principals = json.loads(settings.api_principals_json.get_secret_value())
    except json.JSONDecodeError as exc:
        raise FrontendCredentialError(
            "EAKTP_API_PRINCIPALS_JSON is invalid; fix .env and restart both services."
        ) from exc
    if not isinstance(principals, list):
        raise FrontendCredentialError("EAKTP_API_PRINCIPALS_JSON must contain a JSON list.")
    return [item for item in principals if isinstance(item, dict)]


def _find_principal(
    principals: list[dict[str, object]], key: str
) -> dict[str, object] | None:
    return next(
        (
            item for item in principals
            if isinstance(item.get("key"), str)
            and hmac.compare_digest(str(item["key"]), key)
        ),
        None,
    )


def resolve_frontend_api_key(settings: Settings) -> str | None:
    """Select a tenant-scoped UI credential; never elevate an ordinary key."""
    ui_key = settings.ui_api_key.get_secret_value().strip() if settings.ui_api_key else ""
    principals = _registry(settings)
    if principals is not None:
        if not ui_key:
            raise FrontendCredentialError(
                "EAKTP_UI_API_KEY is required when EAKTP_API_PRINCIPALS_JSON is configured. "
                "Set it to the intended tenant-scoped principal key and restart Streamlit."
            )
        match = _find_principal(principals, ui_key)
        if match is None:
            raise FrontendCredentialError(
                "EAKTP_UI_API_KEY does not match a configured API principal."
            )
        if match.get("role") not in {"reader", "writer"}:
            raise FrontendCredentialError(
                "EAKTP_UI_API_KEY must be a reader or writer key. Configure admin actions "
                "separately with EAKTP_ADMIN_UI_API_KEY."
            )
        return ui_key
    if ui_key:
        return ui_key
    for configured in (settings.write_api_key, settings.read_api_key):
        if configured is not None and configured.get_secret_value().strip():
            return configured.get_secret_value()
    return None


def resolve_frontend_admin_api_key(settings: Settings) -> str | None:
    """Resolve a distinct admin credential for evaluation, training, and agent operations."""
    key = settings.admin_ui_api_key.get_secret_value().strip() if settings.admin_ui_api_key else ""
    if not key:
        # Admin workflows remain unavailable unless a separate credential is explicitly set.
        return None
    registry_configured = (
        settings.api_principals_json is not None
        and bool(settings.api_principals_json.get_secret_value().strip())
    )
    if registry_configured:
        principals = _registry(settings)
        match = _find_principal(principals or [], key)
        if match is None:
            raise FrontendCredentialError(
                "EAKTP_ADMIN_UI_API_KEY does not match a configured API principal."
            )
        if match.get("role") != "admin":
            raise FrontendCredentialError(
                "EAKTP_ADMIN_UI_API_KEY must match an administrator principal; "
                "writer keys are not allowed."
            )
    else:
        legacy_admin = settings.api_key.get_secret_value().strip() if settings.api_key else ""
        if not legacy_admin or not hmac.compare_digest(legacy_admin, key):
            raise FrontendCredentialError(
                "EAKTP_ADMIN_UI_API_KEY must match the configured "
                "EAKTP_API_KEY administrator credential."
            )
    return key
