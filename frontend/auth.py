"""Frontend credential selection and validation."""

import hmac
import json

from app.core.config import Settings


class FrontendCredentialError(RuntimeError):
    """The frontend cannot identify a valid configured service credential."""


def resolve_frontend_api_key(settings: Settings) -> str | None:
    """Select the explicit UI credential without bypassing backend authorization."""

    ui_key = settings.ui_api_key.get_secret_value().strip() if settings.ui_api_key else ""
    if settings.api_principals_json is not None:
        if not ui_key:
            raise FrontendCredentialError(
                "EAKTP_UI_API_KEY is required when EAKTP_API_PRINCIPALS_JSON is configured. "
                "Set it to the intended tenant-scoped principal key and restart Streamlit."
            )
        try:
            principals = json.loads(settings.api_principals_json.get_secret_value())
        except json.JSONDecodeError as exc:
            raise FrontendCredentialError(
                "EAKTP_API_PRINCIPALS_JSON is invalid; fix .env and restart both services."
            ) from exc
        if not isinstance(principals, list) or not any(
            isinstance(item, dict)
            and isinstance(item.get("key"), str)
            and hmac.compare_digest(item["key"], ui_key)
            for item in principals
        ):
            raise FrontendCredentialError(
                "EAKTP_UI_API_KEY does not match a configured API principal. "
                "Use the exact intended principal key and restart both services."
            )
        return ui_key

    if ui_key:
        return ui_key
    for configured in (settings.write_api_key, settings.read_api_key):
        if configured is not None and configured.get_secret_value().strip():
            return configured.get_secret_value()
    return None
