"""Upload validation independent of any web framework."""

from pathlib import Path

from app.ingestion.models import SUPPORTED_EXTENSIONS, safe_filename


def validate_upload(filename: str, content: bytes, max_bytes: int) -> str:
    """Validate filename, size, and non-empty content; return normalized extension."""

    safe_filename(filename)
    if len(content) == 0:
        raise ValueError("document is empty")
    if len(content) > max_bytes:
        raise ValueError("document exceeds the configured size limit")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported document type: {extension or 'none'}")
    return extension

