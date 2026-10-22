"""Ingestion domain models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedDocument:
    """Normalized document content ready for privacy processing."""

    filename: str
    media_type: str
    text: str
    page_count: int | None = None


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def safe_filename(filename: str) -> str:
    """Return a basename-only filename and reject empty or suspicious names."""

    candidate = Path(filename).name
    if (
        not candidate
        or candidate in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or candidate != filename
    ):
        raise ValueError("filename must be a safe basename")
    return candidate
