"""Deterministic ingestion and privacy tests."""

import pytest
from app.ingestion.parsers import parse_document
from app.ingestion.validation import validate_upload
from app.privacy.service import detect_pii, redact_pii


def test_upload_validation_rejects_path_traversal_and_oversize() -> None:
    with pytest.raises(ValueError, match="safe basename"):
        validate_upload("../secret.txt", b"content", 100)
    with pytest.raises(ValueError, match="size limit"):
        validate_upload("notes.txt", b"12345", 4)


def test_plain_text_parsing_normalizes_supported_content() -> None:
    extension = validate_upload("notes.txt", b"\xef\xbb\xbfPolicy text", 100)

    document = parse_document("notes.txt", b"\xef\xbb\xbfPolicy text", extension)

    assert document.text == "Policy text"
    assert document.media_type == ".txt"


def test_pii_is_detected_and_redacted_without_raw_values_in_sanitized_text() -> None:
    text = "Contact Alice at alice@example.com or +1 (555) 123-4567."

    detection = detect_pii(text)
    redaction = redact_pii(text, detection)

    assert detection.counts == {"EMAIL": 1, "PHONE": 1}
    assert "alice@example.com" not in redaction.text
    assert "[EMAIL_1]" in redaction.text
    assert "[PHONE_1]" in redaction.text


def test_empty_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_upload("empty.md", b"", 100)
