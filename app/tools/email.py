"""Email draft tool that never sends messages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmailDraft:
    """Validated, unsent email content."""

    recipient: str
    subject: str
    body: str
    requires_approval: bool = True


def create_email_draft(recipient: str, subject: str, body: str) -> EmailDraft:
    """Create a draft and explicitly require human approval."""

    if not recipient.strip() or "@" not in recipient:
        raise ValueError("recipient must be a valid email address")
    if not subject.strip() or not body.strip():
        raise ValueError("subject and body are required")
    return EmailDraft(recipient=recipient.strip(), subject=subject.strip(), body=body.strip())

