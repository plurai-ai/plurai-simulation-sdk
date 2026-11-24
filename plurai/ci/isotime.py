"""The one reader of the ISO-8601 instants the backend and the saved login record (``expiresAt``, ``expires_at``)."""

from __future__ import annotations

from datetime import UTC, datetime


def parse_instant(value: object) -> datetime | None:
    """``value`` as an aware instant, or ``None`` when it is no ISO-8601 text. A time with no offset is UTC, the zone the
    backend writes in; ``Z`` and any offset are read as written."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
