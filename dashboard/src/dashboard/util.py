"""Small shared helpers for the dashboard (no web-stack dependency)."""

from datetime import datetime, timezone


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (used in API/WS payloads)."""
    return datetime.now(timezone.utc).isoformat()
