"""Structured logging for EventBus connection lifecycle (issue #256)."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

CONNECTION_EVENT = "eventbus.connection"

# Stable transition identifiers for alerting pipelines.
TRANSITION_ALREADY_CONNECTED = "already_connected"
TRANSITION_CONNECTING = "connecting"
TRANSITION_CONNECTED = "connected"
TRANSITION_DISCONNECTED = "disconnected"
TRANSITION_RECONNECTED = "reconnected"
TRANSITION_CLOSING = "closing"
TRANSITION_CLOSED = "closed"
TRANSITION_FORCE_RECONNECT_START = "force_reconnect_start"
TRANSITION_FORCE_RECONNECT_CLOSE_FAILED = "force_reconnect_close_failed"
TRANSITION_FORCE_RECONNECT_RESUBSCRIBE_FAILED = "force_reconnect_resubscribe_failed"
TRANSITION_FORCE_RECONNECT_COMPLETE = "force_reconnect_complete"


def log_connection_event(
    transition: str,
    *,
    client_name: str,
    nats_url: str,
    level: int = logging.INFO,
    **fields: Any,
) -> dict[str, Any]:
    """Emit a single-line JSON connection-lifecycle event.

    Returns the payload so tests can assert on structure without parsing logs.
    """
    payload: dict[str, Any] = {
        "event": CONNECTION_EVENT,
        "transition": transition,
        "client_name": client_name,
        "nats_url": nats_url,
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    logger.log(level, "%s", json.dumps(payload, default=str))
    return payload
