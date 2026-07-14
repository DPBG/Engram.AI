"""JetStream durable consumer lag alerting (issue #224)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CONSUMER_LAG_EVENT = "jetstream.consumer_lag"

_DEFAULT_LAG_THRESHOLD = 10
_DEFAULT_LAG_CHECK_INTERVAL_S = 30.0

# Ephemeral decision-waiter durables are short-lived; skip them for lag alerts.
_EPHEMERAL_CONSUMER_PREFIXES = ("waiter-action-", "waiter-code-")


@dataclass(frozen=True)
class ConsumerLagSnapshot:
    """Pending/ack lag for one durable JetStream consumer."""

    stream: str
    consumer: str
    num_pending: int
    num_ack_pending: int
    filter_subject: str | None = None

    @property
    def lag(self) -> int:
        return max(self.num_pending, self.num_ack_pending)


def lag_threshold() -> int:
    """Messages (pending or unacked) before a lag alert fires."""
    raw = os.environ.get("JETSTREAM_LAG_THRESHOLD", str(_DEFAULT_LAG_THRESHOLD))
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_LAG_THRESHOLD
    return max(1, value)


def lag_check_interval_seconds() -> float:
    """How often the background lag monitor polls consumer_info."""
    raw = os.environ.get("JETSTREAM_LAG_CHECK_INTERVAL_S", str(_DEFAULT_LAG_CHECK_INTERVAL_S))
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_LAG_CHECK_INTERVAL_S
    return max(5.0, value)


def should_monitor_consumer(consumer_name: str) -> bool:
    """Return False for ephemeral waiter durables that are not operator-facing."""
    return not consumer_name.startswith(_EPHEMERAL_CONSUMER_PREFIXES)


def should_emit_lag_alert(lag: int, threshold: int, alerts_emitted: int) -> bool:
    """Return True when lag crosses threshold, 2×, 3×, … since last clear."""
    if lag < threshold:
        return False
    return lag // threshold > alerts_emitted


def emit_consumer_lag_alert(
    snapshot: ConsumerLagSnapshot,
    *,
    threshold: int,
    monitor: str,
) -> dict[str, Any]:
    """Emit a structured operator-facing lag alert (JSON log + return payload)."""
    payload: dict[str, Any] = {
        "event": CONSUMER_LAG_EVENT,
        "stream": snapshot.stream,
        "consumer": snapshot.consumer,
        "num_pending": snapshot.num_pending,
        "num_ack_pending": snapshot.num_ack_pending,
        "lag": snapshot.lag,
        "threshold": threshold,
        "monitor": monitor,
        "message": (
            f"JetStream consumer {snapshot.consumer!r} on {snapshot.stream!r} "
            f"has lag {snapshot.lag} (pending={snapshot.num_pending}, "
            f"ack_pending={snapshot.num_ack_pending}, threshold={threshold})"
        ),
    }
    if snapshot.filter_subject is not None:
        payload["filter_subject"] = snapshot.filter_subject
    logger.warning("%s", json.dumps(payload, default=str))
    return payload
