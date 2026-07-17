"""Dead-letter queue consumption and alerting (issue #246).

``EventBus._route_to_poison`` republishes exhausted/unprocessable messages to
``dlq.<subject>`` (see ``nats_client.poison_subject``), but until this module
nothing subscribed to that namespace — dead letters accumulated silently and
invisibly. ``parse_dlq_envelope``/``emit_dlq_alert`` turn a raw DLQ envelope
into a structured, operator-facing alert; ``EventBus.start_dlq_monitor()``
(in ``nats_client.py``) wires them to a ``dlq.>`` wildcard subscription.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DLQ_ALERT_EVENT = "dlq.dead_letter"


@dataclass(frozen=True)
class DlqRecord:
    """A parsed dead-letter envelope, ready to log, persist, or broadcast."""

    dlq_subject: str
    original_subject: str
    reason: str
    num_delivered: int
    payload: str


def parse_dlq_envelope(dlq_subject: str, data: dict[str, Any]) -> DlqRecord:
    """Build a DlqRecord from the envelope ``_route_to_poison`` publishes.

    Tolerant of missing/malformed fields — a malformed envelope must never
    crash the monitor, since that would silently drop dead-letter visibility
    right where the fix is supposed to restore it.
    """
    try:
        num_delivered = int(data.get("num_delivered", 0) or 0)
    except (TypeError, ValueError):
        num_delivered = 0
    return DlqRecord(
        dlq_subject=dlq_subject,
        original_subject=str(data.get("original_subject", "unknown")),
        reason=str(data.get("reason", "unknown")),
        num_delivered=num_delivered,
        payload=str(data.get("payload", "")),
    )


def emit_dlq_alert(record: DlqRecord, *, monitor: str) -> dict[str, Any]:
    """Emit a structured operator-facing dead-letter alert (JSON log + return payload)."""
    payload: dict[str, Any] = {
        "event": DLQ_ALERT_EVENT,
        "dlq_subject": record.dlq_subject,
        "original_subject": record.original_subject,
        "reason": record.reason,
        "num_delivered": record.num_delivered,
        "monitor": monitor,
        "message": (
            f"Dead-lettered message on {record.original_subject!r} "
            f"(reason={record.reason!r}, delivered={record.num_delivered}x)"
        ),
    }
    logger.error("%s", json.dumps(payload, default=str))
    return payload
