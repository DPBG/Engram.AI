"""Structured operator alerts for launcher supervisor events (issue #261)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("launcher.supervisor")

SERVICE_FLAP_EVENT = "supervisor.service_flap"

_DEFAULT_FLAP_THRESHOLD = 3
_DEFAULT_FLAP_WINDOW_S = 60.0


def flap_threshold() -> int:
    """Restarts within the flap window before an operator alert fires."""
    raw = os.environ.get("SUPERVISOR_FLAP_THRESHOLD", str(_DEFAULT_FLAP_THRESHOLD))
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_FLAP_THRESHOLD
    return max(1, value)


def flap_window_seconds() -> float:
    """Rolling window (seconds) used to count restart flaps."""
    raw = os.environ.get("SUPERVISOR_FLAP_WINDOW_S", str(_DEFAULT_FLAP_WINDOW_S))
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_FLAP_WINDOW_S
    return max(1.0, value)


def record_restart(
    restart_times: list[float],
    *,
    now: float,
    window_seconds: float,
) -> list[float]:
    """Append a restart timestamp and drop entries outside the rolling window."""
    restart_times.append(now)
    cutoff = now - window_seconds
    kept = [t for t in restart_times if t >= cutoff]
    restart_times[:] = kept
    return kept


def should_emit_flap_alert(
    restarts_in_window: int,
    threshold: int,
    alerts_emitted: int,
) -> bool:
    """Return True when a new flap alert should fire (threshold, 2x, 3x, …)."""
    if restarts_in_window < threshold:
        return False
    return restarts_in_window // threshold > alerts_emitted


def emit_service_flap_alert(
    *,
    service_name: str,
    restart_count: int,
    restarts_in_window: int,
    window_seconds: float,
    threshold: int,
    exit_code: int,
    uptime_seconds: float,
) -> dict[str, Any]:
    """Emit a structured operator-facing flap alert (JSON log + return payload)."""
    payload: dict[str, Any] = {
        "event": SERVICE_FLAP_EVENT,
        "service": service_name,
        "restart_count": restart_count,
        "restarts_in_window": restarts_in_window,
        "window_seconds": window_seconds,
        "threshold": threshold,
        "exit_code": exit_code,
        "uptime_seconds": round(uptime_seconds, 3),
        "message": (
            f"Service {service_name!r} restarted {restarts_in_window} times in "
            f"{window_seconds:.0f}s (threshold={threshold}) — possible crash loop"
        ),
    }
    logger.warning("%s", json.dumps(payload, default=str))
    return payload
