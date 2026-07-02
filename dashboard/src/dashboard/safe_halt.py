"""SAFE_HALT state management and payload helpers (Phase E1.9.6).

Pure stdlib — no FastAPI/NATS imports — so this module can be unit-tested
in the governance CI environment without the web stack.
"""

from __future__ import annotations

from typing import Any

#: Maximum length for sanitized reason and operator_id fields.
MAX_REASON_LEN: int = 200
MAX_OPERATOR_LEN: int = 64

# Current halt state — updated when safety.halt.status arrives from the Kernel.
_halt_state: dict[str, Any] = {"halted": False}


def get_halt_state() -> dict[str, Any]:
    """Return a copy of the current halt state."""
    return dict(_halt_state)


def update_halt_state(data: dict[str, Any]) -> None:
    """Overwrite the cached halt state with the latest Kernel broadcast."""
    global _halt_state
    _halt_state = dict(data)


def sanitize_halt_payload(payload: Any) -> tuple[bool, str, str]:
    """Validate and sanitize a ``safe_halt`` WebSocket payload.

    Returns ``(ok, reason, operator_id)``.  Fails closed (``ok=False``) when
    the payload is not a dict, so a malformed message cannot trigger a halt.
    """
    if not isinstance(payload, dict):
        return False, "", ""
    reason = str(payload.get("reason", "operator SAFE_HALT"))[:MAX_REASON_LEN].strip()
    operator_id = str(payload.get("operator_id", "dashboard"))[:MAX_OPERATOR_LEN].strip()
    if not reason:
        reason = "operator SAFE_HALT"
    if not operator_id:
        operator_id = "dashboard"
    return True, reason, operator_id


def sanitize_resume_payload(payload: Any) -> tuple[bool, str]:
    """Validate and sanitize a ``safe_resume`` WebSocket payload.

    Returns ``(ok, operator_id)``.  Fails closed (``ok=False``) on non-dict input.
    """
    if not isinstance(payload, dict):
        return False, ""
    operator_id = str(payload.get("operator_id", "dashboard"))[:MAX_OPERATOR_LEN].strip()
    if not operator_id:
        operator_id = "dashboard"
    return True, operator_id
