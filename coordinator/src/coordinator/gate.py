"""
Kernel gate for Coordinator task execution (Phase 1.6).

The Coordinator used to execute learned tasks directly, bypassing the Kernel —
so a task could run without ever passing the safety gate. This module routes
every execution through the Kernel: the Coordinator publishes an action
proposal and only proceeds on an ALLOW/TRANSFORM decision.

The decision check is **fail-closed**: anything that is not an explicit
ALLOW/TRANSFORM (a DENY, a DEFER, a timeout that yields ``None``/``{}``, or a
malformed payload) means "do not execute." Decision authenticity (HMAC
signature) is enforced upstream by ``EventBus.wait_for_decision``; a forged
decision never reaches this check.
"""

from __future__ import annotations

from typing import Any, Optional

#: Subject the Kernel listens on for action proposals.
KERNEL_PROPOSAL_SUBJECT = "proposal.new"

#: Decision types that permit execution. Everything else is denied.
APPROVED_TYPES = frozenset({"ALLOW", "TRANSFORM"})


def build_execution_proposal(
    trace_id: str,
    task_id: str,
    parameters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the action proposal the Kernel evaluates before a task runs."""
    return {
        "trace_id": trace_id,
        "provenance": "coordinator",
        "action": {
            "type": "task_execution",
            "task_id": task_id,
            "parameters": parameters or {},
        },
    }


def decision_allows(decision: Optional[dict[str, Any]]) -> bool:
    """Return ``True`` only for an explicit ALLOW/TRANSFORM decision.

    Fail-closed: ``None`` (timeout), ``{}``, missing/unknown ``type``, DENY, and
    DEFER all return ``False`` so the Coordinator declines to execute.
    """
    if not isinstance(decision, dict):
        return False
    return decision.get("type") in APPROVED_TYPES
