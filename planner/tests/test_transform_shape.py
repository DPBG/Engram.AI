"""Planner TRANSFORM must execute Kernel bare action dicts, not empty action={}."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from planner.service import PlannerService


def _run(coro):
    return asyncio.run(coro)


def test_handle_kernel_decision_transform_wraps_bare_action():
    svc = PlannerService.__new__(PlannerService)
    svc.logger = logging.getLogger("test-planner-transform")
    executed: list[dict[str, Any]] = []

    async def _capture(proposal: dict[str, Any]) -> None:
        executed.append(proposal)

    svc._execute_action = _capture

    proposal = {
        "trace_id": "t-xform",
        "provenance": "planner",
        "action": {"channel": "manipulation", "intensity": 0.95, "type": "motor_command"},
    }
    # KernelDecision as used by planner (simple namespace / duck type)
    decision = type(
        "KD",
        (),
        {
            "trace_id": "t-xform",
            "type": "TRANSFORM",
            "reason": "profile clamp",
            "transformations": [
                {"channel": "manipulation", "intensity": 0.4, "type": "motor_command"}
            ],
            "risk_score": 0.3,
        },
    )()

    _run(svc._handle_kernel_decision(decision, proposal))

    assert len(executed) == 1
    assert executed[0]["trace_id"] == "t-xform"
    assert executed[0]["action"]["intensity"] == 0.4
    assert executed[0]["action"]["channel"] == "manipulation"
