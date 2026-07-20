"""Code-proposal happy-path publish must match EventBus.publish arity.

A leftover third positional argument (duplicate signed payload) raised
TypeError against the real EventBus, which the fail-safe handler converted
into DENY — flipping every successful ALLOW/TRANSFORM/DEFER.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import Any

from activelearning import RiskAnalysis
from activelearning.subjects import code_decision_subject

from kernel.evaluator import DecisionType, KernelDecision
from kernel.service import KernelService


class _RealisticBus:
    """EventBus stand-in that enforces the real publish signature."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []

    async def publish(
        self,
        subject: str,
        data: dict[str, Any],
        *,
        message_model: Any = None,
    ) -> None:
        self.published.append((subject, data))


def _make_service() -> KernelService:
    svc = KernelService.__new__(KernelService)
    svc.logger = logging.getLogger("test-code-publish-arity")
    svc._deny_count = 0
    svc._allow_count = 0
    svc._transform_count = 0
    svc._defer_count = 0
    svc._proposal_timestamps = defaultdict(deque)
    svc._proposal_rate_limit = 100
    svc._proposal_rate_window_s = 60.0
    svc.event_bus = _RealisticBus()

    async def _no_log(*args, **kwargs):
        return None

    async def _low_risk(*args, **kwargs):
        return RiskAnalysis(trace_id="t", risk_score=0.0, flags=[])

    svc._log_decision = _no_log
    svc._get_risk_analysis = _low_risk
    svc._update_metrics = lambda decision_type: None
    return svc


def test_code_proposal_allow_publishes_allow_not_failsafe_deny():
    """Happy-path ALLOW must reach the bus; must not flip to internal-error DENY."""
    svc = _make_service()

    def _allow(proposal, risk_analysis=None):
        return KernelDecision(
            trace_id=proposal.get("trace_id", ""),
            type=DecisionType.ALLOW,
            reason="safe",
            risk_score=0.0,
        )

    svc._evaluator = type("Ev", (), {"evaluate_code_proposal": staticmethod(_allow)})()

    asyncio.run(
        svc._handle_code_proposal(
            {
                "trace_id": "t-allow",
                "source": "meta-programmer",
                "target_path": "/data/plugins/ok.py",
                "code_preview": "x = 1",
            }
        )
    )

    assert len(svc.event_bus.published) == 1
    subject, payload = svc.event_bus.published[0]
    assert subject == code_decision_subject("t-allow")
    assert payload["type"] == DecisionType.ALLOW.value
    assert "internal error" not in (payload.get("reason") or "").lower()
    assert svc._deny_count == 0
