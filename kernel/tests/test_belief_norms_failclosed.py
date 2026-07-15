"""Fail-closed when Beliefs norm checks are unavailable (mirrors PR #122 path)."""

from __future__ import annotations

import asyncio
import logging

from activelearning import RiskAnalysis

from kernel.evaluator import DecisionType, unavailable_belief_norm_violations
from kernel.service import KernelService


def _run(coro):
    return asyncio.run(coro)


def _make_service() -> KernelService:
    svc = KernelService.__new__(KernelService)
    svc.logger = logging.getLogger("test-kernel-beliefs")
    svc._deny_count = 0
    svc._evaluator = KernelService.__new__(KernelService)
    from kernel.evaluator import KernelEvaluator

    svc._evaluator = KernelEvaluator()
    return svc


def test_unavailable_belief_norm_violations_forces_deny():
    from kernel.evaluator import KernelEvaluator

    evaluator = KernelEvaluator()
    proposal = {
        "trace_id": "t-norms",
        "action": {"channel": "manipulation", "intensity": 0.95, "type": "motor_command"},
    }
    risk = RiskAnalysis(trace_id="t-norms", risk_score=0.65, flags=[])
    decision = evaluator.evaluate_action_proposal(
        proposal,
        risk_analysis=risk,
        norm_violations=unavailable_belief_norm_violations("timeout"),
    )
    assert decision.type == DecisionType.DENY


def test_check_belief_norms_fail_closed_on_handler_error():
    svc = _make_service()

    async def _error_response(*args, **kwargs):
        return {"result": None, "error": "db unavailable"}

    svc.event_bus = type("Bus", (), {"request": _error_response})()

    violations = _run(
        svc._check_belief_norms(
            {
                "trace_id": "t-err",
                "action": {"channel": "manipulation", "intensity": 0.95},
            }
        )
    )
    assert len(violations) == 1
    assert violations[0]["norm_id"] == "BELIEFS_UNAVAILABLE"
    assert violations[0]["risk_boost"] == 1.0


def test_action_proposal_denies_when_beliefs_unavailable():
    svc = _make_service()
    decisions: list = []

    async def _beliefs_timeout(*args, **kwargs):
        raise TimeoutError("beliefs request timed out")

    async def _low_risk(*args, **kwargs):
        return RiskAnalysis(trace_id="t-down", risk_score=0.65, flags=[])

    async def _capture_publish(
        trace_id, proposal_type, source, decision, **kwargs
    ):
        decisions.append(decision)

    svc.event_bus = type("Bus", (), {"request": _beliefs_timeout})()
    svc._get_risk_analysis = _low_risk
    svc._publish_and_log_decision = _capture_publish
    svc._update_metrics = lambda decision_type: None

    _run(
        svc._handle_action_proposal(
            {
                "trace_id": "t-down",
                "provenance": "neuromorphic",
                "action": {
                    "channel": "manipulation",
                    "intensity": 0.95,
                    "type": "motor_command",
                },
            }
        )
    )

    assert len(decisions) == 1
    assert decisions[0].type == DecisionType.DENY
    assert svc._deny_count == 1
