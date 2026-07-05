"""Tests for KernelService decision publishing — fail-safe on internal error.

These avoid pytest-asyncio (not available in the Governance CI job) by driving
the async handlers through ``asyncio.run``, and bypass ``KernelService.__init__``
(which would open NATS/SQLite) via ``__new__`` + stubbed dependencies.
"""

import asyncio
import logging

from activelearning import KernelDecisionType as DecisionType
from activelearning.subjects import code_decision_subject
from kernel.evaluator import unavailable_risk_analysis
from kernel.service import KernelService


class _FakeBus:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload):
        self.published.append((subject, payload))


class _RaisingEvaluator:
    def evaluate_code_proposal(self, *args, **kwargs):
        raise RuntimeError("boom — simulated internal kernel error")


def _make_service():
    """A KernelService with __init__ bypassed and only the bits the code-proposal
    handler touches stubbed in."""
    svc = KernelService.__new__(KernelService)
    svc.logger = logging.getLogger("test-kernel")
    svc._deny_count = 0
    svc._evaluator = _RaisingEvaluator()
    svc.event_bus = _FakeBus()

    async def _no_risk(*args, **kwargs):
        return None

    async def _no_log(*args, **kwargs):
        return None

    svc._get_risk_analysis = _no_risk
    svc._log_decision = _no_log
    return svc


def test_code_proposal_publishes_fail_safe_deny_on_internal_error():
    # When evaluating a code proposal raises, the Kernel must still publish a
    # decision (it is the sole decision authority and must fail closed) — a DENY
    # on the code-decision subject — instead of silently swallowing the error.
    svc = _make_service()

    asyncio.run(svc._handle_code_proposal({"trace_id": "t1", "source": "meta-programmer"}))

    assert len(svc.event_bus.published) == 1, "no decision published — fail-open"
    subject, payload = svc.event_bus.published[0]
    assert subject == code_decision_subject("t1")
    assert payload["type"] == DecisionType.DENY.value
    assert payload["trace_id"] == "t1"
    assert payload["risk_score"] == 1.0
    assert svc._deny_count == 1


class _RequestBus:
    def __init__(self, response):
        self._response = response

    async def request(self, subject, payload, timeout=5.0):
        return self._response


def _make_risk_service(response):
    svc = KernelService.__new__(KernelService)
    svc.logger = logging.getLogger("test-kernel")
    svc.event_bus = _RequestBus(response)
    return svc


def _run_risk_analysis(response, proposal=None):
    svc = _make_risk_service(response)
    return asyncio.run(
        svc._get_risk_analysis(proposal or {"trace_id": "t1"})
    )


def test_malformed_nan_risk_score_fails_closed():
    result = _run_risk_analysis({"type": "analysis", "risk_score": float("nan"), "flags": []})
    assert result.risk_score == 1.0
    assert unavailable_risk_analysis().flags == result.flags


def test_missing_risk_score_fails_closed():
    result = _run_risk_analysis({"type": "analysis", "flags": []})
    assert result.risk_score == 1.0
    assert unavailable_risk_analysis().flags == result.flags


def test_non_numeric_risk_score_fails_closed():
    result = _run_risk_analysis({"type": "analysis", "risk_score": "high", "flags": []})
    assert result.risk_score == 1.0
    assert unavailable_risk_analysis().flags == result.flags


def test_non_list_flags_fails_closed():
    result = _run_risk_analysis({"type": "analysis", "risk_score": 0.1, "flags": "oops"})
    assert result.risk_score == 1.0
    assert unavailable_risk_analysis().flags == result.flags


def test_valid_risk_analysis_parsed():
    result = _run_risk_analysis({"type": "analysis", "risk_score": 0.2, "flags": ["OK"]})
    assert result.risk_score == 0.2
    assert result.flags == ["OK"]
