"""Tests for KernelService decision publishing — fail-safe on internal error.

These avoid pytest-asyncio (not available in the Governance CI job) by driving
the async handlers through ``asyncio.run``, and bypass ``KernelService.__init__``
(which would open NATS/SQLite) via ``__new__`` + stubbed dependencies.
"""

import asyncio
import logging
from collections import deque

from activelearning import KernelDecisionType as DecisionType
from activelearning.subjects import code_decision_subject

from kernel.evaluator import unavailable_risk_analysis
from kernel.service import _LATENCY_WINDOW, KernelService
from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path

from activelearning import KernelDecisionType as DecisionType
from activelearning.nats_client import EventBus
from activelearning.subjects import code_decision_subject

from kernel.evaluator import unavailable_risk_analysis
from kernel.service import KernelService

_KERNEL_SRC = Path(__file__).resolve().parents[1] / "src" / "kernel"
_GATE_PATH = Path(__file__).resolve().parents[2] / "coordinator" / "src" / "coordinator" / "gate.py"


class _FakeBus:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload):
        self.published.append((subject, payload))


class _RaisingEvaluator:
    def evaluate_code_proposal(self, *args, **kwargs):
        raise RuntimeError("boom — simulated internal kernel error")


def _make_service():
    def evaluate_action_proposal(self, *args, **kwargs):
        raise RuntimeError("boom — simulated internal kernel error")


class _FailingPublishBus:
    """Simulates a broker outage on every publish attempt."""

    def __init__(self):
        self.attempts = 0

    async def publish(self, subject, payload):
        self.attempts += 1
        raise RuntimeError("NATS publish failed — simulated broker outage")


def _load_coordinator_gate():
    spec = importlib.util.spec_from_file_location("coord_gate", _GATE_PATH)
    gate = importlib.util.module_from_spec(spec)
    sys.modules["coord_gate_test_service"] = gate
    spec.loader.exec_module(gate)
    return gate


def _caller_decision_on_timeout(trace_id: str, nats_server: str, *, code: bool = False) -> dict:
    """Mirror coordinator / meta-programmer / overrides fail-closed timeout handling."""

    async def _run() -> dict:
        bus = EventBus(nats_url=nats_server, name=f"waiter-{trace_id}")
        await bus.connect()
        try:
            try:
                await bus.wait_for_decision(trace_id, timeout=0.5, code=code)
            except TimeoutError:
                return {"type": "DENY", "reason": "Decision timeout"}
            raise AssertionError("wait_for_decision returned without a decision — fail-open")
        finally:
            await bus.close()

    return asyncio.run(_run())


def _make_service(*, publish_raises: bool = False):
    """A KernelService with __init__ bypassed and only the bits the code-proposal
    handler touches stubbed in."""
    svc = KernelService.__new__(KernelService)
    svc.logger = logging.getLogger("test-kernel")
    svc._deny_count = 0
    svc._evaluator = _RaisingEvaluator()
    svc.event_bus = _FakeBus()
    svc._latency_samples = deque(maxlen=_LATENCY_WINDOW)
    svc._SLO_P99_MS = 50.0
    svc._slo_breach_count = 0
    svc._last_slo_breach_at = 0.0
    svc.event_bus = _FailingPublishBus() if publish_raises else _FakeBus()

    async def _no_risk(*args, **kwargs):
        return None

    async def _no_log(*args, **kwargs):
        return None

    svc._get_risk_analysis = _no_risk
    svc._log_decision = _no_log
    return svc


    async def _no_norms(*args, **kwargs):
        return []

    svc._get_risk_analysis = _no_risk
    svc._log_decision = _no_log
    svc._check_belief_norms = _no_norms
    if not publish_raises:
        return svc

    svc._publish_and_log_decision = KernelService._publish_and_log_decision.__get__(
        svc,
        KernelService,
    )
    return svc


def _make_action_service(*, publish_raises: bool = False):
    return _make_service(publish_raises=publish_raises)


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


def test_action_proposal_recovery_publish_failure_caller_fails_closed(nats_server: str):
    # Issue #208: if the fail-safe DENY publish itself raises, the handler
    # swallows it — but callers must still fail closed via wait_for_decision
    # timing out (coordinator / meta-programmer / overrides pattern).
    trace_id = "t-action-recovery-fail"
    svc = _make_action_service(publish_raises=True)

    asyncio.run(
        svc._handle_action_proposal(
            {
                "trace_id": trace_id,
                "provenance": "test",
                "action": {"type": "motor", "channel": "head"},
            }
        )
    )

    assert svc.event_bus.attempts >= 1, "recovery path never attempted DENY publish"
    decision = _caller_decision_on_timeout(trace_id, nats_server, code=False)
    assert decision["type"] == "DENY"
    gate = _load_coordinator_gate()
    assert gate.decision_allows(decision) is False


def test_code_proposal_recovery_publish_failure_caller_fails_closed(nats_server: str):
    trace_id = "t-code-recovery-fail"
    svc = _make_service(publish_raises=True)

    asyncio.run(svc._handle_code_proposal({"trace_id": trace_id, "source": "meta-programmer"}))

    assert svc.event_bus.attempts >= 1, "recovery path never attempted DENY publish"
    decision = _caller_decision_on_timeout(trace_id, nats_server, code=True)
    assert decision["type"] == "DENY"
    gate = _load_coordinator_gate()
    assert gate.decision_allows(decision) is False


def test_kernel_bare_except_pass_audit_issue_208():
    """Grep audit (issue #208): no new silent fail-open handlers in kernel/."""
    marker = "pass  # caller will timeout"
    hits: list[tuple[Path, int]] = []
    for path in _KERNEL_SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if marker in line:
                hits.append((path, lineno))

    assert len(hits) == 2, f"expected exactly 2 documented recovery blocks, found {hits}"
    for path, _lineno in hits:
        assert path.name == "service.py"


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
    return asyncio.run(svc._get_risk_analysis(proposal or {"trace_id": "t1"}))


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
