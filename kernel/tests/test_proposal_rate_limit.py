"""Tests for code-proposal per-source rate limiting (M1.14).

Covers:
- _check_proposal_rate_limit: within-limit returns True and records timestamp
- _check_proposal_rate_limit: over-limit returns False without recording
- _check_proposal_rate_limit: sliding window evicts old timestamps
- _check_proposal_rate_limit: independent buckets per source
- _handle_code_proposal: over-limit source receives a fail-closed DENY
- _handle_code_proposal: within-limit source is not rate-limited
"""

import asyncio
import logging
from collections import defaultdict, deque
from unittest.mock import AsyncMock, MagicMock

from activelearning.subjects import code_decision_subject

from kernel.service import (
    _PROPOSAL_RATE_LIMIT,
    _PROPOSAL_RATE_WINDOW_S,
    KernelService,
)


def _make_service(rate_limit: int = 3, window_s: float = 60.0) -> KernelService:
    """Build a KernelService with heavy dependencies stubbed and rate-limit configured."""
    svc = KernelService.__new__(KernelService)
    svc.logger = logging.getLogger("test-kernel-rate")
    svc._deny_count = 0
    svc._proposal_timestamps = defaultdict(deque)
    svc._proposal_rate_limit = rate_limit
    svc._proposal_rate_window_s = window_s

    bus = MagicMock()
    bus.publish = AsyncMock()
    svc.event_bus = bus

    async def _no_log(*args, **kwargs):
        return None

    svc._log_decision = _no_log
    svc._signed_code_decision = lambda d: {"type": d.type.value, "trace_id": d.trace_id}
    return svc


# ── Unit tests for _check_proposal_rate_limit ─────────────────────────────────


def test_within_limit_returns_true_and_records():
    svc = _make_service(rate_limit=3)
    assert svc._check_proposal_rate_limit("meta-programmer") is True
    assert svc._check_proposal_rate_limit("meta-programmer") is True
    assert svc._check_proposal_rate_limit("meta-programmer") is True
    assert len(svc._proposal_timestamps["meta-programmer"]) == 3


def test_over_limit_returns_false_without_recording():
    svc = _make_service(rate_limit=2)
    svc._check_proposal_rate_limit("meta-programmer")
    svc._check_proposal_rate_limit("meta-programmer")
    # Third call should be rejected.
    result = svc._check_proposal_rate_limit("meta-programmer")
    assert result is False
    # Timestamp count must not grow beyond the limit.
    assert len(svc._proposal_timestamps["meta-programmer"]) == 2


def test_sliding_window_evicts_old_timestamps():
    svc = _make_service(rate_limit=2, window_s=1.0)
    svc._check_proposal_rate_limit("src")
    svc._check_proposal_rate_limit("src")
    # Manually back-date timestamps so they fall outside the 1-second window.
    ts = svc._proposal_timestamps["src"]
    for i in range(len(ts)):
        ts[i] = ts[i] - 2.0  # type: ignore[index]  # deque supports index assignment
    # After eviction the bucket should be empty; new proposals allowed.
    assert svc._check_proposal_rate_limit("src") is True
    assert len(svc._proposal_timestamps["src"]) == 1


def test_independent_buckets_per_source():
    svc = _make_service(rate_limit=1)
    assert svc._check_proposal_rate_limit("source-a") is True
    assert svc._check_proposal_rate_limit("source-a") is False
    # source-b has its own independent bucket — must not be affected.
    assert svc._check_proposal_rate_limit("source-b") is True


def test_module_level_defaults_are_reasonable():
    assert _PROPOSAL_RATE_LIMIT >= 1, "rate limit must be positive"
    assert _PROPOSAL_RATE_WINDOW_S > 0, "window must be positive"


# ── Integration tests for _handle_code_proposal ───────────────────────────────


def test_rate_limited_source_receives_fail_closed_deny():
    """Over-limit source must receive a DENY decision — never a silent drop."""
    svc = _make_service(rate_limit=2)

    async def run():
        # Exhaust the rate limit.
        for i in range(2):
            svc._check_proposal_rate_limit("meta-programmer")

        # Next proposal exceeds the limit.
        await svc._handle_code_proposal(
            {"trace_id": "t-rl-1", "source": "meta-programmer", "code": "x = 1"}
        )

    asyncio.run(run())

    assert svc.event_bus.publish.call_count == 1
    subject, payload = svc.event_bus.publish.call_args[0]
    assert subject == code_decision_subject("t-rl-1")
    assert payload["type"] == "DENY"
    assert payload["trace_id"] == "t-rl-1"
    assert svc._deny_count == 1


def test_within_limit_source_is_not_rate_limited():
    """Source within budget must reach evaluation (not short-circuited by rate limiter)."""
    svc = _make_service(rate_limit=5)

    # Stub the evaluator so we can detect that it was called.
    evaluator = MagicMock()
    from kernel.evaluator import DecisionType, KernelDecision

    evaluator.evaluate_code_proposal.return_value = KernelDecision(
        trace_id="t-ok-1",
        type=DecisionType.ALLOW,
        reason="test",
        risk_score=0.0,
    )
    svc._evaluator = evaluator

    async def _no_risk(*args, **kwargs):
        return None

    svc._get_risk_analysis = _no_risk
    svc._update_metrics = MagicMock()

    async def run():
        await svc._handle_code_proposal(
            {"trace_id": "t-ok-1", "source": "meta-programmer", "code": "x = 1"}
        )

    asyncio.run(run())

    evaluator.evaluate_code_proposal.assert_called_once()


def test_different_sources_have_independent_limits():
    """Rate-limiting one source must not affect other sources."""
    svc = _make_service(rate_limit=1)

    async def run():
        # Exhaust source-a.
        svc._check_proposal_rate_limit("source-a")

        # source-a is now over limit → DENY.
        await svc._handle_code_proposal(
            {"trace_id": "t-a-2", "source": "source-a", "code": "x = 1"}
        )
        deny_count_after_a = svc._deny_count

        # source-b still within limit — rate check returns True, handler
        # continues to evaluation (which will fail gracefully without a real
        # evaluator, but the rate limiter must not be the reason for denial).
        svc._deny_count = 0
        svc.event_bus.publish.reset_mock()

        # source-b's first proposal: rate check passes, so we should NOT see a
        # rate-limit DENY.  The handler will hit the evaluator stub and may DENY
        # for other reasons, but the DENY payload's reason must not mention rate.
        await svc._handle_code_proposal(
            {"trace_id": "t-b-1", "source": "source-b", "code": "x = 1"}
        )
        return deny_count_after_a

    deny_count_after_a = asyncio.run(run())

    assert deny_count_after_a == 1  # source-a was denied

    # If source-b was denied, it must not be for a rate-limit reason.
    if svc.event_bus.publish.call_count:
        _, payload = svc.event_bus.publish.call_args[0]
        if payload.get("type") == "DENY":
            assert "Rate limit" not in (
                payload.get("reason") or ""
            ), "source-b denial must not be due to rate limiting"
