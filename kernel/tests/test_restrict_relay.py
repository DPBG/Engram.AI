"""Tests for kernel policy.restrict.request relay (E1.1.4 — ADR 0001 §3).

The kernel is the sole authorised publisher of ``policy.restrict``. The brain
and dashboard publish ``policy.restrict.request``; the Kernel validates and
re-publishes the authoritative ``policy.restrict`` that consumers act on.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from activelearning.subjects import Subjects

# ---------------------------------------------------------------------------
# Fixture: stub beliefs.profiles per-test so sys.modules is clean after each run.
# Module-level injection would leak into beliefs/tests in the same CI session and
# break real beliefs imports (MagicMock is not a package). monkeypatch restores
# sys.modules entries after every test automatically.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def beliefs_stub(monkeypatch):
    """Inject a stub beliefs.profiles for the kernel's lazy import, scoped to one test."""
    mock_profiles = MagicMock()
    monkeypatch.delitem(sys.modules, "beliefs", raising=False)
    monkeypatch.delitem(sys.modules, "beliefs.profiles", raising=False)
    monkeypatch.setitem(sys.modules, "beliefs", MagicMock(profiles=mock_profiles))
    monkeypatch.setitem(sys.modules, "beliefs.profiles", mock_profiles)
    return mock_profiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine synchronously (no pytest-asyncio dependency)."""
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_kernel_service():
    """Return a KernelService wired with a mock event bus and evaluator."""
    from kernel.service import KernelService

    svc = KernelService.__new__(KernelService)
    svc.logger = MagicMock()
    svc._body_profile = None

    mock_profile = MagicMock()
    mock_profile.name = "test_profile"
    mock_profile.motor_limits = {"locomotion": MagicMock(max_intensity=1.0)}
    mock_profile.capabilities = {}
    svc._evaluator = MagicMock()
    svc._evaluator._body_profile = mock_profile
    svc._evaluator.set_body_profile = MagicMock()

    published: list[tuple[str, dict]] = []

    async def _publish(subject, data, **kwargs):
        published.append((subject, data))

    svc.event_bus = AsyncMock()
    svc.event_bus.publish.side_effect = _publish
    svc._published = published

    return svc


def _restrict_payload(**overrides):
    base = {
        "motor_limits": {"locomotion": {"max_intensity": 0.5}},
        "reason": "test restriction",
        "operator_id": "system:test",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: _apply_restriction
# ---------------------------------------------------------------------------


def test_apply_restriction_updates_evaluator(beliefs_stub):
    svc = _make_kernel_service()
    restricted_profile = MagicMock()
    restricted_profile.name = "restricted"
    restricted_profile.motor_limits = {"locomotion": MagicMock(max_intensity=0.5)}
    restricted_profile.capabilities = {}
    beliefs_stub.apply_runtime_restrictions.return_value = restricted_profile

    result = _run(svc._apply_restriction(_restrict_payload()))

    assert result is True
    svc._evaluator.set_body_profile.assert_called_once_with(restricted_profile)


def test_apply_restriction_publishes_status_applied(beliefs_stub):
    svc = _make_kernel_service()
    beliefs_stub.apply_runtime_restrictions.return_value = svc._evaluator._body_profile

    _run(svc._apply_restriction(_restrict_payload()))

    status_publishes = [
        (subj, data) for subj, data in svc._published
        if subj == "policy.restrict.status"
    ]
    assert len(status_publishes) == 1
    assert status_publishes[0][1]["status"] == "applied"


def test_apply_restriction_returns_false_when_no_profile():
    svc = _make_kernel_service()
    svc._evaluator._body_profile = None

    result = _run(svc._apply_restriction(_restrict_payload()))

    assert result is False
    error_publishes = [
        d for s, d in svc._published
        if s == "policy.restrict.status" and d.get("status") == "error"
    ]
    assert len(error_publishes) == 1


# ---------------------------------------------------------------------------
# Tests: _handle_restrict_request
# ---------------------------------------------------------------------------


def test_restrict_request_broadcasts_policy_restrict(beliefs_stub):
    """Kernel must re-publish policy.restrict when it approves a request."""
    svc = _make_kernel_service()
    payload = _restrict_payload()
    beliefs_stub.apply_runtime_restrictions.return_value = svc._evaluator._body_profile

    _run(svc._handle_restrict_request(payload))

    policy_restrict_publishes = [
        (subj, data) for subj, data in svc._published
        if subj == Subjects.POLICY_RESTRICT
    ]
    assert len(policy_restrict_publishes) == 1, (
        "Kernel must broadcast policy.restrict after approving a restrict request"
    )
    assert policy_restrict_publishes[0][1] is payload


def test_restrict_request_does_not_broadcast_when_rejected(beliefs_stub):
    """If the restriction is rejected (ValueError), policy.restrict must NOT be published."""
    svc = _make_kernel_service()
    beliefs_stub.apply_runtime_restrictions.side_effect = ValueError("out of bounds")

    _run(svc._handle_restrict_request(_restrict_payload()))

    policy_restrict_publishes = [
        subj for subj, _ in svc._published if subj == Subjects.POLICY_RESTRICT
    ]
    assert len(policy_restrict_publishes) == 0, (
        "Kernel must NOT broadcast policy.restrict when the restriction is rejected"
    )


def test_restrict_request_does_not_broadcast_when_no_profile():
    """If no body profile is loaded, policy.restrict must NOT be published."""
    svc = _make_kernel_service()
    svc._evaluator._body_profile = None

    _run(svc._handle_restrict_request(_restrict_payload()))

    policy_restrict_publishes = [
        subj for subj, _ in svc._published if subj == Subjects.POLICY_RESTRICT
    ]
    assert len(policy_restrict_publishes) == 0


# ---------------------------------------------------------------------------
# Tests: SAFE_HALT broadcasts policy.restrict directly (no subscription round-trip)
# ---------------------------------------------------------------------------


def test_safe_halt_broadcasts_policy_restrict(beliefs_stub):
    """SAFE_HALT must publish policy.restrict to zero all motor channels."""
    svc = _make_kernel_service()
    svc._evaluator.halt = MagicMock()
    beliefs_stub.apply_runtime_restrictions.return_value = svc._evaluator._body_profile

    _run(svc._handle_safety_halt({"reason": "operator halt", "operator_id": "op1"}))

    policy_restrict_publishes = [
        (subj, data) for subj, data in svc._published
        if subj == Subjects.POLICY_RESTRICT
    ]
    assert len(policy_restrict_publishes) == 1
    motor_limits = policy_restrict_publishes[0][1]["motor_limits"]
    for ch in ("locomotion", "manipulation", "head", "speech"):
        assert motor_limits[ch]["max_intensity"] == 0.0


def test_safe_halt_calls_evaluator_halt(beliefs_stub):
    """SAFE_HALT must call evaluator.halt() to deny all future proposals."""
    svc = _make_kernel_service()
    svc._evaluator.halt = MagicMock()
    beliefs_stub.apply_runtime_restrictions.return_value = svc._evaluator._body_profile

    _run(svc._handle_safety_halt({"reason": "test", "operator_id": "op1"}))

    svc._evaluator.halt.assert_called_once_with("test")


# ---------------------------------------------------------------------------
# Tests: NATS config — privileged subject permission matrix
# ---------------------------------------------------------------------------


def _read_nats_conf() -> str:
    from pathlib import Path
    conf = Path(__file__).resolve().parents[2] / "deploy" / "nats-1m.conf"
    return conf.read_text(encoding="utf-8")


def test_nats_conf_has_named_user_blocks_for_all_four_safety_services():
    conf = _read_nats_conf()
    for service in ("kernel", "safety-supervisor", "beliefs", "overrides"):
        assert f"user: {service}" in conf, f"Missing user block for {service}"


def test_nats_conf_kernel_publish_includes_decision_subjects():
    conf = _read_nats_conf()
    assert "decision.>" in conf


def test_nats_conf_non_kernel_services_deny_privileged_subjects():
    conf = _read_nats_conf()
    deny_count = conf.count('"policy.*"')
    assert deny_count >= 3, (
        f"Expected deny entries for policy.* in safety-supervisor, beliefs, overrides "
        f"(got {deny_count})"
    )


def test_nats_conf_decision_subject_denied_for_non_kernel_services():
    conf = _read_nats_conf()
    deny_count = conf.count('"decision.>"')
    assert deny_count >= 4, (
        f"Expected decision.> in kernel allow + 3 service deny lists (got {deny_count})"
    )


def test_nats_conf_has_dev_default_fallback():
    conf = _read_nats_conf()
    assert "no_auth_user" in conf
    assert "dev_default" in conf


def test_nats_conf_overrides_can_publish_proposal_new():
    conf = _read_nats_conf()
    assert "proposal.new" in conf


def test_nats_conf_cognitive_response_validated_present():
    conf = _read_nats_conf()
    assert "cognitive.response.validated" in conf
