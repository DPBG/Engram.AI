"""Regression tests for SAFE_HALT resume operator authentication (issue #192).

SAFE_HALT is the system-wide emergency stop. Releasing it (``safety.resume``)
must be restricted to authenticated operators — accepting an arbitrary
operator_id string with no verification lets anyone on the bus un-halt the
system. These tests cover the signing helpers in ``activelearning.signing``
and the end-to-end gate in ``kernel.service._handle_safety_resume``.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from activelearning.signing import (
    OPERATOR_KEY_ENV,
    OPERATOR_TIMESTAMP_TOLERANCE_MS,
    sign_operator_action,
    verify_operator_action,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _payload(operator_id: str = "ops:alice", skew_ms: int = 0) -> dict:
    """Build a resume payload with a fresh timestamp."""
    return {
        "operator_id": operator_id,
        "timestamp": int(time.time() * 1000) + skew_ms,
    }


def _make_service():
    """Return a KernelService stub wired with a mock event bus, pre-halted."""
    from kernel.evaluator import KernelEvaluator
    from kernel.service import KernelService

    svc = object.__new__(KernelService)
    svc.logger = MagicMock()
    svc.event_bus = MagicMock()
    svc.event_bus.publish = AsyncMock()
    svc._evaluator = KernelEvaluator()
    svc._evaluator.halt("test halt")
    return svc


# ── verify_operator_action unit tests ────────────────────────────────────────


def test_dev_mode_accepts_unsigned_with_warning(monkeypatch):
    """No ENGRAM_OPERATOR_KEY → dev mode accepts any payload (one-time warning)."""
    monkeypatch.delenv(OPERATOR_KEY_ENV, raising=False)
    assert verify_operator_action({"operator_id": "anyone", "timestamp": 0}, "resume") is True


def test_signed_payload_accepted(monkeypatch):
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    signed = sign_operator_action(_payload(), "resume")
    assert verify_operator_action(signed, "resume") is True


def test_missing_signature_rejected(monkeypatch):
    """Payload with no _op_sig must be rejected when key is set."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    assert verify_operator_action(_payload(), "resume") is False


def test_wrong_key_rejected(monkeypatch):
    """Signature produced with a different key must not verify."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "correct-key")
    signed = sign_operator_action(_payload(), "resume", key="wrong-key")
    assert verify_operator_action(signed, "resume") is False


def test_tampered_operator_id_rejected(monkeypatch):
    """Changing operator_id after signing must invalidate the signature."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    signed = sign_operator_action(_payload("ops:alice"), "resume")
    signed["operator_id"] = "ops:attacker"
    assert verify_operator_action(signed, "resume") is False


def test_stale_timestamp_rejected(monkeypatch):
    """Timestamp older than OPERATOR_TIMESTAMP_TOLERANCE_MS prevents replay."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    stale = sign_operator_action(
        _payload(skew_ms=-(OPERATOR_TIMESTAMP_TOLERANCE_MS + 1_000)), "resume"
    )
    assert verify_operator_action(stale, "resume") is False


def test_future_timestamp_too_far_rejected(monkeypatch):
    """Timestamps far in the future (clock skew attack) are also rejected."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    future = sign_operator_action(
        _payload(skew_ms=OPERATOR_TIMESTAMP_TOLERANCE_MS + 1_000), "resume"
    )
    assert verify_operator_action(future, "resume") is False


def test_missing_timestamp_rejected(monkeypatch):
    """Payload without a timestamp field must be rejected (default=0 is in 1970)."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    # Manually sign without a timestamp so _op_sig exists but timestamp is absent
    bare = {"operator_id": "ops:alice"}
    bare["_op_sig"] = "fakesig"
    assert verify_operator_action(bare, "resume") is False


def test_cross_action_replay_rejected(monkeypatch):
    """A halt-action signature cannot be replayed as a resume signature."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    halt_signed = sign_operator_action(_payload(), "halt")
    assert verify_operator_action(halt_signed, "resume") is False


# ── _handle_safety_resume integration tests ───────────────────────────────────


def test_unsigned_resume_rejected_when_key_set(monkeypatch):
    """Unsigned resume must be rejected; evaluator must stay halted."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    svc = _make_service()
    asyncio.run(svc._handle_safety_resume(_payload()))
    assert svc._evaluator.is_halted is True


def test_authenticated_resume_releases_halt(monkeypatch):
    """Correctly signed resume must disengage the kill switch."""
    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    svc = _make_service()
    signed = sign_operator_action(_payload(), "resume")
    asyncio.run(svc._handle_safety_resume(signed))
    assert svc._evaluator.is_halted is False


def test_dev_mode_resume_accepted_without_key(monkeypatch):
    """Dev mode (no key): resume accepted after warning, halt released."""
    monkeypatch.delenv(OPERATOR_KEY_ENV, raising=False)
    svc = _make_service()
    asyncio.run(svc._handle_safety_resume(_payload()))
    assert svc._evaluator.is_halted is False


def test_rejected_resume_publishes_still_halted_status(monkeypatch):
    """Rejected resume must publish SAFETY_HALT_STATUS with halted=True."""
    from activelearning.subjects import Subjects

    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    svc = _make_service()
    asyncio.run(svc._handle_safety_resume(_payload("ops:attacker")))

    svc.event_bus.publish.assert_awaited_once()
    subject, status = svc.event_bus.publish.call_args[0]
    assert subject == Subjects.SAFETY_HALT_STATUS
    assert status["halted"] is True
    assert "rejection_reason" in status


def test_successful_resume_publishes_not_halted_status(monkeypatch):
    """Successful resume must publish SAFETY_HALT_STATUS with halted=False."""
    from activelearning.subjects import Subjects

    monkeypatch.setenv(OPERATOR_KEY_ENV, "test-key")
    svc = _make_service()
    signed = sign_operator_action(_payload("ops:alice"), "resume")
    asyncio.run(svc._handle_safety_resume(signed))

    svc.event_bus.publish.assert_awaited_once()
    subject, status = svc.event_bus.publish.call_args[0]
    assert subject == Subjects.SAFETY_HALT_STATUS
    assert status["halted"] is False
    assert status["operator_id"] == "ops:alice"
