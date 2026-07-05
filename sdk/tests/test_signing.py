"""Tests for Kernel-decision signing — the unforgeable safety gate (Phase 1.2)."""

from activelearning.signing import (
    DECISION_KEY_ENV,
    SIGNATURE_FIELD,
    sign_decision,
    signing_enabled,
    verify_decision,
)

KEY = "unit-test-decision-secret"


def _decision(decision_type="ALLOW", trace="trace-1"):
    return {
        "trace_id": trace,
        "type": decision_type,
        "reason": "ok",
        "transformations": None,
        "risk_score": 0.1,
        "issued_at": 1234567890,
        "expires_at": 1234567890 + 60000,
    }


def test_sign_then_verify_roundtrip():
    signed = sign_decision(_decision(), KEY)
    assert SIGNATURE_FIELD in signed
    assert verify_decision(signed, KEY) is True


def test_forged_unsigned_decision_rejected():
    # An attacker publishes a decision with no signature — must be rejected.
    assert verify_decision(_decision("ALLOW"), KEY) is False


def test_tampered_type_rejected():
    # Attacker signs a DENY then flips it to ALLOW — signature must no longer match.
    signed = sign_decision(_decision("DENY"), KEY)
    signed["type"] = "ALLOW"
    assert verify_decision(signed, KEY) is False


def test_tampered_trace_id_rejected():
    signed = sign_decision(_decision(trace="trace-1"), KEY)
    signed["trace_id"] = "trace-2"
    assert verify_decision(signed, KEY) is False


def test_wrong_key_rejected():
    signed = sign_decision(_decision(), KEY)
    assert verify_decision(signed, "a-different-key") is False


def test_non_security_field_change_does_not_break_signature():
    # `reason` is not part of the signed fields, so editing it stays valid.
    signed = sign_decision(_decision(), KEY)
    signed["reason"] = "rephrased explanation"
    assert verify_decision(signed, KEY) is True


def test_legacy_mode_without_key_accepts(monkeypatch):
    monkeypatch.delenv(DECISION_KEY_ENV, raising=False)
    assert signing_enabled() is False
    # No key configured anywhere → signing disabled → verification passes.
    assert verify_decision(_decision()) is True
    # sign_decision is a no-op (no signature attached) when disabled.
    assert SIGNATURE_FIELD not in sign_decision(_decision())


def test_env_key_enforced(monkeypatch):
    monkeypatch.setenv(DECISION_KEY_ENV, KEY)
    assert signing_enabled() is True
    signed = sign_decision(_decision())  # uses env key
    assert verify_decision(signed) is True  # uses env key
    assert verify_decision(_decision()) is False  # forged/unsigned rejected
