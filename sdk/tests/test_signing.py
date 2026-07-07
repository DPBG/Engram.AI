"""Tests for Kernel-decision signing — the unforgeable safety gate (Phase 1.2)."""

import time

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


# ── signing latency budget (Issue #185 / M1.2 regression gate) ───────────────
#
# The cryptographic overhead of the decision-bus signing path must stay within
# a formal budget. If it exceeds this budget, enabling signing in production
# would push the Kernel's p99 decision latency over the SLO (issue #193).
#
# Budget rationale: HMAC-SHA256 over a ~150-byte canonical JSON payload takes
# < 100 µs on any modern CPU. 2 ms (20× headroom) is tight enough to catch a
# real regression (accidental SHA-512, double-hashing, etc.) but generous
# enough that CI hardware variance cannot cause spurious failures.
#
# The round-trip budget (sign + verify) covers the full per-proposal overhead
# the Kernel incurs: sign on emit, verify on every downstream waiter.

_N_SAMPLES = 500
_SIGN_P99_BUDGET_MS = 2.0
_VERIFY_P99_BUDGET_MS = 2.0
_ROUNDTRIP_P99_BUDGET_MS = 4.0


def _p99_ms(times_s: list) -> float:
    s = sorted(times_s)
    return s[min(int(0.99 * len(s)), len(s) - 1)] * 1000.0


def test_sign_decision_p99_latency_budget():
    """sign_decision isolation: p99 must stay under the signing budget."""
    payload = _decision()
    samples = []
    for _ in range(_N_SAMPLES):
        t0 = time.perf_counter()
        sign_decision(payload, KEY)
        samples.append(time.perf_counter() - t0)
    p99 = _p99_ms(samples)
    assert p99 < _SIGN_P99_BUDGET_MS, (
        f"sign_decision p99 {p99:.3f} ms exceeds budget {_SIGN_P99_BUDGET_MS} ms — "
        "a signing regression was introduced"
    )


def test_verify_decision_p99_latency_budget():
    """verify_decision isolation: p99 must stay under the verification budget."""
    signed = sign_decision(_decision(), KEY)
    samples = []
    for _ in range(_N_SAMPLES):
        t0 = time.perf_counter()
        verify_decision(signed, KEY)
        samples.append(time.perf_counter() - t0)
    p99 = _p99_ms(samples)
    assert p99 < _VERIFY_P99_BUDGET_MS, (
        f"verify_decision p99 {p99:.3f} ms exceeds budget {_VERIFY_P99_BUDGET_MS} ms — "
        "a verification regression was introduced"
    )


def test_sign_then_verify_roundtrip_p99_latency_budget():
    """Full per-proposal signing overhead: sign on emit + verify on receive.

    This is the end-to-end crypto cost the Kernel pays on every decision:
    sign_decision in _signed_code_decision / sign_decision in _handle_propose_action,
    then verify_decision in every downstream waiter.
    """
    payload = _decision()
    samples = []
    for _ in range(_N_SAMPLES):
        t0 = time.perf_counter()
        signed = sign_decision(payload, KEY)
        verify_decision(signed, KEY)
        samples.append(time.perf_counter() - t0)
    p99 = _p99_ms(samples)
    assert p99 < _ROUNDTRIP_P99_BUDGET_MS, (
        f"sign+verify round-trip p99 {p99:.3f} ms exceeds budget {_ROUNDTRIP_P99_BUDGET_MS} ms — "
        "total signing overhead now threatens the Kernel decision SLO"
    )
