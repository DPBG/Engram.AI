"""
HMAC signing for Kernel decisions — makes the safety gate unforgeable.

Kernel decisions travel over open NATS pub/sub. Without authentication any
participant on the bus (including code the Meta-Programmer generated) could
publish a forged ``decision.<trace_id>`` with ``type: ALLOW`` and have it beat
the real Kernel. To prevent that, the Kernel signs every decision with a shared
secret (``ENGRAM_DECISION_KEY``) and decision waiters verify the signature
before accepting it.

Design notes
------------
- The signature is an HMAC-SHA256 over a canonical JSON of the **security-
  relevant, JSON-stable** fields (``trace_id``, ``type``, ``risk_score``,
  ``expires_at``, ``issued_at``). These round-trip identically through NATS
  (de)serialization, so sign-side and verify-side agree.
- **Fail-safe by configuration:** when ``ENGRAM_DECISION_KEY`` is set, a missing
  or invalid signature is *rejected*. When it is not set, signing is disabled
  and verification passes (legacy/dev mode) — but a one-time warning is logged
  so operators know the gate is unauthenticated.
- ``hmac.compare_digest`` is used for constant-time comparison.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

#: Environment variable holding the shared decision-signing secret.
DECISION_KEY_ENV = "ENGRAM_DECISION_KEY"

#: Field name used to carry the signature inside a decision payload.
SIGNATURE_FIELD = "_sig"

#: Fields covered by the signature. Kept to JSON-stable primitives so the
#: signed bytes are identical on both the signing and verifying sides.
SIGNED_FIELDS = ("trace_id", "type", "risk_score", "expires_at", "issued_at")

_warned_unsigned = False


def get_decision_key() -> str | None:
    """Return the configured signing key, or ``None`` if signing is disabled."""
    key = os.environ.get(DECISION_KEY_ENV, "").strip()
    return key or None


def signing_enabled() -> bool:
    """True when a signing key is configured (signatures are enforced)."""
    return get_decision_key() is not None


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical, deterministic encoding of the signed subset of a payload."""
    subset = {k: payload.get(k) for k in SIGNED_FIELDS}
    return json.dumps(subset, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def compute_signature(payload: dict[str, Any], key: str) -> str:
    """Compute the HMAC-SHA256 signature (hex) for a decision payload."""
    return hmac.new(key.encode("utf-8"), _canonical_bytes(payload), hashlib.sha256).hexdigest()


def sign_decision(payload: dict[str, Any], key: str | None = None) -> dict[str, Any]:
    """Return a copy of ``payload`` with a signature attached.

    No-op (returns the payload unchanged) when signing is disabled. The
    signature itself is excluded from the signed bytes, so re-signing or
    verifying an already-signed payload is consistent.
    """
    key = key or get_decision_key()
    if not key:
        return payload
    signed = dict(payload)
    signed[SIGNATURE_FIELD] = compute_signature(signed, key)
    return signed


def verify_decision(payload: dict[str, Any], key: str | None = None) -> bool:
    """Return ``True`` if ``payload`` is an authentic Kernel decision.

    - Signing disabled (no key) → ``True`` (legacy mode), with a one-time warning.
    - Signing enabled → the payload must carry a signature matching its signed
      fields; missing/invalid signatures (i.e. forgeries or tampering) → ``False``.
    """
    global _warned_unsigned
    key = key or get_decision_key()
    if not key:
        if not _warned_unsigned:
            logger.warning(
                "Decision signing is DISABLED (%s not set) — the Kernel gate is "
                "unauthenticated. Set %s on the Kernel and all services to enforce it.",
                DECISION_KEY_ENV,
                DECISION_KEY_ENV,
            )
            _warned_unsigned = True
        return True
    sig = payload.get(SIGNATURE_FIELD)
    if not isinstance(sig, str) or not sig:
        return False
    expected = compute_signature(payload, key)
    return hmac.compare_digest(sig, expected)


# ── Operator-action signing (SAFE_HALT resume and similar privileged commands) ─
#
# Operator actions (``safety.resume``) require a separate key so that
# operator credentials and decision-bus credentials can be rotated
# independently. The same fail-closed, HMAC-SHA256 pattern is used:
# when ``ENGRAM_OPERATOR_KEY`` is set a missing or invalid signature is
# rejected; when it is not set a one-time warning is emitted and the
# action is accepted (dev / pre-credential mode).
#
# Signed fields: ``operator_id``, ``action`` (the action name, e.g.
# ``"resume"``), ``timestamp`` (epoch ms). The ``action`` field binds the
# signature to a specific command so a halt signature cannot be replayed
# as a resume. The ``timestamp`` field limits the replay window to
# ±``OPERATOR_TIMESTAMP_TOLERANCE_MS`` milliseconds.

#: Environment variable holding the operator-action signing secret.
OPERATOR_KEY_ENV = "ENGRAM_OPERATOR_KEY"

#: Signature field name embedded in operator-action payloads.
OPERATOR_SIGNATURE_FIELD = "_op_sig"

#: Maximum age (or future skew) accepted for an operator-action timestamp.
OPERATOR_TIMESTAMP_TOLERANCE_MS: int = 5 * 60 * 1000  # 5 minutes

_warned_unsigned_operator = False


def get_operator_key() -> str | None:
    """Return the configured operator-action signing key, or ``None`` if unset."""
    key = os.environ.get(OPERATOR_KEY_ENV, "").strip()
    return key or None


def operator_signing_enabled() -> bool:
    """True when an operator signing key is configured."""
    return get_operator_key() is not None


def _operator_canonical_bytes(payload: dict[str, Any], action: str) -> bytes:
    """Canonical, deterministic encoding of the operator-action signed fields."""
    subset = {
        "action": action,
        "operator_id": payload.get("operator_id", ""),
        "timestamp": payload.get("timestamp", 0),
    }
    return json.dumps(subset, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sign_operator_action(
    payload: dict[str, Any], action: str, key: str | None = None
) -> dict[str, Any]:
    """Return a copy of ``payload`` with an operator-action signature attached.

    No-op when ``ENGRAM_OPERATOR_KEY`` is not set. The payload must contain
    an ``operator_id`` (str) and a ``timestamp`` (epoch ms int).
    """
    key = key or get_operator_key()
    if not key:
        return payload
    signed = dict(payload)
    signed[OPERATOR_SIGNATURE_FIELD] = hmac.new(
        key.encode("utf-8"),
        _operator_canonical_bytes(signed, action),
        hashlib.sha256,
    ).hexdigest()
    return signed


def verify_operator_action(payload: dict[str, Any], action: str, key: str | None = None) -> bool:
    """Return ``True`` if ``payload`` is an authenticated operator action.

    Fail-closed rules when ``ENGRAM_OPERATOR_KEY`` is set:

    - Missing ``_op_sig`` → ``False``
    - Invalid HMAC (wrong key, tampered fields) → ``False``
    - ``timestamp`` absent or outside ``OPERATOR_TIMESTAMP_TOLERANCE_MS`` → ``False``
      (prevents replay after a new SAFE_HALT is issued)

    When ``ENGRAM_OPERATOR_KEY`` is *not* set the function returns ``True``
    after emitting a one-time warning (dev / pre-credential mode).
    """
    global _warned_unsigned_operator
    key = key or get_operator_key()
    if not key:
        if not _warned_unsigned_operator:
            logger.warning(
                "Operator-action signing is DISABLED (%s not set) — SAFE_HALT "
                "resume accepts any caller. Set %s to enforce authenticated resume.",
                OPERATOR_KEY_ENV,
                OPERATOR_KEY_ENV,
            )
            _warned_unsigned_operator = True
        return True

    # Timestamp check — rejects replays outside the tolerance window.
    timestamp = payload.get("timestamp", 0)
    if not isinstance(timestamp, (int, float)) or timestamp == 0:
        logger.error("Operator-action rejected: missing or zero timestamp (action=%r)", action)
        return False
    now_ms = int(time.time() * 1000)
    if abs(now_ms - int(timestamp)) > OPERATOR_TIMESTAMP_TOLERANCE_MS:
        logger.error(
            "Operator-action rejected: timestamp outside tolerance window "
            "(action=%r, skew_ms=%d)",
            action,
            now_ms - int(timestamp),
        )
        return False

    # Signature check.
    sig = payload.get(OPERATOR_SIGNATURE_FIELD)
    if not isinstance(sig, str) or not sig:
        logger.error("Operator-action rejected: missing signature (action=%r)", action)
        return False
    expected = hmac.new(
        key.encode("utf-8"),
        _operator_canonical_bytes(payload, action),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        logger.error("Operator-action rejected: invalid signature (action=%r)", action)
        return False
    return True
