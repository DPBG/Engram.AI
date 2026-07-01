"""
M1 prerequisite gate for Meta-Programmer autonomy.

The meta-programmer may not process knowledge gaps (generate or deploy code)
until two M1 safety controls are confirmed live:

  1. Decision-bus signing (Task 1.2) — all kernel decisions carry a
     cryptographic signature; forgery by another service is detectable.
  2. Sandbox fail-closed hardening (E1.3.2) — the sandbox refuses to run when
     containment cannot be fully applied; there is no partial-isolation fallback.

Both controls are tracked via environment variables set by ops/CI when the
corresponding milestone work ships. Until both are confirmed, every incoming
``knowledge.gap`` is refused with a denial so the meta-programmer cannot acquire
new capabilities during an unsafe window.

Env vars (set to "1", "true", or "yes" to enable; anything else → incomplete):
  ENGRAM_DECISION_BUS_SIGNING_ENABLED   — Task 1.2 complete
  ENGRAM_SANDBOX_FAIL_CLOSED_ENABLED    — E1.3.2 complete

See: CLAUDE.md §3 (Safety Architecture), docs/SANDBOX-THREAT-MODEL.md §3,
     GitHub issue #140.
"""

from __future__ import annotations

import os

SIGNING_ENV = "ENGRAM_DECISION_BUS_SIGNING_ENABLED"
SANDBOX_FAILCLOSED_ENV = "ENGRAM_SANDBOX_FAIL_CLOSED_ENABLED"

_TRUE_VALUES = {"1", "true", "yes"}


def _flag(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip().lower() in _TRUE_VALUES


def signing_enabled() -> bool:
    """True when decision-bus signing (Task 1.2) is confirmed live."""
    return _flag(SIGNING_ENV)


def sandbox_failclosed_enabled() -> bool:
    """True when sandbox fail-closed hardening (E1.3.2) is confirmed live."""
    return _flag(SANDBOX_FAILCLOSED_ENV)


def m1_complete() -> bool:
    """True only when *both* M1 prerequisites are confirmed live.

    Fails closed: an unset or unrecognised value is treated as incomplete.
    """
    return signing_enabled() and sandbox_failclosed_enabled()


def check_m1_or_deny() -> tuple[bool, str]:
    """Check M1 prerequisites; return (ok, denial_reason).

    When ``ok`` is True the caller may proceed with code generation / deploy.
    When ``ok`` is False the caller must deny the proposal immediately and
    surface ``denial_reason`` — do NOT fall through to generation or staging.
    """
    if m1_complete():
        return True, ""

    missing = []
    if not signing_enabled():
        missing.append(
            f"decision-bus signing not yet live " f"(set {SIGNING_ENV}=1 when Task 1.2 ships)"
        )
    if not sandbox_failclosed_enabled():
        missing.append(
            f"sandbox fail-closed not yet live "
            f"(set {SANDBOX_FAILCLOSED_ENV}=1 when E1.3.2 ships)"
        )

    reason = (
        "M1 safety prerequisites incomplete — meta-programmer autonomy blocked "
        "(fail-closed). Missing: " + "; ".join(missing)
    )
    return False, reason
