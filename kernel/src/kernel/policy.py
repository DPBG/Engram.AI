"""Policy Management — validation, rollback, and decision tracking.

Handles:
1. Policy update validation (schema + hierarchy enforcement)
2. Rollback: stores last-known-good profile for revert on anomaly
3. Decision sequence tracking: consecutive DENY escalation
4. Cognitive response validation gate
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from beliefs.profiles import BodyProfile

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Policy Rollback
# ------------------------------------------------------------------

@dataclass
class PolicySnapshot:
    """A snapshot of the active policy for rollback.

    Stores a deep copy of the full BodyProfile so rollback is independent
    of disk — protects against file tampering between snapshot and revert.
    """
    profile_name: str
    profile: Any  # BodyProfile — stored as deep copy
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


class PolicyRollbackManager:
    """Maintains a rollback stack for policy changes.

    On every profile load or restriction, the previous state is saved.
    If anomaly is detected (e.g., watchdog CRITICAL after a policy change),
    the Kernel can revert to the last-known-good policy.
    """

    def __init__(self, max_history: int = 5):
        self._history: deque[PolicySnapshot] = deque(maxlen=max_history)
        self._current: Optional[PolicySnapshot] = None

    def snapshot(self, profile: "BodyProfile", reason: str = "") -> None:
        """Save current profile state before applying a change."""
        import copy
        snap = PolicySnapshot(
            profile_name=profile.name,
            profile=copy.deepcopy(profile),
            reason=reason,
        )
        if self._current is not None:
            self._history.append(self._current)
        self._current = snap

    def rollback(self) -> Optional[PolicySnapshot]:
        """Pop the last-known-good snapshot for revert.

        Returns None if no history exists.
        The returned snapshot contains a full BodyProfile — no disk I/O needed.
        """
        if not self._history:
            return None
        snap = self._history.pop()
        self._current = snap
        return snap

    @property
    def has_rollback(self) -> bool:
        return len(self._history) > 0

    @property
    def current(self) -> Optional[PolicySnapshot]:
        return self._current

    def get_state(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "current": {
                "profile_name": self._current.profile_name,
                "timestamp": self._current.timestamp,
                "reason": self._current.reason,
            } if self._current else None,
            "history_len": len(self._history),
        }


# ------------------------------------------------------------------
# Decision Sequence Tracker
# ------------------------------------------------------------------

@dataclass
class DenySequence:
    """Tracks consecutive DENY decisions for escalation."""
    channel: str
    count: int = 0
    first_at: float = 0.0
    last_at: float = 0.0


class DecisionSequenceTracker:
    """Tracks consecutive DENY decisions per channel for escalation.

    If a channel receives N consecutive DENYs within a time window,
    escalate: increase risk score, or request motor limit reduction.

    Escalation levels:
      3 consecutive DENYs → WARNING log
      5 consecutive DENYs → Publish escalation event
      10 consecutive DENYs → Request channel disable via policy.restrict
    """

    WARN_THRESHOLD = 3
    ESCALATE_THRESHOLD = 5
    DISABLE_THRESHOLD = 10

    def __init__(self, window_seconds: float = 60.0):
        self._sequences: dict[str, DenySequence] = {}
        self._window = window_seconds

    def record_decision(
        self, channel: str, decision_type: str,
    ) -> Optional[str]:
        """Record a decision and check for escalation.

        Args:
            channel: Motor channel (e.g., "locomotion").
            decision_type: "ALLOW", "DENY", "TRANSFORM", "DEFER".

        Returns:
            Escalation action if threshold hit:
            "warn", "escalate", "disable", or None.
        """
        now = time.time()

        if decision_type != "DENY":
            # Reset streak on any non-DENY
            self._sequences.pop(channel, None)
            return None

        seq = self._sequences.get(channel)
        if seq is None:
            seq = DenySequence(channel=channel, count=0, first_at=now)
            self._sequences[channel] = seq

        # Check if the window has expired — reset
        if now - seq.first_at > self._window:
            seq.count = 0
            seq.first_at = now

        seq.count += 1
        seq.last_at = now

        if seq.count >= self.DISABLE_THRESHOLD:
            logger.error(
                f"Channel '{channel}': {seq.count} consecutive DENYs — "
                f"requesting channel disable"
            )
            return "disable"
        elif seq.count >= self.ESCALATE_THRESHOLD:
            logger.warning(
                f"Channel '{channel}': {seq.count} consecutive DENYs — escalating"
            )
            return "escalate"
        elif seq.count >= self.WARN_THRESHOLD:
            logger.warning(
                f"Channel '{channel}': {seq.count} consecutive DENYs"
            )
            return "warn"

        return None

    def get_state(self) -> dict[str, Any]:
        """Get current deny sequences for status reporting."""
        return {
            ch: {"count": seq.count, "first_at": seq.first_at, "last_at": seq.last_at}
            for ch, seq in self._sequences.items()
            if seq.count > 0
        }


# ------------------------------------------------------------------
# Cognitive Response Validation
# ------------------------------------------------------------------

# Patterns that should never appear in LLM responses injected into the brain
_BLOCKED_RESPONSE_PATTERNS = [
    "ignore previous",
    "disregard instructions",
    "override safety",
    "bypass kernel",
    "disable watchdog",
    "modify beliefs",
    "change values",
    "remove norms",
    "sudo",
    "rm -rf",
    "format c:",
]

# Common Unicode confusables (Cyrillic/Greek → Latin) for homoglyph attack defense.
# Covers the most common lookalike characters used in prompt injection.
_CONFUSABLE_MAP = str.maketrans({
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "p",  # Cyrillic р → p
    "\u0441": "c",  # Cyrillic с
    "\u0443": "y",  # Cyrillic у
    "\u0456": "i",  # Cyrillic і
    "\u0455": "s",  # Cyrillic ѕ
    "\u04bb": "h",  # Cyrillic һ
    "\u0501": "d",  # Cyrillic ԁ
    "\u051b": "q",  # Cyrillic ԛ
    "\u0391": "A",  # Greek Α
    "\u0392": "B",  # Greek Β
    "\u0395": "E",  # Greek Ε
    "\u039a": "K",  # Greek Κ
    "\u039c": "M",  # Greek Μ
    "\u039d": "N",  # Greek Ν
    "\u039f": "O",  # Greek Ο
    "\u03a1": "P",  # Greek Ρ
    "\u03a4": "T",  # Greek Τ
    "\u03bf": "o",  # Greek ο
})

# Max length for cognitive responses (characters)
_MAX_RESPONSE_LENGTH = 2000


def validate_cognitive_response(
    response_text: str,
    profile: Optional["BodyProfile"] = None,
) -> tuple[bool, str]:
    """Validate an LLM response before it's injected into the sensory pipeline.

    Checks:
    1. Response not empty
    2. Response not too long (truncation risk)
    3. No prompt injection patterns
    4. Profile allows cognitive queries

    Args:
        response_text: The LLM response text.
        profile: Active body profile (if loaded).

    Returns:
        (is_valid, reason) — if invalid, reason explains why.
    """
    if not response_text or not response_text.strip():
        return False, "Empty response"

    if len(response_text) > _MAX_RESPONSE_LENGTH:
        return False, (
            f"Response too long ({len(response_text)} chars, "
            f"max {_MAX_RESPONSE_LENGTH})"
        )

    # Prompt injection detection — multi-layer Unicode defense:
    # 1. NFKD normalization (catches compatibility chars)
    # 2. Strip zero-width/format characters (catches invisible insertions)
    # 3. Confusable character replacement (catches Cyrillic/Greek homoglyphs)
    import unicodedata
    normalized = unicodedata.normalize("NFKD", response_text)
    # Strip zero-width and format characters
    cleaned = "".join(
        ch for ch in normalized
        if unicodedata.category(ch) != "Cf"  # format characters (zero-width, BOM, etc.)
    )
    lower = cleaned.lower()
    # Replace common confusable characters (Cyrillic/Greek → Latin)
    deconfused = lower.translate(_CONFUSABLE_MAP)
    for pattern in _BLOCKED_RESPONSE_PATTERNS:
        if pattern in lower or pattern in deconfused:
            return False, f"Blocked pattern detected: '{pattern}'"

    # Profile check
    if profile is not None:
        if not profile.capabilities.get("can_query_llm", True):
            return False, "Body profile disallows cognitive queries"

    return True, ""


# ------------------------------------------------------------------
# Policy Update Validation Schema
# ------------------------------------------------------------------

_VALID_POLICY_FIELDS = {
    "profile_name",      # string — switch to a different profile
    "motor_limits",      # dict[channel → {max_intensity, max_acceleration}]
    "capabilities",      # dict[cap_key → bool/numeric]
    "reason",            # string — why this update was issued
    "operator_id",       # string — who issued the update
    "timestamp",         # float — when the update was issued
}

_REQUIRED_POLICY_FIELDS = {"reason", "operator_id"}


def validate_policy_update(update: dict[str, Any]) -> tuple[bool, str]:
    """Validate a policy update message from the cloud.

    Checks:
    1. No unknown fields (prevents injection of unexpected keys)
    2. Required fields present (reason, operator_id for audit)
    3. Motor limits are valid numbers
    4. Capabilities are valid types

    Args:
        update: The policy update payload.

    Returns:
        (is_valid, reason) — if invalid, reason explains why.
    """
    # Check for unknown fields
    unknown = set(update.keys()) - _VALID_POLICY_FIELDS
    if unknown:
        return False, f"Unknown policy fields: {unknown}"

    # Check required fields
    missing = _REQUIRED_POLICY_FIELDS - set(update.keys())
    if missing:
        return False, f"Missing required fields: {missing}"

    # Validate motor_limits structure
    motor_limits = update.get("motor_limits", {})
    if not isinstance(motor_limits, dict):
        return False, "motor_limits must be a dict"
    for channel, limits in motor_limits.items():
        if not isinstance(limits, dict):
            return False, f"motor_limits['{channel}'] must be a dict"
        for key in ("max_intensity", "max_acceleration"):
            if key in limits:
                val = limits[key]
                if not isinstance(val, (int, float)):
                    return False, f"motor_limits['{channel}']['{key}'] must be numeric"
                if val < 0.0 or val > 1.0:
                    return False, (
                        f"motor_limits['{channel}']['{key}']={val} "
                        f"out of range [0.0, 1.0]"
                    )

    # Validate capabilities structure
    capabilities = update.get("capabilities", {})
    if not isinstance(capabilities, dict):
        return False, "capabilities must be a dict"
    for cap_key, cap_val in capabilities.items():
        if not isinstance(cap_val, (bool, int, float)):
            return False, (
                f"capabilities['{cap_key}'] must be bool or numeric, "
                f"got {type(cap_val).__name__}"
            )

    return True, ""
