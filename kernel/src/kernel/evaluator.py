"""
Kernel Evaluator - Decision making logic.

The Moral Kernel is the immutable gatekeeper that evaluates all proposals
and makes ALLOW/TRANSFORM/DENY/DEFER decisions.

Body-profile integration:
  When a BodyProfile is loaded, the evaluator enforces:
  1. Channel allowed — DENY if capability marker says channel is disabled
  2. Motor limit clamping — TRANSFORM intensity down to profile max
  3. Profile norm risk boost — adds risk from profile-specific norms
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import time

from activelearning import KernelDecisionType as DecisionType, RiskAnalysis

if TYPE_CHECKING:
    from beliefs.profiles import BodyProfile

logger = logging.getLogger(__name__)


@dataclass
class KernelDecision:
    """A decision from the Kernel."""
    trace_id: str
    type: DecisionType
    reason: Optional[str] = None
    transformations: Optional[list[dict[str, Any]]] = None
    risk_score: float = 0.0
    issued_at: int = field(default_factory=lambda: int(time.time() * 1000))
    expires_at: Optional[int] = None


# Protected paths that cannot be modified
PROTECTED_PATHS = [
    "/kernel/",
    "/safety-supervisor/",
    "/meta-programmer/orchestrator.py",
    "/meta-programmer/agents/",
]

# Elevated scrutiny patterns
DANGEROUS_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"__import__\s*\(",
    r"\bimportlib\b",
    r"\bos\.system\s*\(",
    r"\bsubprocess\b",
    r"\bsocket\b",
    r"\bopen\s*\([^)]*['\"]w",
]

# Self-referential patterns
SELF_REFERENTIAL_PATTERNS = [
    r"meta.?programmer",
    r"kernel",
    r"safety.?supervisor",
]


class KernelEvaluator:
    """
    The Moral Kernel evaluator.

    Evaluates proposals against safety rules and returns decisions.
    This is the immutable core that cannot be bypassed.
    """

    def __init__(
        self,
        deny_threshold: float = 0.8,
        defer_threshold: float = 0.5,
        decision_ttl_ms: int = 60000,  # 1 minute
        defer_ttl_ms: int = 300000,  # 5 minutes for a human to respond
        body_profile: Optional["BodyProfile"] = None,
    ):
        self.deny_threshold = deny_threshold
        self.defer_threshold = defer_threshold
        self.decision_ttl_ms = decision_ttl_ms
        # A DEFER carries a deadline: if no human answers before it expires, the
        # approval consumer must treat the pending proposal as DENY (fail-closed,
        # Phase 1.9) rather than letting it linger indefinitely.
        self.defer_ttl_ms = defer_ttl_ms
        self._body_profile: Optional["BodyProfile"] = body_profile
        # SAFE_HALT kill switch (Phase 1.9). When halted, the Kernel — the sole
        # authority that may approve anything — DENIES every proposal. Because
        # actions, code deployments, and Coordinator task execution all route
        # through here, one flag stops the whole system at once. Fail-safe.
        self._halted = False
        self._halt_reason = ""

    @property
    def is_halted(self) -> bool:
        return self._halted

    def halt(self, reason: str = "operator SAFE_HALT") -> None:
        """Engage the kill switch: every subsequent proposal is DENIED."""
        self._halted = True
        self._halt_reason = reason or "operator SAFE_HALT"
        logger.critical("KERNEL SAFE_HALT engaged: %s — denying all proposals", self._halt_reason)

    def resume(self) -> None:
        """Release the kill switch (operator action). Resumes normal evaluation."""
        self._halted = False
        self._halt_reason = ""
        logger.warning("Kernel SAFE_HALT released — resuming normal evaluation")

    def _halt_decision(self, trace_id: str) -> "KernelDecision":
        return KernelDecision(
            trace_id=trace_id,
            type=DecisionType.DENY,
            reason=f"SAFE_HALT active: {self._halt_reason}",
            risk_score=1.0,
        )

    def set_body_profile(self, profile: "BodyProfile") -> None:
        """Set or replace the active body profile.

        Once set, all subsequent proposals are checked against profile
        capability markers and motor limits.
        """
        self._body_profile = profile
        logger.info(f"Evaluator loaded body profile: {profile.name}")

    def evaluate_action_proposal(
        self,
        proposal: dict[str, Any],
        risk_analysis: Optional[RiskAnalysis] = None,
        norm_violations: Optional[list[dict[str, Any]]] = None,
    ) -> KernelDecision:
        """
        Evaluate an action proposal.

        Args:
            proposal: The action proposal
            risk_analysis: Optional risk analysis from Safety Supervisor
            norm_violations: Optional list of violated belief norms
                (each dict has 'norm_id', 'content', 'risk_boost')

        Returns:
            KernelDecision
        """
        trace_id = proposal.get("trace_id", str(uuid.uuid4()))
        action = proposal.get("action", {})

        # Kill switch: deny everything while halted.
        if self._halted:
            return self._halt_decision(trace_id)

        # Initialize risk score — clamp external input to [0.0, 1.0]
        risk_score = max(0.0, min(risk_analysis.risk_score, 1.0)) if risk_analysis else 0.0
        flags = list(risk_analysis.flags) if risk_analysis else []
        reason = None

        # Apply norm violations from Beliefs system
        if norm_violations:
            for violation in norm_violations:
                flags.append(f"NORM_VIOLATION:{violation.get('norm_id', 'unknown')}")
                risk_score += violation.get("risk_boost", 0.1)
            risk_score = min(risk_score, 1.0)  # clamp

        # Check for missing trace_id
        if not proposal.get("trace_id"):
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.DENY,
                reason="Missing trace_id",
                risk_score=1.0,
            )

        # === Body-profile checks (capability markers + motor limits) ===
        profile_result = self._check_body_profile(action, flags)
        if profile_result is not None:
            return KernelDecision(
                trace_id=trace_id,
                type=profile_result["type"],
                reason=profile_result["reason"],
                risk_score=profile_result.get("risk_score", 0.9),
                transformations=profile_result.get("transformations"),
                expires_at=(
                    int(time.time() * 1000) + self.decision_ttl_ms
                    if profile_result["type"] == DecisionType.TRANSFORM
                    else None
                ),
            )

        # Check action envelope limits
        envelope_violation = self._check_envelope(action)
        if envelope_violation:
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.DENY,
                reason=f"Envelope violation: {envelope_violation}",
                risk_score=0.9,
            )

        # Apply risk thresholds
        if risk_score >= self.deny_threshold:
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.DENY,
                reason=f"Risk too high: {risk_score:.2f}",
                risk_score=risk_score,
            )

        if risk_score >= self.defer_threshold:
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.DEFER,
                reason=f"Elevated risk ({risk_score:.2f}) - requires human approval",
                risk_score=risk_score,
                expires_at=int(time.time() * 1000) + self.defer_ttl_ms,
            )

        # Check for transformable actions
        transformations = self._generate_transformations(action, flags)
        if transformations:
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.TRANSFORM,
                reason="Action transformed for safety",
                transformations=transformations,
                risk_score=risk_score,
                expires_at=int(time.time() * 1000) + self.decision_ttl_ms,
            )

        # Allow
        return KernelDecision(
            trace_id=trace_id,
            type=DecisionType.ALLOW,
            risk_score=risk_score,
            expires_at=int(time.time() * 1000) + self.decision_ttl_ms,
        )

    def evaluate_code_proposal(
        self,
        proposal: dict[str, Any],
        risk_analysis: Optional[RiskAnalysis] = None,
    ) -> KernelDecision:
        """
        Evaluate a code proposal from Meta-Programmer.

        Args:
            proposal: The code proposal
            risk_analysis: Optional risk analysis from Safety Supervisor

        Returns:
            KernelDecision
        """
        trace_id = proposal.get("trace_id", str(uuid.uuid4()))
        target_path = proposal.get("target_path", "")
        code_preview = proposal.get("code_preview", "")

        # Kill switch: deny all code deployment while halted.
        if self._halted:
            return self._halt_decision(trace_id)

        # Initialize risk analysis
        risk_score = risk_analysis.risk_score if risk_analysis else 0.0
        flags = risk_analysis.flags.copy() if risk_analysis else []

        # Check protected paths (always DENY)
        if self._is_protected_path(target_path):
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.DENY,
                reason=f"Protected path: {target_path}",
                risk_score=1.0,
            )

        # Check for dangerous patterns
        dangerous_flags = self._check_dangerous_patterns(code_preview)
        flags.extend(dangerous_flags)

        if dangerous_flags:
            risk_score = max(risk_score, 0.7)
            logger.warning(f"Dangerous patterns detected: {dangerous_flags}")

        # Check for self-referential code — code that touches the safety/meta
        # machinery itself (kernel, safety-supervisor, meta-programmer) is never
        # auto-approved. Fail closed: DENY (not DEFER).
        if self._is_self_referential(code_preview):
            flags.append("SELF_REFERENTIAL")
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.DENY,
                reason=f"Self-referential code touches system internals: {', '.join(flags)}",
                risk_score=max(risk_score, 0.95),
            )

        # Apply thresholds
        if risk_score >= self.deny_threshold:
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.DENY,
                reason=f"Code risk too high: {', '.join(flags)}",
                risk_score=risk_score,
            )

        if risk_score >= self.defer_threshold or dangerous_flags:
            return KernelDecision(
                trace_id=trace_id,
                type=DecisionType.DEFER,
                reason=f"Code requires human review: {', '.join(flags)}",
                risk_score=risk_score,
                expires_at=int(time.time() * 1000) + self.defer_ttl_ms,
            )

        return KernelDecision(
            trace_id=trace_id,
            type=DecisionType.ALLOW,
            risk_score=risk_score,
            expires_at=int(time.time() * 1000) + self.decision_ttl_ms,
        )

    def _is_protected_path(self, path: str) -> bool:
        """Check if a path is protected."""
        path = path.lower()
        for protected in PROTECTED_PATHS:
            if protected.lower() in path:
                return True
        return False

    def _check_dangerous_patterns(self, code: str) -> list[str]:
        """Check code for dangerous patterns."""
        flags = []
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                flags.append(f"DANGEROUS_PATTERN:{pattern}")
        return flags

    def _is_self_referential(self, code: str) -> bool:
        """Check if code references system internals."""
        code_lower = code.lower()
        for pattern in SELF_REFERENTIAL_PATTERNS:
            if re.search(pattern, code_lower):
                return True
        return False

    # Valid motor channels from the brain's motor cortex sub-ranges.
    _KNOWN_CHANNELS = {
        "locomotion", "manipulation", "head",
        "speech", "expression", "cognitive",
    }

    def _check_envelope(self, action: dict[str, Any]) -> Optional[str]:
        """Check if action violates safety envelopes."""
        # Motor command envelope
        if "intensity" in action:
            intensity = action["intensity"]
            if not isinstance(intensity, (int, float)):
                return "intensity must be numeric"
            if intensity < 0.0 or intensity > 1.0:
                return f"intensity {intensity} out of safe range [0.0, 1.0]"

        if "channel" in action:
            channel = action["channel"]
            if channel not in self._KNOWN_CHANNELS:
                return f"unknown motor channel: {channel}"

        # Legacy envelope checks
        if "angle" in action:
            angle = action["angle"]
            if not isinstance(angle, (int, float)):
                return "angle must be numeric"
            if angle < 0 or angle > 180:
                return f"angle {angle} out of safe range [0, 180]"

        if "speed" in action:
            speed = action["speed"]
            if not isinstance(speed, (int, float)):
                return "speed must be numeric"
            if speed < 0 or speed > 100:
                return f"speed {speed} out of safe range [0, 100]"

        return None

    def _generate_transformations(
        self,
        action: dict[str, Any],
        flags: list[str],
    ) -> Optional[list[dict[str, Any]]]:
        """Generate safe transformations for an action."""
        if not flags:
            return None

        transformed = action.copy()
        made_changes = False

        # Clamp motor intensity
        if "intensity" in transformed:
            intensity = transformed["intensity"]
            if isinstance(intensity, (int, float)):
                safe = max(0.0, min(1.0, float(intensity)))
                if safe != intensity:
                    transformed["intensity"] = safe
                    made_changes = True

        if "angle" in transformed:
            angle = transformed["angle"]
            if isinstance(angle, (int, float)):
                safe_angle = max(0, min(180, angle))
                if safe_angle != angle:
                    transformed["angle"] = safe_angle
                    made_changes = True

        if "speed" in transformed:
            speed = transformed["speed"]
            if isinstance(speed, (int, float)):
                safe_speed = max(0, min(100, speed))
                if safe_speed != speed:
                    transformed["speed"] = safe_speed
                    made_changes = True

        if made_changes:
            return [transformed]

        return None

    # ------------------------------------------------------------------
    # Body-profile enforcement
    # ------------------------------------------------------------------

    def _check_body_profile(
        self,
        action: dict[str, Any],
        flags: list[str],
    ) -> Optional[dict[str, Any]]:
        """Check action against body-profile capability markers and motor limits.

        Returns:
            None if no profile-level violation.
            A dict with 'type', 'reason', and optional 'transformations'/'risk_score'
            if the action should be DENIED or TRANSFORMED.
        """
        # Snapshot profile reference — safe even if set_body_profile() is called
        # between awaits in the Kernel service (asyncio cooperative scheduling).
        profile = self._body_profile
        if profile is None:
            return None

        channel = action.get("channel", "")
        intensity = action.get("intensity")
        action_type = action.get("type", "")

        # 1. Channel capability check — hard DENY if channel is disabled
        if channel and not profile.is_channel_allowed(channel):
            flags.append(f"PROFILE_DENY:{channel}")
            return {
                "type": DecisionType.DENY,
                "reason": (
                    f"Body profile '{profile.name}' disallows "
                    f"channel '{channel}'"
                ),
                "risk_score": 1.0,
            }

        # Cognitive channel check (action_type based)
        if action_type == "cognitive_query":
            if not profile.is_channel_allowed("cognitive"):
                flags.append("PROFILE_DENY:cognitive")
                return {
                    "type": DecisionType.DENY,
                    "reason": (
                        f"Body profile '{profile.name}' disallows "
                        f"cognitive queries"
                    ),
                    "risk_score": 1.0,
                }

        # 2. Motor-limit clamping — TRANSFORM if intensity exceeds profile max
        if channel and isinstance(intensity, (int, float)):
            limit = profile.get_motor_limit(channel)
            if intensity > limit.max_intensity:
                clamped = limit.max_intensity
                flags.append(f"PROFILE_CLAMP:{channel}:{intensity:.2f}->{clamped:.2f}")
                transformed = action.copy()
                transformed["intensity"] = clamped
                return {
                    "type": DecisionType.TRANSFORM,
                    "reason": (
                        f"Body profile '{profile.name}' caps "
                        f"'{channel}' at {clamped:.2f} (was {intensity:.2f})"
                    ),
                    "risk_score": 0.3,
                    "transformations": [transformed],
                }

        return None
