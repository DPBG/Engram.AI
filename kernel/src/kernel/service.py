"""
Kernel Service - The Moral Kernel event bus interface.

All proposals must pass through this service for evaluation.
Body profiles are loaded from BODY_PROFILE env var on startup.
"""

import asyncio
import json
import math
import os
import time
from collections import defaultdict, deque
from typing import Any

from activelearning import (
    BaseService,
    current_timestamp,
    generate_trace_id,
    sign_decision,
    verify_operator_action,
)
from activelearning.nats_client import serialize_message
from activelearning.subjects import (
    Subjects,
    code_decision_subject,
    decision_subject,
)
from nats.aio.msg import Msg

from kernel.evaluator import (
    DecisionType,
    KernelDecision,
    KernelEvaluator,
    RiskAnalysis,
    unavailable_risk_analysis,
)
from kernel.policy import (
    DecisionSequenceTracker,
    PolicyRollbackManager,
    validate_cognitive_response,
    validate_policy_update,
)

_DECISION_TYPES = ("ALLOW", "TRANSFORM", "DENY", "DEFER")

# Per-source code-proposal rate limiting (M1.14).
# Exceeding the limit is fail-closed: a DENY decision is published so the
# meta-programmer's Future resolves rather than hanging, and the over-rate
# source is logged.  Both values are overridable via environment variables.
_PROPOSAL_RATE_LIMIT = int(os.environ.get("KERNEL_PROPOSAL_RATE_LIMIT", "10"))
_PROPOSAL_RATE_WINDOW_S = float(os.environ.get("KERNEL_PROPOSAL_RATE_WINDOW_S", "60"))


def _empty_bucket() -> dict[str, Any]:
    """A zeroed decision-rate bucket: raw counts + normalized rates."""
    return {
        "total": 0,
        "counts": {t: 0 for t in _DECISION_TYPES},
        "rates": {t: 0.0 for t in _DECISION_TYPES},
    }


def _accumulate(bucket: dict[str, Any], decision_type: str, count: int) -> None:
    """Add a grouped SQL row's count into a bucket, ignoring unknown types."""
    if decision_type not in bucket["counts"]:
        return
    bucket["counts"][decision_type] += count
    bucket["total"] += count


def _finalize_rates(bucket: dict[str, Any]) -> None:
    """Convert accumulated counts into fractions of the bucket's total."""
    total = bucket["total"]
    if total <= 0:
        return
    bucket["rates"] = {t: round(c / total, 4) for t, c in bucket["counts"].items()}


class KernelService(BaseService):
    """
    Moral Kernel service.

    Subscribes to proposal.new and code.proposal events,
    evaluates proposals, and publishes decisions.
    """

    def __init__(self):
        super().__init__("kernel", use_database=True, use_event_bus=True)
        self._evaluator = KernelEvaluator()

        # Metrics
        self._allow_count = 0
        self._transform_count = 0
        self._deny_count = 0
        self._defer_count = 0

        # Active body profile name (for audit logging)
        self._body_profile: str | None = None

        # Policy management
        self._rollback = PolicyRollbackManager()
        self._deny_tracker = DecisionSequenceTracker()

        # Decision-rate governance signal (M6): periodic trend publish
        self._decision_rates_task: asyncio.Task | None = None
        self._decision_rates_interval_sec = float(
            os.environ.get("KERNEL_DECISION_RATES_INTERVAL_SEC", "300")
        )
        self._decision_rates_window_hours = float(
            os.environ.get("KERNEL_DECISION_RATES_WINDOW_HOURS", "24")
        )
        # Per-source proposal rate limiting (M1.14): sliding-window timestamp deques
        # keyed by source name.  Fail-closed: over-limit proposals get a DENY.
        self._proposal_timestamps: dict[str, deque] = defaultdict(deque)
        self._proposal_rate_limit = _PROPOSAL_RATE_LIMIT
        self._proposal_rate_window_s = _PROPOSAL_RATE_WINDOW_S

        # Heartbeat (E1.9.3): publish kernel.heartbeat so the watchdog can detect loss.
        # A non-positive or non-finite interval would collapse the publish cadence,
        # so we validate and refuse to start with an invalid configuration.
        import math as _math

        _raw_hb = float(os.environ.get("KERNEL_HEARTBEAT_INTERVAL_S", "5.0"))
        if not (_math.isfinite(_raw_hb) and _raw_hb > 0.0):
            raise ValueError(
                "KERNEL_HEARTBEAT_INTERVAL_S must be a positive finite number; "
                f"got {os.environ.get('KERNEL_HEARTBEAT_INTERVAL_S')!r}"
            )
        self._heartbeat_interval_s = _raw_hb
        self._heartbeat_task: asyncio.Task | None = None

        # Load body profile from env if set
        self._load_body_profile()

    def _load_body_profile(self) -> None:
        """Load body profile from BODY_PROFILE env var (if set).

        Called during __init__ (sync). The profile is validated and
        passed to the evaluator so all subsequent proposals are checked
        against it. If the env var is unset, the evaluator runs without
        profile constraints (base constitutional only).
        """
        profile_name = os.environ.get("BODY_PROFILE")
        if not profile_name:
            return

        try:
            from beliefs.profiles import load_profile

            profile = load_profile(profile_name)
            self._body_profile = profile.name
            self._evaluator.set_body_profile(profile)
            self.logger.info(f"Kernel loaded body profile: {profile.name}")
        except FileNotFoundError:
            self.logger.error(
                f"Body profile '{profile_name}' not found — " f"running without profile constraints"
            )
        except ValueError as e:
            self.logger.error(
                f"Body profile '{profile_name}' invalid: {e} — "
                f"running without profile constraints"
            )
        except ImportError:
            self.logger.warning("beliefs.profiles not available — " "running without body profile")

    async def _setup(self) -> None:
        """Service-specific setup."""
        # Subscribe to proposal events via durable JetStream consumers so that
        # proposals published while this service was restarting are not lost.
        await self.event_bus.js_subscribe(
            Subjects.PROPOSAL_NEW,
            self._handle_action_proposal,
            durable="kernel-action-proposals",
        )
        await self.event_bus.js_subscribe(
            Subjects.CODE_PROPOSAL,
            self._handle_code_proposal,
            durable="kernel-code-proposals",
        )
        await self.event_bus.subscribe(
            Subjects.KERNEL_STATUS,
            self._handle_status,
            is_request_handler=True,
        )
        await self.event_bus.subscribe(
            Subjects.POLICY_LOAD_PROFILE,
            self._handle_load_profile,
        )
        # Brain/dashboard may not publish policy.restrict directly (ADR 0001 §3).
        # They publish policy.restrict.request; the Kernel validates and re-publishes
        # as authoritative policy.restrict that consumers (planner, brain) act on.
        # Two separate calls: a single call with 4 positional args would silently
        # pass the second subject as the queue-group parameter and the second
        # handler as pending_msgs_limit, dropping the subscription entirely.
        await self.event_bus.subscribe(
            Subjects.POLICY_RESTRICT,
            self._handle_restrict,
        )
        await self.event_bus.subscribe(
            Subjects.POLICY_RESTRICT_REQUEST,
            self._handle_restrict_request,
        )
        await self.event_bus.subscribe(
            Subjects.POLICY_ROLLBACK,
            self._handle_rollback,
        )
        await self.event_bus.subscribe(
            Subjects.POLICY_UPDATE,
            self._handle_policy_update,
        )
        await self.event_bus.subscribe(
            Subjects.COGNITIVE_RESPONSE_VALIDATE,
            self._handle_cognitive_validate,
        )
        # SAFE_HALT kill switch (Phase 1.9)
        await self.event_bus.subscribe(Subjects.SAFETY_HALT, self._handle_safety_halt)
        await self.event_bus.subscribe(Subjects.SAFETY_RESUME, self._handle_safety_resume)

        # Decision-rate governance signal (M6): periodic trend publish
        self._decision_rates_task = asyncio.create_task(self._decision_rates_loop())
        # Heartbeat loop — lets the kernel-loss watchdog (E1.9.3) detect our death.
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        if self._decision_rates_task and not self._decision_rates_task.done():
            self._decision_rates_task.cancel()
            try:
                await self._decision_rates_task
            except asyncio.CancelledError:
                pass
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self) -> None:
        """Publish kernel.heartbeat every KERNEL_HEARTBEAT_INTERVAL_S seconds (E1.9.3)."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval_s)
                await self.event_bus.publish(
                    Subjects.KERNEL_HEARTBEAT,
                    {"status": "alive", "timestamp": current_timestamp()},
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning("Heartbeat publish error: %s", e)

    async def _handle_safety_halt(self, data: dict) -> None:
        """Engage the system-wide kill switch (Phase 1.9).

        Sets the evaluator to deny every proposal, then propagates the halt:
        the Planner is forced to SAFE_HALT (cancels its queue) and all motor
        channels are restricted to zero. Because every action and code
        proposal passes through the Kernel, the deny-all flag alone already
        stops new execution and deployment; the propagation halts in-flight
        motor output and planning too.
        """
        reason = data.get("reason", "operator SAFE_HALT")
        operator = data.get("operator_id", "unknown")
        self._evaluator.halt(reason)
        self.logger.critical(f"SAFE_HALT engaged by {operator}: {reason}")

        # Propagate: stop the planner queue and zero all motor channels.
        # Kernel applies restrictions directly (no NATS round-trip) because it is
        # the sole publisher of policy.restrict (ADR 0001 §3) and no longer
        # subscribes to it.
        try:
            await self.event_bus.publish(
                Subjects.PLANNER_MODE, {"mode": "SAFE_HALT", "reason": reason}
            )
            restrict_data = {
                "motor_limits": {
                    ch: {"max_intensity": 0.0}
                    for ch in ("locomotion", "manipulation", "head", "speech")
                },
                "reason": f"SAFE_HALT: {reason}",
                "operator_id": operator,
            }
            await self._apply_restriction(restrict_data)
            await self.event_bus.publish(Subjects.POLICY_RESTRICT, restrict_data)
        except Exception as e:
            self.logger.error(f"SAFE_HALT propagation error: {e}")

        await self.event_bus.publish(
            Subjects.SAFETY_HALT_STATUS,
            {
                "halted": True,
                "reason": reason,
                "operator_id": operator,
            },
        )

    async def _handle_safety_resume(self, data: dict) -> None:
        """Release the kill switch (operator action). Resumes evaluation.

        Resume is deliberately narrow: it re-enables the Kernel but does NOT
        auto-restore motor limits or planner mode — an operator must restore
        those explicitly, so the system never silently un-halts itself.

        The payload must carry a valid HMAC signature (``_op_sig``) and a
        fresh ``timestamp`` (epoch ms) when ``ENGRAM_OPERATOR_KEY`` is set.
        Fail-closed: an unauthenticated, tampered, or replayed resume is
        REJECTED and the system remains halted.
        """
        operator = data.get("operator_id", "unknown")

        if not verify_operator_action(data, "resume"):
            self.logger.critical(
                "SAFE_HALT resume REJECTED — unauthenticated, tampered, or "
                "replayed request (operator_id=%r). System remains halted.",
                operator,
            )
            # Confirm still-halted so consumers don't assume resumption occurred.
            await self.event_bus.publish(
                Subjects.SAFETY_HALT_STATUS,
                {
                    "halted": True,
                    "operator_id": operator,
                    "rejection_reason": "unauthenticated_resume",
                },
            )
            return

        self._evaluator.resume()
        self.logger.warning(f"SAFE_HALT released by {operator}")
        await self.event_bus.publish(
            Subjects.SAFETY_HALT_STATUS,
            {"halted": False, "operator_id": operator},
        )

    async def _handle_load_profile(self, data: dict) -> None:
        """Handle runtime profile load/switch via NATS.

        Expected payload: {"profile_name": "construction_heavy"}
        Publishes result to ``policy.profile.status``.
        """
        profile_name = data.get("profile_name", "")
        if not profile_name:
            await self.event_bus.publish(
                "policy.profile.status",
                {
                    "status": "error",
                    "reason": "Missing profile_name",
                },
            )
            return

        try:
            from beliefs.profiles import load_profile

            profile = load_profile(profile_name)

            # Ensure the OLD profile is in rollback history before switching.
            # On first switch, _current may be None — snapshot the old profile
            # to seed the rollback stack. On subsequent switches, the old
            # profile is already in _current and gets pushed to history
            # when we snapshot the new one.
            old_profile = self._evaluator._body_profile
            if old_profile is not None and self._rollback.current is None:
                self._rollback.snapshot(old_profile, reason=f"Initial: {old_profile.name}")

            self._body_profile = profile.name
            self._evaluator.set_body_profile(profile)
            self._rollback.snapshot(profile, reason=f"Loaded {profile_name}")

            self.logger.info(f"Runtime profile switch: {profile.name}")
            await self.event_bus.publish(
                "policy.profile.status",
                {
                    "status": "loaded",
                    "profile": profile.name,
                    "version": profile.version,
                    "capabilities": profile.capabilities,
                    "motor_limits": {
                        ch: {"max_intensity": lim.max_intensity}
                        for ch, lim in profile.motor_limits.items()
                    },
                },
            )
        except (FileNotFoundError, ValueError, ImportError) as e:
            self.logger.error(f"Failed to load profile '{profile_name}': {e}")
            await self.event_bus.publish(
                "policy.profile.status",
                {
                    "status": "error",
                    "reason": str(e),
                },
            )

    async def _handle_rollback(self, data: dict) -> None:
        """Revert to the last-known-good policy snapshot.

        Uses the deep-copied BodyProfile stored in the snapshot — no disk I/O,
        immune to profile file tampering between snapshot and rollback.

        Expected payload: {"reason": "..."}
        """
        snap = self._rollback.rollback()
        if snap is None:
            await self.event_bus.publish(
                "policy.rollback.status",
                {
                    "status": "error",
                    "reason": "No rollback history available",
                },
            )
            return

        try:
            profile = snap.profile
            self._body_profile = profile.name
            self._evaluator.set_body_profile(profile)
            self.logger.info(
                f"Policy rollback to '{snap.profile_name}' " f"(snapshot from {snap.reason})"
            )
            await self.event_bus.publish(
                "policy.rollback.status",
                {
                    "status": "rolled_back",
                    "profile": snap.profile_name,
                    "snapshot_reason": snap.reason,
                },
            )
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            await self.event_bus.publish(
                "policy.rollback.status",
                {
                    "status": "error",
                    "reason": str(e),
                },
            )

    async def _handle_policy_update(self, data: dict) -> None:
        """Handle validated policy updates from cloud.

        This is the primary cloud→edge policy channel. All updates
        are validated against the schema before applying.
        """
        valid, reason = validate_policy_update(data)
        if not valid:
            self.logger.warning(f"Policy update rejected: {reason}")
            await self.event_bus.publish(
                "policy.update.status",
                {
                    "status": "rejected",
                    "reason": reason,
                    "operator_id": data.get("operator_id", "unknown"),
                },
            )
            return

        # Apply profile switch if requested
        if "profile_name" in data:
            await self._handle_load_profile(data)

        # Apply restrictions if present
        if "motor_limits" in data or "capabilities" in data:
            await self._handle_restrict(data)

        self.logger.info(
            f"Policy update applied by {data.get('operator_id', 'unknown')}: "
            f"{data.get('reason', 'no reason')}"
        )

    async def _handle_cognitive_validate(self, data: dict) -> None:
        """Validate cognitive (LLM) response before sensory injection.

        Expected payload: {"response_text": "...", "trace_id": "..."}
        Publishes to cognitive.response.validated or cognitive.response.rejected.
        """
        response_text = data.get("response_text", "")
        trace_id = data.get("trace_id", "")

        if not trace_id:
            self.logger.warning(
                "Cognitive validation: missing trace_id — response cannot be correlated"
            )

        valid, reason = validate_cognitive_response(
            response_text,
            self._evaluator._body_profile,
        )

        if valid:
            await self.event_bus.publish(
                "cognitive.response.validated",
                {
                    "trace_id": trace_id,
                    "response_text": response_text,
                    "query_step": data.get("query_step", 0),
                    "prediction_error": data.get("prediction_error", 0.0),
                    "model": data.get("model", ""),
                },
            )
        else:
            self.logger.warning(f"Cognitive response rejected (trace={trace_id}): {reason}")
            await self.event_bus.publish(
                "cognitive.response.rejected",
                {
                    "trace_id": trace_id,
                    "reason": reason,
                },
            )

    async def _apply_restriction(self, data: dict) -> bool:
        """Apply motor/capability restrictions to the kernel's body profile.

        Updates the evaluator's profile in-place so future proposals are checked
        against the new limits. Publishes ``policy.restrict.status`` for the dashboard.

        Returns True if restrictions were applied, False if rejected/error.
        """
        if self._evaluator._body_profile is None:
            await self.event_bus.publish(
                "policy.restrict.status",
                {
                    "status": "error",
                    "reason": "No body profile loaded — cannot apply restrictions",
                },
            )
            return False

        try:
            from beliefs.profiles import apply_runtime_restrictions

            restricted = apply_runtime_restrictions(
                self._evaluator._body_profile,
                data,
            )
            self._evaluator.set_body_profile(restricted)
            self.logger.info("Applied runtime restrictions to body profile")
            await self.event_bus.publish(
                "policy.restrict.status",
                {
                    "status": "applied",
                    "profile": restricted.name,
                    "motor_limits": {
                        ch: {"max_intensity": lim.max_intensity}
                        for ch, lim in restricted.motor_limits.items()
                    },
                    "capabilities": restricted.capabilities,
                },
            )
            return True
        except ValueError as e:
            self.logger.warning(f"Runtime restriction rejected: {e}")
            await self.event_bus.publish(
                "policy.restrict.status",
                {
                    "status": "rejected",
                    "reason": str(e),
                },
            )
            return False
        except Exception as e:
            self.logger.error(f"Error applying restrictions: {e}")
            await self.event_bus.publish(
                "policy.restrict.status",
                {
                    "status": "error",
                    "reason": str(e),
                },
            )
            return False

    async def _handle_restrict_request(self, data: dict) -> None:
        """Relay a policy.restrict.request from brain/dashboard as authoritative policy.restrict.

        The brain and dashboard may not publish ``policy.restrict`` directly (ADR 0001 §3);
        they publish ``policy.restrict.request`` and the Kernel — as the sole
        decision authority — validates, applies internally, and re-publishes the
        authoritative ``policy.restrict`` that consumers (planner, brain) act on.
        """
        if await self._apply_restriction(data):
            await self.event_bus.publish(Subjects.POLICY_RESTRICT, data)

    async def _handle_restrict(self, data: dict) -> None:
        """Apply a restriction coming from an internal Kernel operation.

        Called by _handle_policy_update when the operator update contains
        motor_limits or capabilities. Applies the restriction to the evaluator
        and broadcasts the authoritative policy.restrict.
        """
        if await self._apply_restriction(data):
            await self.event_bus.publish(Subjects.POLICY_RESTRICT, data)

    async def _handle_action_proposal(self, data: dict) -> None:
        """Handle action proposals from Planner or Neuromorphic brain."""
        proposal = data
        trace_id = proposal.get("trace_id", "")
        source = proposal.get("provenance", proposal.get("source", "unknown"))
        action = proposal.get("action", {})

        # Determine proposal_type from action content
        action_type = action.get("type", "")
        if action_type == "cognitive_query":
            proposal_type = "cognitive"
        elif action.get("channel") == "speech":
            proposal_type = "speech"
        else:
            proposal_type = "action"

        try:
            self.logger.debug(f"Evaluating {proposal_type} proposal: {trace_id}")

            # Validate proposal has required fields
            if "action" not in proposal:
                self.logger.warning(f"Proposal {trace_id} missing 'action' key — denying")
                deny = KernelDecision(
                    trace_id=trace_id,
                    type=DecisionType.DENY,
                    reason="Malformed proposal: missing 'action' key",
                    risk_score=1.0,
                )
                await self._publish_and_log_decision(
                    trace_id,
                    proposal_type,
                    source,
                    deny,
                )
                self._deny_count += 1
                return

            # Get risk analysis from Safety Supervisor (if available)
            risk_analysis = await self._get_risk_analysis(proposal)

            # Check proposal against belief norms (if Beliefs service available)
            norm_violations = await self._check_belief_norms(proposal)

            # Evaluate
            decision = self._evaluator.evaluate_action_proposal(
                proposal, risk_analysis, norm_violations
            )

            # Update metrics
            self._update_metrics(decision.type)

            # Collect flags for audit
            flags = list(risk_analysis.flags) if risk_analysis else []

            # Log and publish (pass original proposal for forwarding on ALLOW)
            await self._publish_and_log_decision(
                trace_id,
                proposal_type,
                source,
                decision,
                flags=flags,
                norm_violations=norm_violations,
                original_proposal=proposal,
            )

            self.logger.info(
                f"{proposal_type.capitalize()} {trace_id}: {decision.type.value}"
                f" (risk={decision.risk_score:.2f})"
            )

        except Exception as e:
            self.logger.error(f"Error handling action proposal: {e}")
            # Always publish a decision so the caller's Future doesn't hang.
            # Fail-safe: DENY on internal errors.
            try:
                deny = KernelDecision(
                    trace_id=trace_id,
                    type=DecisionType.DENY,
                    reason=f"Kernel internal error: {e}",
                    risk_score=1.0,
                )
                await self._publish_and_log_decision(
                    trace_id,
                    proposal_type,
                    source,
                    deny,
                )
                self._deny_count += 1
            except Exception:
                # Best-effort DENY publish. If this also fails, no decision reaches
                # the stream; callers time out and fail closed — verified by
                # kernel/tests/test_service.py::
                #   test_action_proposal_recovery_publish_failure_caller_fails_closed
                #   test_code_proposal_recovery_publish_failure_caller_fails_closed
                pass  # caller will timeout

    def _signed_code_decision(self, decision: KernelDecision) -> dict:
        """Build the signed wire payload for a code decision."""
        return sign_decision(
            {
                "trace_id": decision.trace_id,
                "type": decision.type.value,
                "reason": decision.reason,
                "risk_score": decision.risk_score,
                "issued_at": decision.issued_at,
                "expires_at": decision.expires_at,
            }
        )

    def _check_proposal_rate_limit(self, source: str) -> bool:
        """Sliding-window rate check for a given source.

        Records the current timestamp and returns True if the source is within
        its budget, False if it has exceeded _proposal_rate_limit proposals in
        the last _proposal_rate_window_s seconds.  The window is maintained as
        a deque of monotonic timestamps; stale entries are evicted on each call.
        """
        now = time.monotonic()
        timestamps = self._proposal_timestamps[source]
        cutoff = now - self._proposal_rate_window_s
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if len(timestamps) >= self._proposal_rate_limit:
            return False
        timestamps.append(now)
        return True

    async def _handle_code_proposal(self, data: dict) -> None:
        """Handle code proposals from Meta-Programmer."""
        proposal = data
        trace_id = proposal.get("trace_id", "")
        source = proposal.get("source", "meta-programmer")
        try:
            self.logger.debug(f"Evaluating code proposal: {trace_id}")

            # Rate-limit check (M1.14): fail-closed DENY — never silently drop.
            if not self._check_proposal_rate_limit(source):
                self.logger.warning(
                    f"Rate limit exceeded for source '{source}' "
                    f"(>{self._proposal_rate_limit} proposals/"
                    f"{self._proposal_rate_window_s:.0f}s) — denying {trace_id}"
                )
                deny = KernelDecision(
                    trace_id=trace_id,
                    type=DecisionType.DENY,
                    reason=(
                        f"Rate limit exceeded: source '{source}' sent more than "
                        f"{self._proposal_rate_limit} proposals in "
                        f"{self._proposal_rate_window_s:.0f}s"
                    ),
                    risk_score=1.0,
                )
                await self._log_decision(trace_id, "code", source, deny)
                await self.event_bus.publish(
                    code_decision_subject(trace_id),
                    self._signed_code_decision(deny),
                )
                self._deny_count += 1
                return

            # Get risk analysis from Safety Supervisor
            risk_analysis = await self._get_risk_analysis(proposal, is_code=True)

            # Evaluate
            decision = self._evaluator.evaluate_code_proposal(proposal, risk_analysis)

            # Update metrics
            self._update_metrics(decision.type)

            flags = list(risk_analysis.flags) if risk_analysis else []

            # Log and publish (code uses different NATS subject)
            await self._log_decision(
                trace_id,
                "code",
                source,
                decision,
                flags=flags,
            )
            await self.event_bus.publish(
                code_decision_subject(trace_id),
                sign_decision(
                    {
                        "trace_id": decision.trace_id,
                        "type": decision.type.value,
                        "reason": decision.reason,
                        "risk_score": decision.risk_score,
                        "issued_at": decision.issued_at,
                        "expires_at": decision.expires_at,
                    }
                ),
                self._signed_code_decision(decision),
            )

            self.logger.info(
                f"Code {trace_id}: {decision.type.value}" f" (risk={decision.risk_score:.2f})"
            )

        except Exception as e:
            self.logger.error(f"Error handling code proposal: {e}")
            # Always publish a decision so the meta-programmer's Future doesn't hang.
            # Fail-safe: DENY on internal errors.
            try:
                deny = KernelDecision(
                    trace_id=trace_id,
                    type=DecisionType.DENY,
                    reason=f"Kernel internal error: {e}",
                    risk_score=1.0,
                )
                await self._log_decision(trace_id, "code", source, deny)
                await self.event_bus.publish(
                    code_decision_subject(trace_id),
                    self._signed_code_decision(deny),
                )
                self._deny_count += 1
            except Exception:
                # Best-effort DENY publish. If this also fails, no decision reaches
                # the stream; callers time out and fail closed — verified by
                # kernel/tests/test_service.py::
                #   test_action_proposal_recovery_publish_failure_caller_fails_closed
                #   test_code_proposal_recovery_publish_failure_caller_fails_closed
                pass  # caller will timeout

    async def _handle_status(self, _data: dict, msg: Msg) -> None:
        """Reply to status requests via request-reply."""
        status = {
            "status": "running",
            "body_profile": self._body_profile,
            "has_rollback": self._rollback.has_rollback,
            "deny_sequences": self._deny_tracker.get_state(),
            "metrics": {
                "allow_count": self._allow_count,
                "transform_count": self._transform_count,
                "deny_count": self._deny_count,
                "defer_count": self._defer_count,
            },
        }
        if msg.reply:
            await msg.respond(serialize_message(status))

    # ── Decision-rate governance signal (M6) ──────────────────────────────
    #
    # Longitudinal ALLOW/TRANSFORM/DENY/DEFER rates from the kernel_decisions
    # audit trail, broken out by source (meta-programmer, neuromorphic,
    # planner, ...). This is a *visibility* signal only — it never feeds back
    # into evaluate_*_proposal, so it cannot weaken the gate (CLAUDE.md §3).
    # It exists to catch drift in generated-code/action quality before any
    # future work expands meta-programmer autonomy.

    async def _decision_rates_loop(self) -> None:
        """Periodically publish decision-rate stats to KERNEL_DECISION_RATES."""
        while True:
            try:
                await asyncio.sleep(self._decision_rates_interval_sec)
                rates = await self._compute_decision_rates()
                await self.event_bus.publish(Subjects.KERNEL_DECISION_RATES, rates)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning(f"Decision-rate publish failed: {e}")

    async def _compute_decision_rates(self) -> dict[str, Any]:
        """Aggregate kernel_decisions into all-time + windowed rates by source.

        Best-effort: returns an empty-but-well-formed structure on any DB
        error rather than raising, since this must never affect the
        request/response path that calls it (status, periodic publish).
        """
        empty = {
            "generated_at": int(time.time() * 1000),
            "window_hours": self._decision_rates_window_hours,
            "all_time": _empty_bucket(),
            "window": _empty_bucket(),
            "by_source": {},
        }
        if self.database is None:
            return empty

        try:
            all_time_rows = await self.database.fetchall(
                "SELECT decision_type, source, COUNT(*) as cnt "
                "FROM kernel_decisions GROUP BY decision_type, source",
            )
            window_start_ms = int(time.time() * 1000) - int(
                self._decision_rates_window_hours * 3600 * 1000
            )
            window_rows = await self.database.fetchall(
                "SELECT decision_type, source, COUNT(*) as cnt "
                "FROM kernel_decisions WHERE issued_at >= ? "
                "GROUP BY decision_type, source",
                (window_start_ms,),
            )
        except Exception as e:
            self.logger.warning(f"Could not compute decision rates: {e}")
            return empty

        by_source: dict[str, Any] = {}
        all_time_bucket = _empty_bucket()
        window_bucket = _empty_bucket()

        for row in all_time_rows:
            _accumulate(all_time_bucket, row["decision_type"], row["cnt"])
            source_bucket = by_source.setdefault(
                row["source"] or "unknown",
                {"all_time": _empty_bucket(), "window": _empty_bucket()},
            )
            _accumulate(source_bucket["all_time"], row["decision_type"], row["cnt"])

        for row in window_rows:
            _accumulate(window_bucket, row["decision_type"], row["cnt"])
            source_bucket = by_source.setdefault(
                row["source"] or "unknown",
                {"all_time": _empty_bucket(), "window": _empty_bucket()},
            )
            _accumulate(source_bucket["window"], row["decision_type"], row["cnt"])

        for bucket in [
            all_time_bucket,
            window_bucket,
            *(b for s in by_source.values() for b in s.values()),
        ]:
            _finalize_rates(bucket)

        return {
            "generated_at": int(time.time() * 1000),
            "window_hours": self._decision_rates_window_hours,
            "all_time": all_time_bucket,
            "window": window_bucket,
            "by_source": by_source,
        }

    async def _get_risk_analysis(
        self,
        proposal: dict,
        is_code: bool = False,
    ) -> RiskAnalysis | None:
        """Request risk analysis from Safety Supervisor."""
        trace_id = proposal.get("trace_id", "")

        def _unavailable(reason: str) -> RiskAnalysis:
            self.logger.warning(
                "Safety analysis unavailable for %s: %s — failing closed",
                trace_id,
                reason,
            )
            return unavailable_risk_analysis(trace_id)

        try:
            # Request analysis from Safety Supervisor
            subject = Subjects.SAFETY_ANALYZE_CODE if is_code else Subjects.SAFETY_ANALYZE_ACTION

            response = await self.event_bus.request(
                subject,
                proposal,
                timeout=5.0,
            )

            data = response
            if data.get("type") == "error":
                return _unavailable(data.get("error", "unknown"))

            raw_score = data.get("risk_score")
            if raw_score is None:
                return _unavailable("missing risk_score")
            try:
                risk_score = float(raw_score)
            except (TypeError, ValueError):
                return _unavailable(f"invalid risk_score: {raw_score!r}")
            if not math.isfinite(risk_score):
                return _unavailable(f"non-finite risk_score: {raw_score!r}")

            flags = data.get("flags", [])
            if not isinstance(flags, list):
                return _unavailable("invalid flags: expected list")

            recommendations = data.get("recommendations", [])
            if not isinstance(recommendations, list):
                recommendations = []

            return RiskAnalysis(
                trace_id=data.get("trace_id", trace_id),
                risk_score=risk_score,
                flags=flags,
                recommendations=recommendations,
            )
        except Exception as e:
            return _unavailable(str(e))

    async def _check_belief_norms(
        self,
        proposal: dict,
    ) -> list[dict[str, Any]]:
        """Query Beliefs service for norm violations relevant to this proposal.

        Returns a list of violated norms with risk_boost values.
        Returns empty list if Beliefs service is unavailable.
        """
        try:
            response = await self.event_bus.request(
                Subjects.BELIEFS_QUERY_REQUEST,
                {"type": "norms", "threshold": 0.8},
                timeout=2.0,
            )
            # Check for error response (from EventBus error-reply handler)
            if response.get("type") == "error":
                self.logger.warning(f"Beliefs service error: {response.get('error', 'unknown')}")
                return []
            norms = response.get("result", [])
            if not norms:
                return []

            action = proposal.get("action", {})
            channel = action.get("channel", "")
            intensity = action.get("intensity", 0.0)
            violations = []

            for norm in norms:
                norm_id = norm.get("id", "")
                metadata = norm.get("metadata", {})

                # Check: gradual_motor — intensity jumps > max_intensity_delta
                if norm_id == "norm.gradual_motor":
                    _max_delta = metadata.get("max_intensity_delta", 0.3)
                    # For now, flag high-intensity commands as potential violations.
                    # Future: track per-channel last intensity for delta checking.
                    if isinstance(intensity, (int, float)) and intensity > 0.85:
                        violations.append(
                            {
                                "norm_id": norm_id,
                                "content": norm.get("content", ""),
                                "risk_boost": 0.15,
                            }
                        )

                # Check: slow_novel — intensity cap in unfamiliar environments
                if norm_id == "norm.slow_novel":
                    _max_novel = metadata.get("max_intensity_novel", 0.5)
                    # Future: check if environment is "novel" via beliefs.
                    # For now, this norm is advisory — no automatic violation.

                # Check: force_limit — specific motor channels
                if norm_id == "norm.force_limit":
                    limited_channels = metadata.get("motor_channels", [])
                    if (
                        channel in limited_channels
                        and isinstance(intensity, (int, float))
                        and intensity > 0.9
                    ):
                        violations.append(
                            {
                                "norm_id": norm_id,
                                "content": norm.get("content", ""),
                                "risk_boost": 0.2,
                            }
                        )

            return violations
        except Exception as e:
            self.logger.warning(f"Could not check belief norms (proceeding without): {e}")
            return []

    async def _publish_and_log_decision(
        self,
        trace_id: str,
        proposal_type: str,
        source: str,
        decision: KernelDecision,
        flags: list[str] | None = None,
        norm_violations: list[dict[str, Any]] | None = None,
        original_proposal: dict[str, Any] | None = None,
    ) -> None:
        """Publish decision to NATS and log to SQLite audit trail.

        Combines publish + log in one call to prevent inconsistency
        (decision published but not logged, or vice versa).
        For cognitive/speech proposals that are ALLOWED, forwards to
        the appropriate execution channel.
        """
        decision_payload = sign_decision(
            {
                "trace_id": decision.trace_id,
                "type": decision.type.value,
                "reason": decision.reason,
                "transformations": decision.transformations,
                "risk_score": decision.risk_score,
                "issued_at": decision.issued_at,
                "expires_at": decision.expires_at,
            }
        )

        # Publish decision (signed) — caller is waiting on this and will
        # reject it unless the signature verifies.
        await self.event_bus.publish(decision_subject(trace_id), decision_payload)

        # Track consecutive DENYs per channel for escalation
        channel = ""
        if original_proposal:
            channel = original_proposal.get("action", {}).get("channel", "")
        if channel:
            escalation = self._deny_tracker.record_decision(
                channel,
                decision.type.value,
            )
            if escalation == "disable":
                await self._auto_disable_channel(channel)
            elif escalation == "escalate":
                await self.event_bus.publish(
                    "safety.deny_escalation",
                    {
                        "channel": channel,
                        "deny_count": self._deny_tracker._sequences[channel].count,
                        "action": "escalate",
                        "operator_id": "system:deny_tracker",
                    },
                )

        # On ALLOW: forward cognitive/speech to their execution channels.
        # This keeps the Kernel as the single gatekeeper — downstream services
        # subscribe to Kernel-gated subjects, not raw brain output.
        if decision.type == DecisionType.ALLOW and original_proposal:
            await self._forward_allowed_proposal(
                proposal_type,
                trace_id,
                original_proposal,
            )

        # Log to audit trail (best-effort, don't block on failure)
        await self._log_decision(
            trace_id,
            proposal_type,
            source,
            decision,
            flags=flags,
            norm_violations=norm_violations,
        )

    async def _forward_allowed_proposal(
        self,
        proposal_type: str,
        trace_id: str,
        proposal: dict[str, Any],
    ) -> None:
        """Forward ALLOWED proposals to their execution channels.

        Cognitive → cognitive.execute (bridge picks up)
        Speech → speech.execute (TTS/dashboard picks up)
        """
        try:
            if proposal_type == "cognitive":
                metadata = proposal.get("metadata", {})
                await self.event_bus.publish(
                    "cognitive.execute",
                    {
                        "trace_id": trace_id,
                        "prediction_error": proposal.get("action", {}).get("prediction_error", 0.0),
                        "intensity": proposal.get("action", {}).get("intensity", 0.0),
                        "step": metadata.get("step", 0),
                        "drives": metadata.get("drives", {}),
                    },
                )
            elif proposal_type == "speech":
                action = proposal.get("action", {})
                await self.event_bus.publish(
                    "speech.execute",
                    {
                        "trace_id": trace_id,
                        "token_idx": action.get("token_idx", 0),
                        "confidence": action.get("confidence", 0.0),
                        "intensity": action.get("intensity", 0.0),
                        "step": proposal.get("metadata", {}).get("step", 0),
                        "output_number": action.get("output_number", 0),
                    },
                )
        except Exception as e:
            self.logger.warning(f"Failed to forward {proposal_type} proposal: {e}")

    async def _log_decision(
        self,
        trace_id: str,
        proposal_type: str,
        source: str,
        decision: KernelDecision,
        flags: list[str] | None = None,
        norm_violations: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log decision to SQLite audit trail."""
        try:
            await self.database.insert(
                "kernel_decisions",
                {
                    "id": generate_trace_id(),
                    "trace_id": trace_id,
                    "proposal_type": proposal_type,
                    "decision_type": decision.type.value,
                    "source": source,
                    "reason": decision.reason,
                    "risk_score": decision.risk_score,
                    "flags": json.dumps(flags) if flags else None,
                    "norm_violations": (
                        json.dumps([v.get("norm_id", "") for v in norm_violations])
                        if norm_violations
                        else None
                    ),
                    "body_profile": self._body_profile,
                    "issued_at": decision.issued_at,
                    "expires_at": decision.expires_at,
                },
            )
        except Exception as e:
            self.logger.error(f"Error logging decision: {e}")

    def _update_metrics(self, decision_type: DecisionType) -> None:
        """Update decision metrics."""
        if decision_type == DecisionType.ALLOW:
            self._allow_count += 1
        elif decision_type == DecisionType.TRANSFORM:
            self._transform_count += 1
        elif decision_type == DecisionType.DENY:
            self._deny_count += 1
        elif decision_type == DecisionType.DEFER:
            self._defer_count += 1

    async def _auto_disable_channel(self, channel: str) -> None:
        """Auto-disable a motor channel after too many consecutive DENYs.

        This is an automated safety response — if the brain keeps
        proposing actions on a channel that are consistently denied,
        we reduce the channel to zero intensity to stop wasting
        evaluation cycles and prevent potential safety issues.
        """
        # Guard: skip if channel is already at 0 intensity
        if self._evaluator._body_profile is not None:
            limit = self._evaluator._body_profile.get_motor_limit(channel)
            if limit.max_intensity <= 0.0:
                return

        self.logger.error(
            f"Auto-disabling channel '{channel}' after "
            f"{DecisionSequenceTracker.DISABLE_THRESHOLD} consecutive DENYs"
        )
        try:
            await self.event_bus.publish(
                Subjects.POLICY_RESTRICT,
                {
                    "motor_limits": {channel: {"max_intensity": 0.0}},
                    "reason": f"Auto-disable: {channel} consecutive DENY threshold",
                    "operator_id": "system:deny_tracker",
                },
            )
            await self.event_bus.publish(
                "safety.deny_escalation",
                {
                    "channel": channel,
                    "action": "auto_disabled",
                    "deny_count": self._deny_tracker._sequences[channel].count,
                    "operator_id": "system:deny_tracker",
                },
            )
            restrict_data = {
                "motor_limits": {channel: {"max_intensity": 0.0}},
                "reason": f"Auto-disable: {channel} consecutive DENY threshold",
                "operator_id": "system:deny_tracker",
            }
            # Apply to kernel evaluator directly (Kernel is the sole policy.restrict
            # publisher so it cannot rely on a subscription round-trip).
            await self._apply_restriction(restrict_data)
            await self.event_bus.publish(Subjects.POLICY_RESTRICT, restrict_data)
            await self.event_bus.publish(
                "safety.deny_escalation",
                {
                    "channel": channel,
                    "action": "auto_disabled",
                    "deny_count": self._deny_tracker._sequences[channel].count,
                    "operator_id": "system:deny_tracker",
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to auto-disable channel '{channel}': {e}")


async def main() -> None:
    """Main entry point."""
    service = KernelService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
