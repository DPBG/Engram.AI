"""
Kernel Service - The Moral Kernel event bus interface.

All proposals must pass through this service for evaluation.
Body profiles are loaded from BODY_PROFILE env var on startup.
"""

import asyncio
import json
import os
import uuid
from typing import Any, Optional

from activelearning import BaseService, sign_decision
from activelearning.subjects import (
    Subjects,
    code_decision_subject,
    decision_subject,
)

from kernel.evaluator import KernelEvaluator, KernelDecision, RiskAnalysis, DecisionType
from kernel.policy import (
    PolicyRollbackManager,
    DecisionSequenceTracker,
    validate_cognitive_response,
    validate_policy_update,
)


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
        self._body_profile: Optional[str] = None

        # Policy management
        self._rollback = PolicyRollbackManager()
        self._deny_tracker = DecisionSequenceTracker()

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
                f"Body profile '{profile_name}' not found — "
                f"running without profile constraints"
            )
        except ValueError as e:
            self.logger.error(
                f"Body profile '{profile_name}' invalid: {e} — "
                f"running without profile constraints"
            )
        except ImportError:
            self.logger.warning(
                "beliefs.profiles not available — "
                "running without body profile"
            )

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
        await self.event_bus.subscribe(Subjects.KERNEL_STATUS, self._handle_status)
        await self.event_bus.subscribe(
            Subjects.POLICY_LOAD_PROFILE, self._handle_load_profile,
        )
        await self.event_bus.subscribe(
            Subjects.POLICY_RESTRICT, self._handle_restrict,
        )
        await self.event_bus.subscribe(
            Subjects.POLICY_ROLLBACK, self._handle_rollback,
        )
        await self.event_bus.subscribe(
            Subjects.POLICY_UPDATE, self._handle_policy_update,
        )
        await self.event_bus.subscribe(
            Subjects.COGNITIVE_RESPONSE_VALIDATE, self._handle_cognitive_validate,
        )
        # SAFE_HALT kill switch (Phase 1.9)
        await self.event_bus.subscribe(Subjects.SAFETY_HALT, self._handle_safety_halt)
        await self.event_bus.subscribe(Subjects.SAFETY_RESUME, self._handle_safety_resume)

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        # No kernel-specific resources to cleanup
        pass

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
        try:
            await self.event_bus.publish(Subjects.PLANNER_MODE, {"mode": "SAFE_HALT", "reason": reason})
            await self.event_bus.publish(Subjects.POLICY_RESTRICT, {
                "motor_limits": {
                    ch: {"max_intensity": 0.0}
                    for ch in ("locomotion", "manipulation", "head", "speech")
                },
                "reason": f"SAFE_HALT: {reason}",
                "operator_id": operator,
            })
        except Exception as e:
            self.logger.error(f"SAFE_HALT propagation error: {e}")

        await self.event_bus.publish(Subjects.SAFETY_HALT_STATUS, {
            "halted": True, "reason": reason, "operator_id": operator,
        })

    async def _handle_safety_resume(self, data: dict) -> None:
        """Release the kill switch (operator action). Resumes evaluation.

        Resume is deliberately narrow: it re-enables the Kernel but does NOT
        auto-restore motor limits or planner mode — an operator must restore
        those explicitly, so the system never silently un-halts itself.
        """
        operator = data.get("operator_id", "unknown")
        self._evaluator.resume()
        self.logger.warning(f"SAFE_HALT released by {operator}")
        await self.event_bus.publish(Subjects.SAFETY_HALT_STATUS, {
            "halted": False, "operator_id": operator,
        })

    async def _handle_load_profile(self, data: dict) -> None:
        """Handle runtime profile load/switch via NATS.

        Expected payload: {"profile_name": "construction_heavy"}
        Publishes result to ``policy.profile.status``.
        """
        profile_name = data.get("profile_name", "")
        if not profile_name:
            await self.event_bus.publish("policy.profile.status", {
                "status": "error", "reason": "Missing profile_name",
            })
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
            await self.event_bus.publish("policy.profile.status", {
                "status": "loaded",
                "profile": profile.name,
                "version": profile.version,
                "capabilities": profile.capabilities,
                "motor_limits": {
                    ch: {"max_intensity": lim.max_intensity}
                    for ch, lim in profile.motor_limits.items()
                },
            })
        except (FileNotFoundError, ValueError, ImportError) as e:
            self.logger.error(f"Failed to load profile '{profile_name}': {e}")
            await self.event_bus.publish("policy.profile.status", {
                "status": "error", "reason": str(e),
            })

    async def _handle_rollback(self, data: dict) -> None:
        """Revert to the last-known-good policy snapshot.

        Uses the deep-copied BodyProfile stored in the snapshot — no disk I/O,
        immune to profile file tampering between snapshot and rollback.

        Expected payload: {"reason": "..."}
        """
        snap = self._rollback.rollback()
        if snap is None:
            await self.event_bus.publish("policy.rollback.status", {
                "status": "error", "reason": "No rollback history available",
            })
            return

        try:
            profile = snap.profile
            self._body_profile = profile.name
            self._evaluator.set_body_profile(profile)
            self.logger.info(
                f"Policy rollback to '{snap.profile_name}' "
                f"(snapshot from {snap.reason})"
            )
            await self.event_bus.publish("policy.rollback.status", {
                "status": "rolled_back",
                "profile": snap.profile_name,
                "snapshot_reason": snap.reason,
            })
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            await self.event_bus.publish("policy.rollback.status", {
                "status": "error", "reason": str(e),
            })

    async def _handle_policy_update(self, data: dict) -> None:
        """Handle validated policy updates from cloud.

        This is the primary cloud→edge policy channel. All updates
        are validated against the schema before applying.
        """
        valid, reason = validate_policy_update(data)
        if not valid:
            self.logger.warning(f"Policy update rejected: {reason}")
            await self.event_bus.publish("policy.update.status", {
                "status": "rejected", "reason": reason,
                "operator_id": data.get("operator_id", "unknown"),
            })
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
            self.logger.warning("Cognitive validation: missing trace_id — response cannot be correlated")

        valid, reason = validate_cognitive_response(
            response_text, self._evaluator._body_profile,
        )

        if valid:
            await self.event_bus.publish("cognitive.response.validated", {
                "trace_id": trace_id,
                "response_text": response_text,
                "query_step": data.get("query_step", 0),
                "prediction_error": data.get("prediction_error", 0.0),
                "model": data.get("model", ""),
            })
        else:
            self.logger.warning(
                f"Cognitive response rejected (trace={trace_id}): {reason}"
            )
            await self.event_bus.publish("cognitive.response.rejected", {
                "trace_id": trace_id,
                "reason": reason,
            })

    async def _handle_restrict(self, data: dict) -> None:
        """Handle runtime capability restrictions via NATS (cloud → edge).

        Expected payload: {"motor_limits": {...}, "capabilities": {...}}
        Applies restrictions on top of the active body profile.
        Rejects any override that attempts to expand beyond the profile.
        """
        if self._evaluator._body_profile is None:
            await self.event_bus.publish("policy.restrict.status", {
                "status": "error",
                "reason": "No body profile loaded — cannot apply restrictions",
            })
            return

        try:
            from beliefs.profiles import apply_runtime_restrictions
            restricted = apply_runtime_restrictions(
                self._evaluator._body_profile, data,
            )
            self._evaluator.set_body_profile(restricted)
            self.logger.info("Applied runtime restrictions to body profile")
            await self.event_bus.publish("policy.restrict.status", {
                "status": "applied",
                "profile": restricted.name,
                "motor_limits": {
                    ch: {"max_intensity": lim.max_intensity}
                    for ch, lim in restricted.motor_limits.items()
                },
                "capabilities": restricted.capabilities,
            })
        except ValueError as e:
            self.logger.warning(f"Runtime restriction rejected: {e}")
            await self.event_bus.publish("policy.restrict.status", {
                "status": "rejected", "reason": str(e),
            })
        except Exception as e:
            self.logger.error(f"Error applying restrictions: {e}")
            await self.event_bus.publish("policy.restrict.status", {
                "status": "error", "reason": str(e),
            })

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
                    trace_id, proposal_type, source, deny,
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
                trace_id, proposal_type, source, decision,
                flags=flags, norm_violations=norm_violations,
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
                    trace_id, proposal_type, source, deny,
                )
                self._deny_count += 1
            except Exception:
                pass  # best-effort — caller will timeout

    async def _handle_code_proposal(self, data: dict) -> None:
        """Handle code proposals from Meta-Programmer."""
        try:
            proposal = data
            trace_id = proposal.get("trace_id", "")
            source = proposal.get("source", "meta-programmer")

            self.logger.debug(f"Evaluating code proposal: {trace_id}")

            # Get risk analysis from Safety Supervisor
            risk_analysis = await self._get_risk_analysis(proposal, is_code=True)

            # Evaluate
            decision = self._evaluator.evaluate_code_proposal(proposal, risk_analysis)

            # Update metrics
            self._update_metrics(decision.type)

            flags = list(risk_analysis.flags) if risk_analysis else []

            # Log and publish (code uses different NATS subject)
            await self._log_decision(
                trace_id, "code", source, decision,
                flags=flags,
            )
            await self.event_bus.publish(
                code_decision_subject(trace_id),
                sign_decision({
                    "trace_id": decision.trace_id,
                    "type": decision.type.value,
                    "reason": decision.reason,
                    "risk_score": decision.risk_score,
                    "issued_at": decision.issued_at,
                    "expires_at": decision.expires_at,
                }),
            )

            self.logger.info(
                f"Code {trace_id}: {decision.type.value}"
                f" (risk={decision.risk_score:.2f})"
            )

        except Exception as e:
            self.logger.error(f"Error handling code proposal: {e}")

    async def _handle_status(self, data: dict) -> None:
        """Handle status requests."""
        try:
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

            # Publish status response
            await self.event_bus.publish("kernel.status.response", status)
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")

    async def _get_risk_analysis(
        self,
        proposal: dict,
        is_code: bool = False,
    ) -> Optional[RiskAnalysis]:
        """Request risk analysis from Safety Supervisor."""
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
                self.logger.warning(f"Safety Supervisor error: {data.get('error', 'unknown')}")
                return None
            return RiskAnalysis(
                trace_id=data.get("trace_id", ""),
                risk_score=data.get("risk_score", 0.0),
                flags=data.get("flags", []),
                recommendations=data.get("recommendations", []),
            )
        except Exception as e:
            self.logger.warning(f"Could not get risk analysis: {e}")
            return None

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
                    max_delta = metadata.get("max_intensity_delta", 0.3)
                    # For now, flag high-intensity commands as potential violations.
                    # Future: track per-channel last intensity for delta checking.
                    if isinstance(intensity, (int, float)) and intensity > 0.85:
                        violations.append({
                            "norm_id": norm_id,
                            "content": norm.get("content", ""),
                            "risk_boost": 0.15,
                        })

                # Check: slow_novel — intensity cap in unfamiliar environments
                if norm_id == "norm.slow_novel":
                    max_novel = metadata.get("max_intensity_novel", 0.5)
                    # Future: check if environment is "novel" via beliefs.
                    # For now, this norm is advisory — no automatic violation.

                # Check: force_limit — specific motor channels
                if norm_id == "norm.force_limit":
                    limited_channels = metadata.get("motor_channels", [])
                    if channel in limited_channels and isinstance(intensity, (int, float)) and intensity > 0.9:
                        violations.append({
                            "norm_id": norm_id,
                            "content": norm.get("content", ""),
                            "risk_boost": 0.2,
                        })

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
        flags: Optional[list[str]] = None,
        norm_violations: Optional[list[dict[str, Any]]] = None,
        original_proposal: Optional[dict[str, Any]] = None,
    ) -> None:
        """Publish decision to NATS and log to SQLite audit trail.

        Combines publish + log in one call to prevent inconsistency
        (decision published but not logged, or vice versa).
        For cognitive/speech proposals that are ALLOWED, forwards to
        the appropriate execution channel.
        """
        decision_payload = sign_decision({
            "trace_id": decision.trace_id,
            "type": decision.type.value,
            "reason": decision.reason,
            "transformations": decision.transformations,
            "risk_score": decision.risk_score,
            "issued_at": decision.issued_at,
            "expires_at": decision.expires_at,
        })

        # Publish decision (signed) — caller is waiting on this and will
        # reject it unless the signature verifies.
        await self.event_bus.publish(decision_subject(trace_id), decision_payload)

        # Track consecutive DENYs per channel for escalation
        channel = ""
        if original_proposal:
            channel = original_proposal.get("action", {}).get("channel", "")
        if channel:
            escalation = self._deny_tracker.record_decision(
                channel, decision.type.value,
            )
            if escalation == "disable":
                await self._auto_disable_channel(channel)
            elif escalation == "escalate":
                await self.event_bus.publish("safety.deny_escalation", {
                    "channel": channel,
                    "deny_count": self._deny_tracker._sequences[channel].count,
                    "action": "escalate",
                    "operator_id": "system:deny_tracker",
                })

        # On ALLOW: forward cognitive/speech to their execution channels.
        # This keeps the Kernel as the single gatekeeper — downstream services
        # subscribe to Kernel-gated subjects, not raw brain output.
        if decision.type == DecisionType.ALLOW and original_proposal:
            await self._forward_allowed_proposal(
                proposal_type, trace_id, original_proposal,
            )

        # Log to audit trail (best-effort, don't block on failure)
        await self._log_decision(
            trace_id, proposal_type, source, decision,
            flags=flags, norm_violations=norm_violations,
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
                await self.event_bus.publish("cognitive.execute", {
                    "trace_id": trace_id,
                    "prediction_error": proposal.get("action", {}).get(
                        "prediction_error", 0.0
                    ),
                    "intensity": proposal.get("action", {}).get("intensity", 0.0),
                    "step": metadata.get("step", 0),
                    "drives": metadata.get("drives", {}),
                })
            elif proposal_type == "speech":
                action = proposal.get("action", {})
                await self.event_bus.publish("speech.execute", {
                    "trace_id": trace_id,
                    "token_idx": action.get("token_idx", 0),
                    "confidence": action.get("confidence", 0.0),
                    "intensity": action.get("intensity", 0.0),
                    "step": proposal.get("metadata", {}).get("step", 0),
                    "output_number": action.get("output_number", 0),
                })
        except Exception as e:
            self.logger.warning(f"Failed to forward {proposal_type} proposal: {e}")

    async def _log_decision(
        self,
        trace_id: str,
        proposal_type: str,
        source: str,
        decision: KernelDecision,
        flags: Optional[list[str]] = None,
        norm_violations: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Log decision to SQLite audit trail."""
        try:
            await self.database.insert(
                "kernel_decisions",
                {
                    "id": str(uuid.uuid4()),
                    "trace_id": trace_id,
                    "proposal_type": proposal_type,
                    "decision_type": decision.type.value,
                    "source": source,
                    "reason": decision.reason,
                    "risk_score": decision.risk_score,
                    "flags": json.dumps(flags) if flags else None,
                    "norm_violations": json.dumps(
                        [v.get("norm_id", "") for v in norm_violations]
                    ) if norm_violations else None,
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
            await self.event_bus.publish(Subjects.POLICY_RESTRICT, {
                "motor_limits": {channel: {"max_intensity": 0.0}},
                "reason": f"Auto-disable: {channel} consecutive DENY threshold",
                "operator_id": "system:deny_tracker",
            })
            await self.event_bus.publish("safety.deny_escalation", {
                "channel": channel,
                "action": "auto_disabled",
                "deny_count": self._deny_tracker._sequences[channel].count,
                "operator_id": "system:deny_tracker",
            })
        except Exception as e:
            self.logger.error(f"Failed to auto-disable channel '{channel}': {e}")


async def main() -> None:
    """Main entry point."""
    service = KernelService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
