"""
Planner Service - Converts observations to action proposals.

Subscribes to observations, generates action proposals,
and manages the execution flow through the Kernel.
"""

import asyncio
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import uuid

from activelearning import BaseService

from planner.scheduler import Scheduler, SchedulerMode, PendingAction


@dataclass
class ActionProposal:
    """A proposed action."""
    trace_id: str
    provenance: str
    action: dict[str, Any]
    priority: int = 0
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelDecision:
    """A decision from the Kernel."""
    trace_id: str
    type: str  # ALLOW, TRANSFORM, DENY, DEFER
    reason: Optional[str] = None
    transformations: Optional[list[dict]] = None
    risk_score: float = 0.0


class PlannerService(BaseService):
    """
    Planner service that processes observations into actions.

    Flow:
    1. Subscribe to observation.* subjects
    2. Generate ActionProposal from observations
    3. Submit to Kernel via proposal.new
    4. Wait for decision on decision.{trace_id}
    5. Execute or handle based on decision
    """

    def __init__(self):
        super().__init__("planner", use_database=False, use_event_bus=True)
        self._scheduler = Scheduler()
        self._pending_decisions: dict[str, asyncio.Future] = {}
        self._process_task: Optional[asyncio.Task] = None

    async def _setup(self) -> None:
        """Service-specific setup."""
        # Subscribe to observations (wildcard)
        await self.event_bus.subscribe("observation.*", self._handle_observation)

        # Subscribe to decisions
        await self.event_bus.subscribe("decision.*", self._handle_decision)

        # Subscribe to mode change requests
        await self.event_bus.subscribe("planner.mode", self._handle_mode_change)

        # Subscribe to status requests
        await self.event_bus.subscribe("planner.status", self._handle_status)

        # Start action processor
        self._process_task = asyncio.create_task(self._process_actions())

        self.logger.info("Planner service setup complete")

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        # Switch to SAFE_HALT
        await self._scheduler.set_mode(SchedulerMode.SAFE_HALT)

        # Cancel process task
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

    async def _handle_observation(self, data: dict) -> None:
        """Handle incoming observations."""
        try:
            # data is already deserialized by EventBus
            subject = data.get("subject", "observation.unknown")
            self.logger.debug(f"Received observation on {subject}: {data.get('trace_id', 'unknown')}")

            # Generate action proposal from observation
            proposal = await self._generate_proposal(data, subject)

            if proposal:
                # Enqueue for processing
                pending = PendingAction(
                    trace_id=proposal.trace_id,
                    priority=proposal.priority,
                    proposal=asdict(proposal),
                )
                await self._scheduler.enqueue(pending)

        except Exception as e:
            self.logger.error(f"Error handling observation: {e}")

    async def _generate_proposal(
        self,
        observation: dict[str, Any],
        subject: str,
    ) -> Optional[ActionProposal]:
        """
        Generate an action proposal from an observation.

        This is a placeholder that should be extended with actual planning logic.
        """
        trace_id = observation.get("trace_id", str(uuid.uuid4()))
        provenance = observation.get("provenance", subject)
        data = observation.get("data", {})

        # Simple reactive planning - extend this with real logic
        action = None
        priority = 0

        # Example: React to motion detection
        if "motion_detected" in data:
            action = {
                "type": "alert",
                "message": "Motion detected",
                "source": provenance,
            }
            priority = 5

        # Example: React to voice commands
        if "voice_command" in data:
            action = {
                "type": "execute_command",
                "command": data["voice_command"],
                "source": provenance,
            }
            priority = 10

        if action is None:
            return None

        return ActionProposal(
            trace_id=trace_id,
            provenance="planner.main",
            action=action,
            priority=priority,
        )

    async def _handle_decision(self, data: dict) -> None:
        """Handle decisions from the Kernel."""
        try:
            # data is already deserialized by EventBus
            trace_id = data.get("trace_id", "")

            self.logger.debug(f"Received decision for {trace_id}: {data.get('type')}")

            # Resolve pending decision future
            if trace_id in self._pending_decisions:
                future = self._pending_decisions.pop(trace_id)
                if not future.done():
                    future.set_result(
                        KernelDecision(
                            trace_id=trace_id,
                            type=data.get("type", "DENY"),
                            reason=data.get("reason"),
                            transformations=data.get("transformations"),
                            risk_score=data.get("risk_score", 0.0),
                        )
                    )

        except Exception as e:
            self.logger.error(f"Error handling decision: {e}")

    async def _handle_mode_change(self, data: dict) -> None:
        """Handle mode change requests."""
        try:
            # data is already deserialized by EventBus
            new_mode = SchedulerMode(data.get("mode", "EXECUTION"))
            await self._scheduler.set_mode(new_mode)
            self.logger.info(f"Scheduler mode changed to {new_mode.value}")
        except Exception as e:
            self.logger.error(f"Error changing mode: {e}")

    async def _handle_status(self, data: dict) -> None:
        """Handle status requests."""
        try:
            status = self._scheduler.get_queue_status()
            self.logger.debug(f"Status requested: {status}")
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")

    async def _process_actions(self) -> None:
        """Background task to process queued actions."""
        try:
            while True:
                try:
                    # Dequeue next action
                    pending = await self._scheduler.dequeue()

                    if pending is None:
                        await asyncio.sleep(0.1)
                        continue

                    # Submit to Kernel
                    proposal = pending.proposal
                    trace_id = proposal["trace_id"]

                    self.logger.debug(f"Submitting proposal to Kernel: {trace_id}")

                    # Create future for decision
                    decision_future: asyncio.Future[KernelDecision] = asyncio.Future()
                    self._pending_decisions[trace_id] = decision_future

                    # Publish proposal
                    await self.event_bus.publish(
                        "proposal.new",
                        proposal,
                    )

                    # Wait for decision with timeout
                    try:
                        decision = await asyncio.wait_for(decision_future, timeout=30.0)
                        await self._handle_kernel_decision(decision, proposal)
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Kernel decision timeout for {trace_id}")
                        self._pending_decisions.pop(trace_id, None)

                except asyncio.CancelledError:
                    self.logger.info("Action processor cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Error processing action: {e}")
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def _handle_kernel_decision(
        self,
        decision: KernelDecision,
        proposal: dict[str, Any],
    ) -> None:
        """Handle a Kernel decision."""
        trace_id = decision.trace_id

        if decision.type == "ALLOW":
            self.logger.info(f"Action {trace_id} ALLOWED")
            await self._execute_action(proposal)

        elif decision.type == "TRANSFORM":
            self.logger.info(f"Action {trace_id} TRANSFORMED")
            if decision.transformations:
                # Execute transformed action
                await self._execute_action(decision.transformations[0])
            else:
                await self._execute_action(proposal)

        elif decision.type == "DENY":
            self.logger.warning(f"Action {trace_id} DENIED: {decision.reason}")
            # Publish denial event
            await self.event_bus.publish(
                f"outcome.{trace_id}",
                {
                    "trace_id": trace_id,
                    "success": False,
                    "error": decision.reason,
                    "decision_type": "DENY",
                },
            )

        elif decision.type == "DEFER":
            self.logger.info(f"Action {trace_id} DEFERRED for human approval")
            # Route to Dashboard for human approval
            await self.event_bus.publish(
                "approval.request",
                {
                    "trace_id": trace_id,
                    "proposal": proposal,
                    "reason": decision.reason,
                },
            )

    async def _execute_action(self, proposal: dict[str, Any]) -> None:
        """Execute an approved action."""
        trace_id = proposal.get("trace_id", "")
        action = proposal.get("action", {})

        self.logger.info(f"Executing action {trace_id}: {action.get('type')}")

        # Publish outcome for actuators
        await self.event_bus.publish(
            f"outcome.{trace_id}",
            {
                "trace_id": trace_id,
                "action": action,
                "decision_type": "ALLOW",
            },
        )


async def main() -> None:
    """Main entry point."""
    service = PlannerService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
