"""
Plugin interfaces for sensors and actuators.

Sensors emit Observations, Actuators execute actions.
All plugins communicate through the NATS event bus.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from activelearning.core import (
    ActionProposal,
    KernelDecision,
    KernelDecisionType,
    Observation,
    Outcome,
    generate_trace_id,
)
from activelearning.nats_client import EventBus

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RiskClass(Enum):
    """Risk classification for plugins."""

    LOW = "low"  # Simple data collection
    MEDIUM = "medium"  # Some physical interaction
    HIGH = "high"  # Significant physical actions
    CRITICAL = "critical"  # Safety-critical operations


@dataclass
class PluginCapability:
    """Describes a capability provided by a plugin."""

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)  # param_name -> type
    risk_class: RiskClass = RiskClass.LOW


@dataclass
class SensorMetadata:
    """Metadata for a registered sensor."""

    sensor_id: str
    name: str
    description: str
    capabilities: list[PluginCapability]
    rate_limit_hz: float = 30.0
    risk_class: RiskClass = RiskClass.LOW


@dataclass
class ActuatorMetadata:
    """Metadata for a registered actuator."""

    actuator_id: str
    name: str
    description: str
    capabilities: list[PluginCapability]
    envelope: dict[str, tuple[float, float]] = field(default_factory=dict)  # param -> (min, max)
    risk_class: RiskClass = RiskClass.MEDIUM


class SensorPlugin(ABC, Generic[T]):
    """
    Base class for sensor plugins.

    Sensors observe the environment and emit Observations to the event bus.
    Observations are processed by the Planner to generate ActionProposals.

    Example:
        class CameraSensor(SensorPlugin[ImageFrame]):
            def __init__(self):
                super().__init__(
                    sensor_id="camera.front",
                    name="Front Camera",
                    description="Main RGB camera"
                )

            async def capture(self) -> ImageFrame:
                return self.camera.read()

            async def _emit_loop(self):
                while self._running:
                    frame = await self.capture()
                    await self.emit(frame)
                    await asyncio.sleep(1 / self.rate_limit_hz)
    """

    def __init__(
        self,
        sensor_id: str,
        name: str,
        description: str = "",
        rate_limit_hz: float = 30.0,
        risk_class: RiskClass = RiskClass.LOW,
    ):
        self.sensor_id = sensor_id
        self.name = name
        self.description = description
        self.rate_limit_hz = rate_limit_hz
        self.risk_class = risk_class
        self._bus: EventBus | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._capabilities: list[PluginCapability] = []

    @abstractmethod
    async def capture(self) -> T:
        """
        Capture a single observation.

        Subclasses must implement this to return their observation data.
        """
        ...

    def add_capability(self, capability: PluginCapability) -> None:
        """Add a capability to this sensor."""
        self._capabilities.append(capability)

    def get_capabilities(self) -> list[PluginCapability]:
        """Get all capabilities of this sensor."""
        return self._capabilities.copy()

    def describe(self) -> SensorMetadata:
        """Get metadata describing this sensor."""
        return SensorMetadata(
            sensor_id=self.sensor_id,
            name=self.name,
            description=self.description,
            capabilities=self._capabilities,
            rate_limit_hz=self.rate_limit_hz,
            risk_class=self.risk_class,
        )

    async def start(self, bus: EventBus | None = None) -> None:
        """
        Start the sensor.

        Args:
            bus: Optional EventBus instance, will create one if not provided
        """
        if self._running:
            logger.warning(f"Sensor {self.sensor_id} already running")
            return

        if bus is None:
            self._bus = EventBus()
            await self._bus.connect()
        else:
            self._bus = bus

        self._running = True
        self._task = asyncio.create_task(self._emit_loop())
        logger.info(f"Sensor {self.sensor_id} started")

    async def stop(self) -> None:
        """Stop the sensor."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(f"Sensor {self.sensor_id} stopped")

    async def emit(self, data: T, tags: list[str] | None = None) -> None:
        """
        Emit an observation to the event bus.

        Args:
            data: Observation data
            tags: Optional semantic tags
        """
        if self._bus is None:
            raise RuntimeError("Sensor not started. Call start() first.")

        observation = Observation(
            trace_id=generate_trace_id(),
            provenance=f"sensor.{self.sensor_id}",
            data=data,
            tags=tags or [],
        )

        subject = f"observation.{self.sensor_id}"
        await self._bus.publish(subject, observation)
        logger.debug(f"Emitted observation to {subject}")

    async def _emit_loop(self) -> None:
        """Internal loop for continuous emission."""
        interval = 1.0 / self.rate_limit_hz if self.rate_limit_hz > 0 else 0
        while self._running:
            try:
                data = await self.capture()
                await self.emit(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sensor {self.sensor_id}: {e}")
            if interval > 0:
                await asyncio.sleep(interval)


class ActuatorPlugin(ABC, Generic[T]):
    """
    Base class for actuator plugins.

    Actuators execute actions that affect the physical world.
    All actions must be approved by the Moral Kernel before execution.

    Example:
        class ServoActuator(ActuatorPlugin[ServoCommand]):
            def __init__(self):
                super().__init__(
                    actuator_id="servo.arm",
                    name="Arm Servo",
                    envelope={"angle": (0.0, 180.0), "speed": (0.0, 100.0)}
                )

            async def _do_execute(self, command: ServoCommand) -> bool:
                return self.servo.set_angle(command.angle, command.speed)
    """

    def __init__(
        self,
        actuator_id: str,
        name: str,
        description: str = "",
        envelope: dict[str, tuple[float, float]] | None = None,
        risk_class: RiskClass = RiskClass.MEDIUM,
    ):
        self.actuator_id = actuator_id
        self.name = name
        self.description = description
        self.envelope = envelope or {}
        self.risk_class = risk_class
        self._bus: EventBus | None = None
        self._capabilities: list[PluginCapability] = []

    @abstractmethod
    async def _do_execute(self, action: T) -> bool:
        """
        Execute the action (after Kernel approval).

        Subclasses must implement this to perform the actual action.
        Returns True on success, False on failure.
        """
        ...

    def add_capability(self, capability: PluginCapability) -> None:
        """Add a capability to this actuator."""
        self._capabilities.append(capability)

    def get_capabilities(self) -> list[PluginCapability]:
        """Get all capabilities of this actuator."""
        return self._capabilities.copy()

    def describe(self) -> ActuatorMetadata:
        """Get metadata describing this actuator."""
        return ActuatorMetadata(
            actuator_id=self.actuator_id,
            name=self.name,
            description=self.description,
            capabilities=self._capabilities,
            envelope=self.envelope,
            risk_class=self.risk_class,
        )

    def validate_envelope(self, action: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Validate action parameters against the envelope.

        Args:
            action: Action parameters as dict

        Returns:
            Tuple of (is_valid, error_message)
        """
        for param, (min_val, max_val) in self.envelope.items():
            if param in action:
                value = action[param]
                if not isinstance(value, (int, float)):
                    return False, f"Parameter {param} must be numeric"
                if value < min_val or value > max_val:
                    return False, f"Parameter {param}={value} outside range [{min_val}, {max_val}]"
        return True, None

    async def start(self, bus: EventBus | None = None) -> None:
        """
        Start the actuator and subscribe to decision events.

        Args:
            bus: Optional EventBus instance
        """
        if bus is None:
            self._bus = EventBus()
            await self._bus.connect()
        else:
            self._bus = bus

        # Observe execution outcomes. Outcomes are published per-trace as
        # `outcome.{trace_id}` (see execute() and planner.service), so an
        # actuator that wants to watch outcomes must subscribe to the wildcard
        # rather than to its own id — the previous `outcome.{actuator_id}`
        # subscription could never match a per-trace publish.
        subject = "outcome.*"
        await self._bus.subscribe(subject, self._handle_outcome)
        logger.info(f"Actuator {self.actuator_id} started")

    async def stop(self) -> None:
        """Stop the actuator."""
        if self._bus is not None:
            await self._bus.unsubscribe("outcome.*")
        logger.info(f"Actuator {self.actuator_id} stopped")

    async def execute(self, proposal: ActionProposal[T]) -> Outcome[T]:
        """
        Execute an action proposal (requires prior Kernel approval).

        This method expects the proposal has already been approved.
        Use submit_and_execute() for the full flow with Kernel submission.

        Args:
            proposal: The approved action proposal

        Returns:
            Outcome of the execution
        """
        if self._bus is None:
            raise RuntimeError("Actuator not started. Call start() first.")

        # Validate against envelope
        if isinstance(proposal.action, dict):
            is_valid, error = self.validate_envelope(proposal.action)
            if not is_valid:
                return Outcome(
                    trace_id=proposal.trace_id,
                    decision=KernelDecision(
                        trace_id=proposal.trace_id,
                        type=KernelDecisionType.DENY,
                        reason=error,
                    ),
                    original=proposal,
                    success=False,
                    error=error,
                )

        # Execute the action
        try:
            success = await self._do_execute(proposal.action)
            decision: KernelDecision[Any] = KernelDecision(
                trace_id=proposal.trace_id,
                type=KernelDecisionType.ALLOW,
            )
            outcome = Outcome(
                trace_id=proposal.trace_id,
                decision=decision,
                original=proposal,
                final=proposal,
                success=success,
            )
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            decision = KernelDecision(
                trace_id=proposal.trace_id,
                type=KernelDecisionType.DENY,
                reason=str(e),
            )
            outcome = Outcome(
                trace_id=proposal.trace_id,
                decision=decision,
                original=proposal,
                success=False,
                error=str(e),
            )

        # Publish outcome
        await self._bus.publish(f"outcome.{proposal.trace_id}", outcome)
        return outcome

    async def submit_and_execute(self, proposal: ActionProposal[T]) -> Outcome[T]:
        """
        Submit a proposal to the Kernel and execute if approved.

        Args:
            proposal: The action proposal

        Returns:
            Outcome of the execution
        """
        if self._bus is None:
            raise RuntimeError("Actuator not started. Call start() first.")

        # Submit to Kernel
        await self._bus.publish("proposal.new", proposal)

        # Wait for decision
        decision_data = await self._bus.wait_for_decision(proposal.trace_id)
        decision = KernelDecision(
            trace_id=decision_data["trace_id"],
            type=KernelDecisionType(decision_data["type"]),
            reason=decision_data.get("reason"),
            transformations=decision_data.get("transformations"),
            risk_score=decision_data.get("risk_score", 0.0),
        )

        if decision.type == KernelDecisionType.DENY:
            return Outcome(
                trace_id=proposal.trace_id,
                decision=decision,
                original=proposal,
                success=False,
                error=decision.reason,
            )

        if decision.type == KernelDecisionType.DEFER:
            return Outcome(
                trace_id=proposal.trace_id,
                decision=decision,
                original=proposal,
                success=False,
                error="Action deferred for human approval",
            )

        # Handle TRANSFORM
        final_proposal = proposal
        if decision.type == KernelDecisionType.TRANSFORM and decision.transformations:
            final_proposal = decision.transformations[0]

        # Execute
        return await self.execute(final_proposal)

    async def _handle_outcome(self, data: dict[str, Any]) -> None:
        """Handle incoming outcome messages."""
        logger.debug(f"Received outcome: {data}")


# Plugin Registry
_sensor_registry: dict[str, SensorPlugin[Any]] = {}
_actuator_registry: dict[str, ActuatorPlugin[Any]] = {}


def register_sensor(sensor: SensorPlugin[Any]) -> SensorPlugin[Any]:
    """
    Register a sensor plugin.

    Args:
        sensor: The sensor plugin to register

    Returns:
        The registered sensor (for decorator use)
    """
    if sensor.sensor_id in _sensor_registry:
        logger.warning(f"Overwriting sensor registration: {sensor.sensor_id}")
    _sensor_registry[sensor.sensor_id] = sensor
    logger.info(f"Registered sensor: {sensor.sensor_id}")
    return sensor


def register_actuator(actuator: ActuatorPlugin[Any]) -> ActuatorPlugin[Any]:
    """
    Register an actuator plugin.

    Args:
        actuator: The actuator plugin to register

    Returns:
        The registered actuator (for decorator use)
    """
    if actuator.actuator_id in _actuator_registry:
        logger.warning(f"Overwriting actuator registration: {actuator.actuator_id}")
    _actuator_registry[actuator.actuator_id] = actuator
    logger.info(f"Registered actuator: {actuator.actuator_id}")
    return actuator


def get_sensor(sensor_id: str) -> SensorPlugin[Any] | None:
    """Get a registered sensor by ID."""
    return _sensor_registry.get(sensor_id)


def get_actuator(actuator_id: str) -> ActuatorPlugin[Any] | None:
    """Get a registered actuator by ID."""
    return _actuator_registry.get(actuator_id)


def list_sensors() -> list[SensorMetadata]:
    """List all registered sensors."""
    return [s.describe() for s in _sensor_registry.values()]


def list_actuators() -> list[ActuatorMetadata]:
    """List all registered actuators."""
    return [a.describe() for a in _actuator_registry.values()]
