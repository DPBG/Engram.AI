"""MuJoCoActuator — the virtual motor body as an ActuatorPlugin.

Wraps MuJoCoBody.step_command() so the MuJoCo path in MotorFeedbackAdapter
can be addressed through the standard ActuatorPlugin interface alongside real
hardware adapters.  MotorFeedbackAdapter creates one instance per channel and
registers it via register_plugin().
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from activelearning.plugins import ActuatorPlugin, PluginCapability, RiskClass

logger = logging.getLogger(__name__)


class MuJoCoActuator(ActuatorPlugin[dict]):
    """ActuatorPlugin backed by the MuJoCo virtual body.

    Each instance is bound to a single motor channel (e.g. ``"locomotion"``,
    ``"manipulation"``).  The body is shared across channels — callers must
    hold any concurrency lock themselves before calling execute() if needed.

    Example::

        body = MuJoCoBody(steps_per_command=200, channel_actuators=cfg.channel_actuators)
        actuator = MuJoCoActuator(body, channel="locomotion")
        adapter.register_plugin("locomotion", actuator)
    """

    def __init__(self, body: Any, channel: str) -> None:
        """
        Args:
            body: A MuJoCoBody instance (typed as Any to avoid a hard circular
                  import; duck-typed on step_command()).
            channel: Motor channel name this actuator drives.
        """
        super().__init__(
            actuator_id=f"mujoco.{channel}",
            name=f"MuJoCo virtual actuator — {channel}",
            description=(
                f"Drives MuJoCo physics joints for the '{channel}' motor channel "
                "via step_command(). Risk class LOW — no physical hardware."
            ),
            envelope={"intensity": (0.0, 1.0)},
            risk_class=RiskClass.LOW,
        )
        self._body = body
        self._channel = channel
        self.add_capability(
            PluginCapability(
                name="step_command",
                description="Apply a normalised torque and advance physics one step.",
                parameters={"intensity": "float"},
                risk_class=RiskClass.LOW,
            )
        )

    async def _do_execute(self, action: dict) -> bool:
        """Run step_command() in a thread-pool executor.

        Args:
            action: Must contain ``"intensity"`` (float 0–1). Remaining keys
                    are ignored (population-vector routing is handled by
                    MotorFeedbackAdapter before dispatch).

        Returns:
            True when step_command() reports success; False on error.
        """
        intensity = float(action.get("intensity", 0.0))
        loop = asyncio.get_running_loop()
        try:
            outcome: dict = await loop.run_in_executor(
                None, self._body.step_command, self._channel, intensity
            )
            return bool(outcome.get("success", False))
        except Exception as exc:
            logger.error(
                "MuJoCoActuator '%s' error on channel '%s': %s",
                self.actuator_id, self._channel, exc,
            )
            return False

    @property
    def channel(self) -> str:
        """Motor channel this actuator is bound to."""
        return self._channel
