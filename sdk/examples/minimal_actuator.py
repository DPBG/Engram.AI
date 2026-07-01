"""
Minimal ActuatorPlugin example — synthetic LED strip.

This module shows the minimum code needed to build a custom actuator plugin
for Engram. The actuator "drives" an LED strip by printing the colour command
to stdout. No hardware is required.

How it fits into Engram
-----------------------
1. The brain or planner creates an ActionProposal and publishes it to
   ``proposal.new`` on the NATS bus.
2. The Kernel evaluates it (ALLOW / DENY / TRANSFORM / DEFER) and publishes
   the decision to ``decision.<trace_id>``.
3. Your code calls actuator.submit_and_execute(proposal) which handles steps
   1–2 automatically and only calls _do_execute() on ALLOW.
4. _do_execute() performs the real (or simulated) hardware action and returns
   True on success, False on failure.

The envelope restricts parameter ranges at the actuator level (belt-and-
suspenders on top of the Kernel's body-profile check).

To run standalone (no NATS needed):
    python sdk/examples/minimal_actuator.py

To run against a local Engram stack (python run.py must be running):
    python sdk/examples/run_plugin_examples.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from activelearning.core import ActionProposal, KernelDecisionType, generate_trace_id
from activelearning.plugins import ActuatorPlugin, PluginCapability, RiskClass, register_actuator


# ---------------------------------------------------------------------------
# 1. Define the command type your actuator accepts.
# ---------------------------------------------------------------------------

@dataclass
class LedCommand:
    red: int    # 0–255
    green: int  # 0–255
    blue: int   # 0–255
    brightness: float  # 0.0–1.0


# ---------------------------------------------------------------------------
# 2. Subclass ActuatorPlugin, parameterised with your command type.
# ---------------------------------------------------------------------------

class LedStripActuator(ActuatorPlugin[LedCommand]):
    """Synthetic RGB LED strip actuator.

    All numeric parameters are declared in the ``envelope`` so the base-class
    validate_envelope() rejects out-of-range values before _do_execute() is
    ever called.

    Parameters
    ----------
    actuator_id:
        Unique bus address used for outcome publishing.
    """

    def __init__(self, actuator_id: str = "actuator.led.strip") -> None:
        super().__init__(
            actuator_id=actuator_id,
            name="Synthetic LED Strip",
            description="Logs colour commands; no hardware required.",
            envelope={
                # parameter_name: (min, max)
                "red":        (0.0, 255.0),
                "green":      (0.0, 255.0),
                "blue":       (0.0, 255.0),
                "brightness": (0.0, 1.0),
            },
            risk_class=RiskClass.LOW,
        )

        self.add_capability(PluginCapability(
            name="set_color",
            description="Set the LED strip colour and brightness",
            parameters={
                "red":        "int [0, 255]",
                "green":      "int [0, 255]",
                "blue":       "int [0, 255]",
                "brightness": "float [0.0, 1.0]",
            },
            risk_class=RiskClass.LOW,
        ))

        # Tracks the last successfully applied colour (for status queries).
        self._current: LedCommand | None = None

    # -----------------------------------------------------------------------
    # 3. Implement _do_execute() — the only required method.
    #    The base class calls it after Kernel approval and envelope validation.
    #    Return True on success, False on failure.
    # -----------------------------------------------------------------------

    async def _do_execute(self, command: LedCommand) -> bool:
        """Apply the colour command (simulated: just log it)."""
        print(
            f"  [LED] rgb({command.red}, {command.green}, {command.blue}) "
            f"@ {command.brightness * 100:.0f}% brightness"
        )
        self._current = command
        return True


# ---------------------------------------------------------------------------
# 4. Optionally register the actuator in the global registry.
# ---------------------------------------------------------------------------

def create_and_register() -> LedStripActuator:
    """Instantiate the actuator and add it to the global plugin registry."""
    actuator = LedStripActuator()
    register_actuator(actuator)
    return actuator


# ---------------------------------------------------------------------------
# Standalone demo — bypasses NATS/Kernel; calls _do_execute() directly so
# you can verify the logic without a running stack.
# ---------------------------------------------------------------------------

async def _demo() -> None:
    actuator = LedStripActuator()
    print(f"Actuator: {actuator.name} ({actuator.actuator_id})")
    print(f"Envelope: {actuator.envelope}\n")

    print("Sending 3 synthetic commands (direct _do_execute, no Kernel):")
    commands = [
        LedCommand(red=255, green=0,   blue=0,   brightness=1.0),
        LedCommand(red=0,   green=255, blue=0,   brightness=0.5),
        LedCommand(red=0,   green=0,   blue=255, brightness=0.75),
    ]
    for cmd in commands:
        await actuator._do_execute(cmd)

    print("\nEnvelope validation (out-of-range brightness rejected):")
    bad_action = {"red": 100, "green": 100, "blue": 100, "brightness": 2.5}
    ok, error = actuator.validate_envelope(bad_action)
    print(f"  valid={ok}, error={error!r}")

    print("\nIn production, call actuator.submit_and_execute(proposal).")
    print("The base class publishes the proposal, waits for Kernel approval,")
    print("then calls _do_execute() only on ALLOW.")


if __name__ == "__main__":
    asyncio.run(_demo())
