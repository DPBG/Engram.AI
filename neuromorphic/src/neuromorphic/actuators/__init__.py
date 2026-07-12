"""Actuator adapters — ActuatorPlugin implementations for the neuromorphic motor loop.

Exports:
    MuJoCoActuator       — virtual body (MuJoCo physics) as an ActuatorPlugin
    SerialServoActuator  — real servo hardware via a serial bus driver
    ServoDriver          — Protocol for injecting hardware backends
    NullServoDriver      — no-op driver for offline development and tests
    SimpleSerialDriver   — minimal binary-framed serial driver (requires pyserial)
"""

from neuromorphic.actuators.mujoco_actuator import MuJoCoActuator
from neuromorphic.actuators.serial_servo_actuator import (
    NullServoDriver,
    SerialServoActuator,
    ServoDriver,
    SimpleSerialDriver,
)

__all__ = [
    "MuJoCoActuator",
    "NullServoDriver",
    "SerialServoActuator",
    "ServoDriver",
    "SimpleSerialDriver",
]
