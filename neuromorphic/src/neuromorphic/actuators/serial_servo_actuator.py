"""SerialServoActuator — real servo hardware as an ActuatorPlugin.

Provides a path from a sim-trained motor policy to physical hardware via a
serial bus controller.  Hardware communication is abstracted behind the
``ServoDriver`` Protocol so the actuator is fully testable without devices.

Architecture
------------
1. ``ServoDriver`` (Protocol) — the hardware interface.  Implement this for
   your specific servo bus (Dynamixel, Feetech, PWM board, custom serial…).
2. ``NullServoDriver`` — no-op stub for offline development and unit tests.
3. ``SimpleSerialDriver`` — minimal binary-framed driver for a generic serial
   servo controller (requires ``pyserial``; optional import).
4. ``SerialServoActuator`` — the ActuatorPlugin that owns a ServoDriver,
   publishes NATS heartbeats so MotorFeedbackAdapter.is_real() returns True,
   and routes _do_execute() calls to the driver.

Usage
-----
    from neuromorphic.actuators import SerialServoActuator, SimpleSerialDriver
    from activelearning.plugins import register_actuator

    driver = SimpleSerialDriver("/dev/ttyUSB0", baud=115200)
    actuator = SerialServoActuator(
        actuator_id="servo.arm",
        channel="manipulation",
        driver=driver,
        servo_ids=[0, 1, 2, 3],
    )
    register_actuator(actuator)
    adapter.register_plugin("manipulation", actuator)
    await actuator.start(bus)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol, runtime_checkable

from activelearning.plugins import ActuatorPlugin, PluginCapability, RiskClass

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_S = 5.0

# Default servo IDs per channel — override via servo_ids in __init__
_DEFAULT_SERVO_IDS: dict[str, list[int]] = {
    "manipulation": [0, 1, 2, 3],
    "locomotion": [4, 5, 6, 7],
    "head": [8, 9],
}


@runtime_checkable
class ServoDriver(Protocol):
    """Hardware interface for a servo bus controller.

    Implement this Protocol to connect SerialServoActuator to your hardware.
    All methods are synchronous — the actuator runs them in a thread-pool
    executor so they do not block the asyncio event loop.
    """

    def connect(self) -> None:
        """Open the hardware connection (serial port, I2C bus, etc.)."""
        ...

    def disconnect(self) -> None:
        """Close the hardware connection and release resources."""
        ...

    def set_position(self, servo_id: int, angle: float, speed: float) -> bool:
        """Send a position command to a single servo.

        Args:
            servo_id: Hardware servo ID (0-based; mapping is hardware-specific).
            angle: Target angle in degrees (0.0 – 180.0).
            speed: Movement speed as a fraction (0.0 = slowest, 1.0 = maximum).

        Returns:
            True if the command was acknowledged or sent without error.
        """
        ...

    def get_position(self, servo_id: int) -> float | None:
        """Read back the current servo position in degrees.

        Returns None if the hardware does not support read-back or an error
        occurred.
        """
        ...


class NullServoDriver:
    """No-op ServoDriver for offline development and unit tests.

    All commands succeed silently.  get_position() always returns None.
    """

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def set_position(self, servo_id: int, angle: float, speed: float) -> bool:
        return True

    def get_position(self, servo_id: int) -> float | None:
        return None


class SimpleSerialDriver:
    """Minimal binary-framed serial servo driver.

    Frame format (8 bytes total):
        [0xAA][servo_id:u8][angle_centideg_hi:u8][angle_centideg_lo:u8][speed_u8][checksum:u8]

    ``angle_centideg`` = int(angle * 100)  so 90° → 9000 → 0x23 0x28
    ``speed_u8``       = int(speed * 255)

    checksum = (sum of bytes 1..4) & 0xFF

    Requires ``pyserial`` (``pip install pyserial``).  The import is deferred
    to ``connect()`` so the module loads without the dependency.
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05) -> None:
        self._port = port
        self._baud = baud
        self._timeout = timeout
        self._serial: Any = None

    def connect(self) -> None:
        try:
            import serial  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for SimpleSerialDriver. "
                "Install it with: pip install pyserial"
            ) from exc
        self._serial = serial.Serial(self._port, self._baud, timeout=self._timeout)
        logger.info("SimpleSerialDriver: opened %s @ %d baud", self._port, self._baud)

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def set_position(self, servo_id: int, angle: float, speed: float) -> bool:
        if self._serial is None:
            return False
        angle_cd = int(min(max(angle, 0.0), 180.0) * 100)  # centidegrees, 0–18000
        speed_u8 = int(min(max(speed, 0.0), 1.0) * 255)
        body = bytes([servo_id & 0xFF, (angle_cd >> 8) & 0xFF, angle_cd & 0xFF, speed_u8])
        checksum = sum(body) & 0xFF
        frame = bytes([0xAA]) + body + bytes([checksum])
        try:
            self._serial.write(frame)
            return True
        except Exception as exc:
            logger.error("SimpleSerialDriver write error: %s", exc)
            return False

    def get_position(self, servo_id: int) -> float | None:
        return None  # read-back not in this minimal protocol


class SerialServoActuator(ActuatorPlugin[dict]):
    """Real-actuator adapter that drives servos via a ServoDriver.

    Registers as the ActuatorPlugin for a motor channel and publishes periodic
    NATS heartbeats (``actuator.heartbeat.{channel}``) so
    ``MotorFeedbackAdapter.is_real(channel)`` returns True while this adapter
    is running.  When MotorFeedbackAdapter calls execute(), the action is
    forwarded to the underlying ServoDriver.

    Action dict expected by ``_do_execute()``::

        {
            "intensity":  float,        # 0.0–1.0 → mapped to angle 0°–180° when
                                        # "angle" is not supplied
            "angle":      float,        # optional: direct target degrees (overrides intensity)
            "speed":      float,        # optional: movement speed 0.0–1.0  (default 0.5)
            "servo_ids":  list[int],    # optional: restrict to a subset of servo IDs
        }
    """

    def __init__(
        self,
        actuator_id: str,
        channel: str,
        driver: ServoDriver | None = None,
        servo_ids: list[int] | None = None,
    ) -> None:
        """
        Args:
            actuator_id: Unique plugin identifier (e.g. ``"servo.arm"``).
            channel: Motor channel this adapter drives (e.g. ``"manipulation"``).
            driver: Hardware backend.  Defaults to NullServoDriver (no-op).
            servo_ids: List of hardware servo IDs to command.  Defaults to
                ``_DEFAULT_SERVO_IDS[channel]`` or ``[0]``.
        """
        super().__init__(
            actuator_id=actuator_id,
            name=f"Servo actuator — {channel}",
            description=(
                f"Drives physical servo motors on the '{channel}' motor channel "
                "via a serial bus controller."
            ),
            envelope={"intensity": (0.0, 1.0), "angle": (0.0, 180.0), "speed": (0.0, 1.0)},
            risk_class=RiskClass.HIGH,
        )
        self._channel = channel
        self._driver: ServoDriver = driver if driver is not None else NullServoDriver()
        self._servo_ids: list[int] = (
            servo_ids if servo_ids is not None else _DEFAULT_SERVO_IDS.get(channel, [0])
        )
        self._heartbeat_task: asyncio.Task | None = None  # type: ignore[type-arg]

        self.add_capability(
            PluginCapability(
                name="set_position",
                description="Command all channel servos to a target angle via serial bus.",
                parameters={
                    "intensity": "float",
                    "angle": "float",
                    "speed": "float",
                },
                risk_class=RiskClass.HIGH,
            )
        )

    @property
    def channel(self) -> str:
        """Motor channel this adapter is bound to."""
        return self._channel

    async def start(self, bus: Any = None) -> None:
        """Connect to hardware and start publishing heartbeats."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._driver.connect)
        try:
            await super().start(bus)
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        except Exception:
            await loop.run_in_executor(None, self._driver.disconnect)
            raise
        logger.info(
            "SerialServoActuator '%s' started (channel=%s, servos=%s)",
            self.actuator_id,
            self._channel,
            self._servo_ids,
        )

    async def stop(self) -> None:
        """Stop heartbeat task and disconnect from hardware."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        await super().stop()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._driver.disconnect)
        logger.info("SerialServoActuator '%s' stopped", self.actuator_id)

    async def _do_execute(self, action: dict) -> bool:
        """Forward the action to all channel servos via the driver.

        Returns True only if every servo command succeeded.
        """
        intensity = float(action.get("intensity", 0.0))
        angle = float(action.get("angle", intensity * 180.0))
        speed = float(action.get("speed", 0.5))
        requested = action.get("servo_ids")
        if requested is not None:
            target_ids = [sid for sid in requested if sid in self._servo_ids]
            if not target_ids:
                logger.warning(
                    "SerialServoActuator '%s': all requested servo_ids %s are outside "
                    "configured set %s — command ignored",
                    self.actuator_id,
                    list(requested),
                    self._servo_ids,
                )
                return False
        else:
            target_ids = self._servo_ids

        loop = asyncio.get_running_loop()
        results: list[bool] = list(
            await asyncio.gather(
                *[
                    loop.run_in_executor(None, self._driver.set_position, sid, angle, speed)
                    for sid in target_ids
                ]
            )
        )
        failed = sum(1 for r in results if not r)
        if failed:
            logger.warning(
                "SerialServoActuator '%s': %d/%d servo(s) did not respond",
                self.actuator_id,
                failed,
                len(results),
            )
        return failed == 0

    async def _heartbeat_loop(self) -> None:
        """Publish ``actuator.heartbeat.{channel}`` every _HEARTBEAT_INTERVAL_S seconds.

        MotorFeedbackAdapter tracks these to decide whether to route commands
        to real hardware (is_real()) or fall back to MuJoCo simulation.
        """
        while True:
            try:
                if self._bus is not None:
                    await self._bus.publish(
                        f"actuator.heartbeat.{self._channel}",
                        {
                            "channel": self._channel,
                            "actuator_id": self.actuator_id,
                            "timestamp": time.time(),
                        },
                    )
            except Exception as exc:
                logger.debug(
                    "SerialServoActuator '%s' heartbeat error: %s",
                    self.actuator_id,
                    exc,
                )
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
