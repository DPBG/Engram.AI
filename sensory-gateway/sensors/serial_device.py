"""
Serial device sensor — reads JSON data from USB serial devices.

Supports Arduino, Raspberry Pi Pico, or any device that sends JSON lines
over serial. Published to observation.{sensor_id} with provenance based
on the device type (e.g., "sensor.joint" for servo encoders).

Expected serial protocol: one JSON object per line, e.g.:
    {"angle": 45.2, "torque": 0.3, "temperature": 28.1}
"""

from __future__ import annotations

import asyncio
import json
import logging

from activelearning.plugins import SensorPlugin, PluginCapability, RiskClass

logger = logging.getLogger(__name__)


class SerialSensor(SensorPlugin[dict]):
    """USB serial device -> JSON readings -> NATS observation."""

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        provenance_type: str = "sensor.serial",
        hz: float = 10.0,
    ):
        # Derive a clean sensor ID from port name
        port_name = port.split("/")[-1]

        super().__init__(
            sensor_id=f"serial.{port_name}",
            name=f"Serial {port_name}",
            description=f"USB serial device on {port}",
            rate_limit_hz=hz,
            risk_class=RiskClass.LOW,
        )
        self._port = port
        self._baud_rate = baud_rate
        self._provenance_type = provenance_type
        self._serial = None

        self.add_capability(PluginCapability(
            name="serial_read",
            description=f"Reads JSON data from {port}",
            parameters={"baud_rate": str(baud_rate), "protocol": "json_lines"},
        ))

    async def start(self, bus=None) -> None:
        """Open the serial port."""
        try:
            import serial
        except ImportError:
            raise RuntimeError("pyserial not installed")

        self._serial = serial.Serial(
            self._port,
            self._baud_rate,
            timeout=0.1,
        )
        logger.info(f"Serial port {self._port} opened at {self._baud_rate} baud")
        await super().start(bus)

    async def stop(self) -> None:
        """Close the serial port."""
        await super().stop()
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    async def capture(self) -> dict:
        """Read one JSON line from the serial device.

        Returns a dict of sensor readings. The neuromorphic encoder
        extracts numeric values from dicts via _extract_features().
        """
        if self._serial is None:
            raise RuntimeError("Serial port not open")

        # Run blocking serial read in thread pool with a hard timeout
        # in case the OS doesn't honor the serial timeout (e.g., device yanked)
        loop = asyncio.get_event_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, self._read_line),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Serial read timeout on {self._port}")
            return {}
        return data

    def _read_line(self) -> dict:
        """Read and parse one JSON line (blocking)."""
        if self._serial is None or not self._serial.is_open:
            return {}

        try:
            line = self._serial.readline().decode("utf-8", errors="ignore").strip()
            if line:
                return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug(f"Serial parse error: {e}")
        except OSError as e:
            logger.error(f"Serial device error on {self._port}: {e}")

        return {}
