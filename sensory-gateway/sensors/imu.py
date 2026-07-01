"""
IMU sensor — reads accelerometer/gyroscope data from a USB serial IMU.

Supports common JSON-over-serial IMUs: MPU-6050, BNO055, ICM-42688-P,
LSM6DSO, and any device that follows the JSON-lines protocol.

Expected serial output (one JSON object per line, short or long form):

    Short:  {"ax": 0.12, "ay": -0.05, "az": 9.81,
             "gx": 0.01, "gy": -0.02, "gz": 0.0}
    Long:   {"accel_x": 0.12, "accel_y": -0.05, "accel_z": 9.81,
             "gyro_x": 0.01, "gyro_y": -0.02, "gyro_z": 0.0}

Optional extras (included when present):
    Magnetometer:  {"mx": 25.4, "my": -8.1, "mz": 42.0}
    Euler angles:  {"roll": 0.12, "pitch": -0.03, "yaw": 1.57}  (radians)
    Quaternion:    {"qw": 0.999, "qx": 0.0, "qy": 0.0, "qz": 0.04}

The output dict is always in the canonical field order defined by
_CANONICAL_FIELDS, so the spike encoder sees a consistent feature
vector regardless of what the hardware happens to emit.

Provenance is ``sensor.imu.{port_name}``, which encoding.py routes to the
proprioceptive sensory cortex via the ``"sensor.imu"`` prefix match in
``_PROVENANCE_MAP``.

Graceful no-op when pyserial is absent — the import guard in
``sensors/__init__.py`` binds ``IMUSensor = None`` and the gateway skips it.
"""

from __future__ import annotations

import asyncio
import json
import logging

from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin

logger = logging.getLogger(__name__)


# ── Normalisation ────────────────────────────────────────────────────────────

# Canonical output field order.  The spike encoder extracts dict values in
# insertion order (Python 3.7+), so a fixed order gives a consistent feature
# vector even when the hardware firmware changes field names between releases.
_CANONICAL_FIELDS: tuple[str, ...] = (
    "accel_x",
    "accel_y",
    "accel_z",  # m/s²  — linear acceleration
    "gyro_x",
    "gyro_y",
    "gyro_z",  # rad/s — angular velocity
    "mag_x",
    "mag_y",
    "mag_z",  # µT    — magnetometer (optional)
    "roll",
    "pitch",
    "yaw",  # rad   — Euler orientation (optional)
    "qw",
    "qx",
    "qy",
    "qz",  # –     — unit quaternion (optional)
)

_CANONICAL_SET: frozenset[str] = frozenset(_CANONICAL_FIELDS)

# Short-form field aliases → canonical key (lookup is done on the lower-cased key)
_ALIASES: dict[str, str] = {
    # Acceleration
    "ax": "accel_x",
    "ay": "accel_y",
    "az": "accel_z",
    "accx": "accel_x",
    "accy": "accel_y",
    "accz": "accel_z",
    # Gyroscope
    "gx": "gyro_x",
    "gy": "gyro_y",
    "gz": "gyro_z",
    "gyrx": "gyro_x",
    "gyry": "gyro_y",
    "gyrz": "gyro_z",
    # Magnetometer
    "mx": "mag_x",
    "my": "mag_y",
    "mz": "mag_z",
}

# At least one acceleration axis must be present for a reading to be valid
_REQUIRED: frozenset[str] = frozenset({"accel_x", "accel_y", "accel_z"})


def _normalize(raw: dict) -> dict:
    """Translate a raw JSON dict to canonical IMU field names.

    Returns a dict keyed by canonical field names in _CANONICAL_FIELDS order.
    Fields that are absent in *raw* are omitted entirely (not zero-padded).
    Returns ``{}`` when none of the required acceleration keys are present.
    """
    out: dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(v, (int, float)):
            continue
        key = k.lower()
        canonical = _ALIASES.get(key) or (key if key in _CANONICAL_SET else None)
        if canonical is not None:
            out[canonical] = float(v)

    if not (out.keys() & _REQUIRED):
        return {}

    return {k: out[k] for k in _CANONICAL_FIELDS if k in out}


# ── Sensor ───────────────────────────────────────────────────────────────────


class IMUSensor(SensorPlugin[dict]):
    """USB serial IMU → normalised readings → NATS observation.

    Publishes to ``observation.imu.{port_name}`` with provenance
    ``sensor.imu.{port_name}``, which encoding.py maps to the
    proprioceptive sensory cortex subrange.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        hz: float = 50.0,
    ):
        port_name = port.split("/")[-1]
        super().__init__(
            sensor_id=f"imu.{port_name}",
            name=f"IMU {port_name}",
            description=f"Inertial measurement unit on {port}",
            rate_limit_hz=hz,
            risk_class=RiskClass.LOW,
        )
        self._port = port
        self._baud_rate = baud_rate
        self._serial = None

        self.add_capability(
            PluginCapability(
                name="imu_read",
                description=f"Reads accel/gyro JSON lines from {port}",
                parameters={"baud_rate": str(baud_rate), "protocol": "json_lines"},
            )
        )

    async def start(self, bus=None) -> None:
        """Open the serial port; raises RuntimeError if pyserial is absent."""
        try:
            import serial  # noqa: PLC0415 — deferred import (optional dep)
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for IMUSensor — install it with: " "pip install pyserial"
            ) from exc

        self._serial = serial.Serial(self._port, self._baud_rate, timeout=0.1)
        logger.info("IMU %s opened at %d baud", self._port, self._baud_rate)
        await super().start(bus)

    async def stop(self) -> None:
        """Stop the emit loop and close the serial port."""
        await super().stop()
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    async def capture(self) -> dict:
        """Read and normalise one JSON line from the IMU.

        Returns the normalised reading dict, or ``{}`` when:
        - the line is empty / malformed JSON
        - the reading lacks the required acceleration axes
        - the serial read times out (device stalled or unplugged)
        """
        if self._serial is None:
            raise RuntimeError("IMU serial port not open — call start() first")

        loop = asyncio.get_event_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, self._read_line),
                timeout=1.0,
            )
        except TimeoutError:
            logger.warning("IMU read timeout on %s", self._port)
            return {}
        return data

    def _read_line(self) -> dict:
        """Read and parse one JSON line (blocking, runs in thread pool)."""
        if self._serial is None or not self._serial.is_open:
            return {}
        try:
            line = self._serial.readline().decode("utf-8", errors="ignore").strip()
            if line:
                return _normalize(json.loads(line))
        except json.JSONDecodeError as e:
            logger.debug("IMU JSON parse error on %s: %s", self._port, e)
        except UnicodeDecodeError as e:
            logger.debug("IMU decode error on %s: %s", self._port, e)
        except OSError as e:
            logger.error("IMU device error on %s: %s", self._port, e)
        return {}
