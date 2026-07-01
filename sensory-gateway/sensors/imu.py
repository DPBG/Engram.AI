"""
IMU sensor — reads a 9-DOF inertial measurement unit (accelerometer,
gyroscope, and fused orientation) and emits readings for binding as
proprioceptive input.

Readings are published to observation.{sensor_id} with provenance
"sensor.imu.{device}", which the neuromorphic encoder (``encoding.py``) routes
to the *proprioceptive* modality and rate-codes into spike trains — see
``_PROVENANCE_MAP`` / ``_extract_features`` in
``neuromorphic/src/neuromorphic/encoding.py``.

The driver is hardware-agnostic: it reads through an injected ``reader``
callable that returns one raw sample as a mapping of numeric fields. Real
deployments supply a reader backed by their bus (I2C / SPI / serial / ROS); the
built-in :func:`open_serial_imu_reader` covers the common case of an IMU dev
board streaming JSON lines over USB serial. With no working reader the sensor is
a graceful no-op: discovery yields nothing and :meth:`start` raises a clear
error rather than silently emitting zeros.

Expected raw sample (aliases in parentheses), one JSON object per serial line::

    {"accel_x": 0.1, "accel_y": 0.0, "accel_z": 9.81,   # (ax, ay, az)  m/s^2
     "gyro_x": 0.01, "gyro_y": 0.0, "gyro_z": -0.02,     # (gx, gy, gz)  rad/s
     "roll": 0.0, "pitch": 0.05, "yaw": 1.57}            # Euler angles  rad
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Callable, Mapping

from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin

logger = logging.getLogger(__name__)

# Canonical reading schema, in a fixed order so the downstream encoder always
# sees a stable proprioceptive feature vector: linear acceleration (m/s^2),
# angular velocity (rad/s), and fused orientation as Euler angles (rad).
IMU_FIELDS = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "roll",
    "pitch",
    "yaw",
)

# Accepted field aliases — device firmwares label the same axes differently.
_ALIASES: dict[str, tuple[str, ...]] = {
    "accel_x": ("accel_x", "ax", "acc_x"),
    "accel_y": ("accel_y", "ay", "acc_y"),
    "accel_z": ("accel_z", "az", "acc_z"),
    "gyro_x": ("gyro_x", "gx", "gyr_x"),
    "gyro_y": ("gyro_y", "gy", "gyr_y"),
    "gyro_z": ("gyro_z", "gz", "gyr_z"),
    "roll": ("roll",),
    "pitch": ("pitch",),
    "yaw": ("yaw", "heading"),
}

# A reader returns one raw sample, or None when no sample is available yet.
Reader = Callable[[], Mapping[str, float] | None]


class ImuSensor(SensorPlugin[dict]):
    """Inertial measurement unit -> proprioceptive observation."""

    def __init__(
        self,
        device_id: str = "0",
        hz: float = 50.0,
        *,
        reader: Reader | None = None,
    ):
        super().__init__(
            sensor_id=f"imu.{device_id}",
            name=f"IMU {device_id}",
            description="9-DOF inertial measurement unit (accel/gyro/orientation)",
            rate_limit_hz=hz,
            risk_class=RiskClass.LOW,
        )
        self._reader = reader

        self.add_capability(
            PluginCapability(
                name="imu_read",
                description="Reads acceleration, angular velocity, and orientation",
                parameters={"fields": ",".join(IMU_FIELDS), "modality": "proprioceptive"},
            )
        )

    async def start(self, bus=None) -> None:
        """Start emitting. Fails loudly when no reader (hardware) is configured."""
        if self._reader is None:
            raise RuntimeError(f"IMU {self.sensor_id} has no reader configured (hardware absent)")
        await super().start(bus)

    async def stop(self) -> None:
        """Stop emitting and close the reader if it owns a resource."""
        await super().stop()
        close = getattr(self._reader, "close", None)
        if callable(close):
            close()

    async def capture(self) -> dict | None:
        """Read one IMU sample and normalize it to the canonical schema.

        Returns a flat dict of floats, or ``None`` when no sample is available
        (skipped by the emit loop) so a stalled device never emits stale zeros.
        The blocking read runs in a thread with a hard timeout in case the
        device is yanked and the underlying read never returns.
        """
        if self._reader is None:
            raise RuntimeError("IMU reader not open")

        loop = asyncio.get_event_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, self._reader),
                timeout=1.0,
            )
        except TimeoutError:
            logger.warning(f"IMU read timeout on {self.sensor_id}")
            return None
        except Exception as e:  # a faulty reader must not kill the emit loop
            logger.error(f"IMU read error on {self.sensor_id}: {e}")
            return None

        if not raw:
            return None
        return normalize_reading(raw)

    async def _emit_loop(self) -> None:
        """Emit loop that skips ``None`` samples (device not ready)."""
        interval = 1.0 / self.rate_limit_hz if self.rate_limit_hz > 0 else 0
        while self._running:
            try:
                data = await self.capture()
                if data is not None:
                    await self.emit(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sensor {self.sensor_id}: {e}")
            if interval > 0:
                await asyncio.sleep(interval)


def normalize_reading(raw: Mapping[str, float]) -> dict:
    """Map a raw device sample onto the canonical IMU schema.

    Accepts common field aliases (e.g. ``"ax"`` for ``"accel_x"``), coerces
    values to float, ignores non-finite values, and fills missing fields with
    ``0.0`` so the result is always a full, stable, all-numeric proprioceptive
    feature vector — exactly what the encoder's dict path consumes.
    """
    out: dict[str, float] = {}
    for field in IMU_FIELDS:
        value = 0.0
        for alias in _ALIASES[field]:
            if alias in raw:
                try:
                    candidate = float(raw[alias])
                except (TypeError, ValueError):
                    break
                if math.isfinite(candidate):
                    value = candidate
                break
        out[field] = value
    return out


def open_serial_imu_reader(port: str, baud_rate: int = 115200) -> Reader:
    """Build a reader for an IMU dev board streaming JSON lines over USB serial.

    Requires ``pyserial``; raises a clear error if it is not installed so the
    sensor fails loudly at start rather than silently emitting nothing.
    """
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial not installed — cannot open serial IMU") from exc

    conn = serial.Serial(port, baud_rate, timeout=0.1)

    def _read() -> dict | None:
        try:
            line = conn.readline().decode("utf-8", errors="ignore").strip()
            if line:
                return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug(f"IMU serial parse error: {e}")
        except OSError as e:
            logger.error(f"IMU serial device error on {port}: {e}")
        return None

    _read.close = conn.close  # type: ignore[attr-defined]
    return _read
