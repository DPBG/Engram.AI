"""
Minimal SensorPlugin example — synthetic temperature sensor.

This module shows the minimum code needed to build a custom sensor plugin for
Engram. The sensor generates a realistic random-walk temperature reading on
every capture() call. No hardware is required.

How it fits into Engram
-----------------------
1. SensorPlugin.capture() returns your raw reading (any type you choose).
2. The base class wraps it in an Observation and publishes it on the NATS bus
   as ``observation.<sensor_id>`` at the configured rate.
3. The neuromorphic brain and sensory-gateway subscribe to ``observation.*``
   and convert those payloads into spike trains.

To run standalone (no NATS needed):
    python sdk/examples/minimal_sensor.py

To run against a local Engram stack (python run.py must be running):
    python sdk/examples/run_plugin_examples.py
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin, register_sensor


# ---------------------------------------------------------------------------
# 1. Define the data type your sensor produces.
#    This can be any Python type — dict, dataclass, float, etc.
# ---------------------------------------------------------------------------

@dataclass
class TemperatureReading:
    celsius: float
    humidity_pct: float
    sensor_id: str


# ---------------------------------------------------------------------------
# 2. Subclass SensorPlugin, parameterised with your data type.
# ---------------------------------------------------------------------------

class TemperatureSensor(SensorPlugin[TemperatureReading]):
    """Synthetic temperature / humidity sensor using a random walk.

    Parameters
    ----------
    sensor_id:
        Unique bus address. Observations are published on
        ``observation.<sensor_id>``. Must be stable across restarts so the
        brain can learn from it over time.
    rate_limit_hz:
        How many observations to emit per second when running in continuous
        mode (started with start()). Defaults to 1 Hz to keep the console
        readable.
    """

    def __init__(
        self,
        sensor_id: str = "sensor.temperature.room",
        rate_limit_hz: float = 1.0,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            name="Synthetic Temperature Sensor",
            description="Random-walk temperature/humidity; no hardware required.",
            rate_limit_hz=rate_limit_hz,
            risk_class=RiskClass.LOW,
        )

        # Declare what this sensor can do. Capabilities are advisory metadata
        # used by the planner and dashboard — they don't affect data flow.
        self.add_capability(PluginCapability(
            name="temperature",
            description="Ambient temperature in Celsius",
            parameters={"celsius": "float", "range": "[-20, 60]"},
            risk_class=RiskClass.LOW,
        ))
        self.add_capability(PluginCapability(
            name="humidity",
            description="Relative humidity as a percentage",
            parameters={"humidity_pct": "float", "range": "[0, 100]"},
            risk_class=RiskClass.LOW,
        ))

        # Internal state: random walk starts at 22 °C / 50 % RH.
        self._temperature: float = 22.0
        self._humidity: float = 50.0

    # -----------------------------------------------------------------------
    # 3. Implement capture() — the only required method.
    #    Return your data type; the base class handles publishing.
    # -----------------------------------------------------------------------

    async def capture(self) -> TemperatureReading:
        """Generate the next synthetic reading via a bounded random walk."""
        # Small Gaussian step, clamped to realistic ranges.
        self._temperature = max(-20.0, min(60.0, self._temperature + random.gauss(0, 0.3)))
        self._humidity = max(0.0, min(100.0, self._humidity + random.gauss(0, 0.5)))

        return TemperatureReading(
            celsius=round(self._temperature, 2),
            humidity_pct=round(self._humidity, 1),
            sensor_id=self.sensor_id,
        )


# ---------------------------------------------------------------------------
# 4. Optionally register the sensor so the global registry knows about it.
#    The registry lets other parts of the system discover it at runtime.
# ---------------------------------------------------------------------------

def create_and_register() -> TemperatureSensor:
    """Instantiate the sensor and add it to the global plugin registry."""
    sensor = TemperatureSensor()
    register_sensor(sensor)
    return sensor


# ---------------------------------------------------------------------------
# Standalone demo — no NATS needed.
# ---------------------------------------------------------------------------

async def _demo() -> None:
    sensor = TemperatureSensor()
    print(f"Sensor: {sensor.name} ({sensor.sensor_id})")
    print(f"Capabilities: {[c.name for c in sensor.get_capabilities()]}\n")

    print("Sampling 5 readings (no NATS, no full stack required):")
    for i in range(5):
        reading = await sensor.capture()
        print(f"  [{i + 1}] {reading.celsius:.2f} °C  |  {reading.humidity_pct:.1f} % RH")

    print("\nIn production, call sensor.start(bus) and the base class emits")
    print(f"each capture() result to NATS on 'observation.{sensor.sensor_id}'.")


if __name__ == "__main__":
    asyncio.run(_demo())
