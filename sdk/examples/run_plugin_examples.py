"""
Plugin example runner — demonstrates both plugins end-to-end.

Usage
-----
Standalone (no NATS / no running stack):
    python sdk/examples/run_plugin_examples.py

With a local Engram stack running (python run.py in another terminal):
    python sdk/examples/run_plugin_examples.py --live

The --live mode connects to NATS on nats://localhost:4222 (default),
starts the sensor's emit loop, subscribes to its observations, and shows
the complete observation.sensor.temperature subject in action.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from the repo root without installing the SDK.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minimal_actuator import LedCommand, LedStripActuator
from minimal_sensor import TemperatureSensor

# ---------------------------------------------------------------------------
# Standalone demo — no NATS, no running services required.
# ---------------------------------------------------------------------------


async def run_standalone() -> None:
    """Run both plugins without any NATS connection."""
    print("=" * 60)
    print(" Engram SDK Plugin Examples — standalone mode")
    print(" (no NATS, no running stack required)")
    print("=" * 60)

    # --- Sensor ---
    sensor = TemperatureSensor(rate_limit_hz=2.0)
    print(f"\nSensor: {sensor.name}")
    print(f"  id          : {sensor.sensor_id}")
    print(f"  rate        : {sensor.rate_limit_hz} Hz")
    print(f"  capabilities: {[c.name for c in sensor.get_capabilities()]}")
    print()

    print("Capturing 5 readings:")
    for _ in range(5):
        reading = await sensor.capture()
        print(f"  {reading.celsius:6.2f} °C   {reading.humidity_pct:5.1f} % RH")

    # --- Actuator ---
    actuator = LedStripActuator()
    print(f"\nActuator: {actuator.name}")
    print(f"  id          : {actuator.actuator_id}")
    print(f"  envelope    : {actuator.envelope}")
    print()

    print("Executing 3 colour commands:")
    colours = [
        LedCommand(255, 0, 0, 1.0),  # full red
        LedCommand(0, 200, 50, 0.7),  # green-ish
        LedCommand(0, 80, 255, 0.4),  # blue
    ]
    for cmd in colours:
        await actuator._do_execute(cmd)

    # Show envelope rejection
    print()
    bad = {"red": 100, "green": 100, "blue": 100, "brightness": 3.0}
    ok, err = actuator.validate_envelope(bad)
    print(f"Envelope check brightness=3.0 → valid={ok}, error={err!r}")

    print()
    print("In a live system:")
    print("  sensor observations  → NATS 'observation.<sensor_id>'")
    print("  actuator proposals   → Kernel → NATS 'decision.<trace_id>'")
    print("  actuator execution   → _do_execute() called only on ALLOW")
    print()
    print("Run with --live (and 'python run.py' in another terminal) to see")
    print("the full NATS data flow.")


# ---------------------------------------------------------------------------
# Live demo — requires a running NATS broker.
# ---------------------------------------------------------------------------


async def run_live(nats_url: str, duration_s: float) -> None:
    """Run both plugins against a live NATS broker for `duration_s` seconds."""
    from activelearning.nats_client import EventBus

    print("=" * 60)
    print(" Engram SDK Plugin Examples — live mode")
    print(f" NATS: {nats_url}   duration: {duration_s}s")
    print("=" * 60)

    bus = EventBus(nats_url=nats_url)
    try:
        await bus.connect()
    except Exception as exc:
        print(f"\nCould not connect to NATS at {nats_url}: {exc}")
        print("Is the stack running?  →  python run.py")
        return

    # Track what arrives on the bus so we can print a summary.
    received: list[dict] = []

    async def on_observation(data: dict) -> None:
        obs_data = data.get("data", data)
        temp = obs_data.get("celsius", "?")
        hum = obs_data.get("humidity_pct", "?")
        print(f"  bus ← observation: {temp} °C  /  {hum} % RH")
        received.append(data)

    subject = "observation.sensor.temperature.room"
    await bus.subscribe(subject, on_observation)

    # Start the sensor emit loop.
    sensor = TemperatureSensor(rate_limit_hz=2.0)
    await sensor.start(bus)

    print(f"\nSensor emitting on '{subject}' for {duration_s}s …\n")
    await asyncio.sleep(duration_s)

    await sensor.stop()
    await bus.close()

    print(f"\nTotal observations delivered to bus: {len(received)}")
    print()
    print("Actuator note: submit_and_execute() requires the Kernel service,")
    print("which is already running under the default core profile ('python")
    print("run.py') — replace the direct _do_execute() call with")
    print("actuator.submit_and_execute(proposal) to exercise it.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Engram SDK plugin examples")
    p.add_argument(
        "--live",
        action="store_true",
        help="Connect to a running NATS broker (requires python run.py)",
    )
    p.add_argument(
        "--nats",
        default="nats://localhost:4222",
        metavar="URL",
        help="NATS URL when --live is set (default: nats://localhost:4222)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="How long to run the live sensor demo (default: 5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.live:
        asyncio.run(run_live(args.nats, args.duration))
    else:
        asyncio.run(run_standalone())
