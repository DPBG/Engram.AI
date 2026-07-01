"""Unit tests for the IMU sensor driver and its discovery wiring.

Runs against a mocked IMU device (an injected ``reader`` callable) — no
hardware, no optional deps. Uses ``asyncio.run`` (no pytest-asyncio) so it runs
under the bare-pytest lanes, mirroring the other service test suites.

Requires ``PYTHONPATH=sensory-gateway`` (flat ``sensors`` / ``discovery``
layout) plus the editable SDK for ``activelearning.plugins``.
"""

import asyncio
from unittest.mock import AsyncMock

from discovery import KNOWN_DEVICE_TYPES, discover_all, discover_imus
from sensors.imu import IMU_FIELDS, ImuSensor, normalize_reading

_RAW = {
    "accel_x": 0.1,
    "accel_y": 0.0,
    "accel_z": 9.81,
    "gyro_x": 0.01,
    "gyro_y": 0.0,
    "gyro_z": -0.02,
    "roll": 0.0,
    "pitch": 0.05,
    "yaw": 1.57,
}


# ── normalize_reading ─────────────────────────────────────────────────────────


def test_normalize_reading_returns_full_float_vector():
    out = normalize_reading(_RAW)
    assert tuple(out.keys()) == IMU_FIELDS  # canonical order preserved
    assert all(isinstance(v, float) for v in out.values())
    assert out["accel_z"] == 9.81 and out["yaw"] == 1.57
    # The encoder's dict path keys off numeric values; every field must be one.
    numeric = [v for v in out.values() if isinstance(v, (int, float))]
    assert len(numeric) == len(IMU_FIELDS)


def test_normalize_reading_accepts_aliases_and_fills_missing():
    out = normalize_reading({"ax": 1.0, "gy": 2.0, "heading": 0.5})
    assert out["accel_x"] == 1.0  # ax -> accel_x
    assert out["gyro_y"] == 2.0  # gy -> gyro_y
    assert out["yaw"] == 0.5  # heading -> yaw
    assert out["accel_z"] == 0.0  # missing -> 0.0


def test_normalize_reading_drops_non_finite_and_nonnumeric():
    out = normalize_reading({"accel_x": float("nan"), "accel_y": "oops", "accel_z": float("inf")})
    assert out["accel_x"] == 0.0
    assert out["accel_y"] == 0.0
    assert out["accel_z"] == 0.0


# ── capture ───────────────────────────────────────────────────────────────────


def test_capture_returns_normalized_reading():
    sensor = ImuSensor(device_id="0", reader=lambda: dict(_RAW))
    out = asyncio.run(sensor.capture())
    assert out == normalize_reading(_RAW)


def test_capture_skips_when_reader_returns_none():
    sensor = ImuSensor(device_id="0", reader=lambda: None)
    assert asyncio.run(sensor.capture()) is None


def test_capture_survives_faulty_reader():
    def boom():
        raise OSError("device yanked")

    sensor = ImuSensor(device_id="0", reader=boom)
    assert asyncio.run(sensor.capture()) is None  # error swallowed, loop survives


# ── lifecycle / provenance ────────────────────────────────────────────────────


def test_start_without_reader_raises():
    sensor = ImuSensor(device_id="0", reader=None)
    try:
        asyncio.run(sensor.start())
        raise AssertionError("expected RuntimeError for absent hardware")
    except RuntimeError as e:
        assert "no reader" in str(e).lower()


def test_emit_publishes_proprioceptive_observation():
    async def run():
        sensor = ImuSensor(device_id="waist", reader=lambda: dict(_RAW))
        bus = AsyncMock()
        sensor._bus = bus  # simulate a started sensor without a live NATS loop

        await sensor.emit(await sensor.capture())

        bus.publish.assert_awaited_once()
        subject, observation = bus.publish.call_args.args
        assert subject == "observation.imu.waist"
        # provenance sensor.imu.* -> proprioceptive modality in encoding.py
        assert observation.provenance == "sensor.imu.waist"
        assert set(observation.data.keys()) == set(IMU_FIELDS)

    asyncio.run(run())


def test_stop_closes_reader_resource():
    async def run():
        closed = {"n": 0}

        def reader():
            return dict(_RAW)

        reader.close = lambda: closed.__setitem__("n", closed["n"] + 1)
        sensor = ImuSensor(device_id="0", reader=reader)
        await sensor.stop()  # stop() is safe before start(); must close the reader
        assert closed["n"] == 1

    asyncio.run(run())


# ── discovery ─────────────────────────────────────────────────────────────────


def test_imu_is_a_known_device_type():
    assert "imu" in KNOWN_DEVICE_TYPES


def test_discover_imus_empty_when_env_unset(monkeypatch):
    monkeypatch.delenv("ENGRAM_IMU_DEVICES", raising=False)
    assert discover_imus() == []  # graceful no-op, no hardware declared


def test_discover_imus_parses_env(monkeypatch):
    monkeypatch.setenv("ENGRAM_IMU_DEVICES", "waist@/dev/ttyUSB0,/dev/ttyUSB1")
    devices = discover_imus()

    assert len(devices) == 2
    assert all(d.device_type == "imu" for d in devices)
    assert devices[0].device_id == "imu:waist"
    assert devices[0].metadata == {"id": "waist", "port": "/dev/ttyUSB0"}
    # bare port entry gets a positional id
    assert devices[1].metadata["port"] == "/dev/ttyUSB1"


def test_discover_all_includes_imus(monkeypatch):
    monkeypatch.setenv("ENGRAM_IMU_DEVICES", "head@/dev/ttyUSB2")
    devices = discover_all(skip_cameras=True, skip_microphones=True, skip_serial=True)
    assert [d.device_type for d in devices] == ["imu"]
    assert devices[0].device_id == "imu:head"
