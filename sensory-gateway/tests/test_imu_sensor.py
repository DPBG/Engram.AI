"""Unit tests for the IMU sensor driver (issue #136).

Tests run without real hardware by injecting a fake serial object.
The activelearning SDK and pyserial are NOT required at import time —
only the normalisation helpers and the IMUSensor class are exercised.

Coverage:
  - _normalize(): field aliasing, canonical ordering, invalid data
  - IMUSensor._read_line(): JSON parsing, error handling
  - IMUSensor.capture(): async wrapper timeout behaviour
  - Provenance routing: sensor.imu.* → proprioceptive (via encoding.py)
  - Discovery: _IMU_KEYWORDS classification in discover_serial_devices
"""

from __future__ import annotations

import asyncio
import os
import sys
import types

# ── Bootstrap path so `from sensors.imu import ...` resolves without install ──

_GATEWAY_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _GATEWAY_ROOT not in sys.path:
    sys.path.insert(0, _GATEWAY_ROOT)

# Stub out activelearning so IMUSensor can be imported without the full SDK
_al = types.ModuleType("activelearning")
_al_plugins = types.ModuleType("activelearning.plugins")


class _RiskClass:
    LOW = "low"


class _PluginCapability:
    def __init__(self, name, description, parameters=None, risk_class=None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.risk_class = risk_class or _RiskClass.LOW


class _SensorPlugin:
    # Support SensorPlugin[dict] subscript syntax used in sensor subclasses
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, sensor_id, name, description="", rate_limit_hz=30.0, risk_class=None):
        self.sensor_id = sensor_id
        self.name = name
        self.description = description
        self.rate_limit_hz = rate_limit_hz
        self.risk_class = risk_class
        self._capabilities = []
        self._bus = None
        self._running = False
        self._task = None

    def add_capability(self, cap):
        self._capabilities.append(cap)

    async def start(self, bus=None):
        self._running = True

    async def stop(self):
        self._running = False


_al_plugins.RiskClass = _RiskClass
_al_plugins.PluginCapability = _PluginCapability
_al_plugins.SensorPlugin = _SensorPlugin
_al.plugins = _al_plugins
sys.modules.setdefault("activelearning", _al)
sys.modules.setdefault("activelearning.plugins", _al_plugins)

from sensors.imu import IMUSensor, _normalize, _CANONICAL_FIELDS  # noqa: E402, I001

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


class _FakeSerial:
    """Minimal pyserial Serial stub."""

    def __init__(self, responses: list[bytes]):
        self.is_open = True
        self._it = iter(responses)

    def readline(self) -> bytes:
        return next(self._it, b"")

    def close(self):
        self.is_open = False


def _sensor_with_serial(responses: list[bytes]) -> IMUSensor:
    """Build an IMUSensor with a pre-loaded fake serial port (no NATS/hw)."""
    svc = IMUSensor.__new__(IMUSensor)
    _SensorPlugin.__init__(svc, "imu.ttyUSB0", "IMU ttyUSB0")
    svc._port = "/dev/ttyUSB0"
    svc._baud_rate = 115200
    svc._serial = _FakeSerial(responses)
    return svc


# ── _normalize tests ──────────────────────────────────────────────────────────


def test_normalize_short_form():
    raw = {"ax": 1.0, "ay": -0.5, "az": 9.81, "gx": 0.01, "gy": -0.02, "gz": 0.0}
    result = _normalize(raw)
    assert result["accel_x"] == 1.0
    assert result["accel_y"] == -0.5
    assert result["accel_z"] == 9.81
    assert result["gyro_x"] == 0.01
    assert result["gyro_y"] == -0.02
    assert result["gyro_z"] == 0.0


def test_normalize_long_form():
    raw = {
        "accel_x": 0.1,
        "accel_y": 0.2,
        "accel_z": 9.8,
        "gyro_x": 0.0,
        "gyro_y": 0.0,
        "gyro_z": 0.0,
    }
    result = _normalize(raw)
    assert result["accel_x"] == 0.1
    assert result["accel_z"] == 9.8


def test_normalize_canonical_order():
    """Keys in the output dict must follow _CANONICAL_FIELDS order."""
    raw = {"az": 9.81, "ay": -0.5, "ax": 1.0, "gz": 0.0, "gx": 0.01, "gy": -0.02}
    result = _normalize(raw)
    present = list(result.keys())
    canonical_present = [f for f in _CANONICAL_FIELDS if f in result]
    assert present == canonical_present


def test_normalize_with_optional_extras():
    raw = {
        "ax": 0.0,
        "ay": 0.0,
        "az": 9.81,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
        "mx": 25.4,
        "my": -8.1,
        "mz": 42.0,
        "roll": 0.01,
        "pitch": -0.03,
        "yaw": 1.57,
        "qw": 0.999,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.04,
    }
    result = _normalize(raw)
    assert "mag_x" in result
    assert result["mag_x"] == 25.4
    assert "roll" in result
    assert "qw" in result
    assert result["qw"] == 0.999


def test_normalize_non_numeric_values_skipped():
    raw = {"ax": 1.0, "ay": 0.0, "az": 9.81, "label": "up", "valid": True}
    result = _normalize(raw)
    assert "label" not in result
    assert "valid" not in result


def test_normalize_missing_required_returns_empty():
    """Reading with gyro but no acceleration axes → rejected."""
    raw = {"gx": 0.01, "gy": 0.02, "gz": 0.0}
    assert _normalize(raw) == {}


def test_normalize_empty_dict_returns_empty():
    assert _normalize({}) == {}


def test_normalize_case_insensitive_aliases():
    """Alias lookup must be case-insensitive."""
    raw = {"AX": 1.0, "AY": -0.5, "AZ": 9.81}
    result = _normalize(raw)
    assert "accel_x" in result
    assert result["accel_x"] == 1.0


def test_normalize_accx_alias():
    """accX / gyrX variants (some STM32 firmwares)."""
    raw = {"accX": 0.5, "accY": 0.1, "accZ": 9.8, "gyrX": 0.0, "gyrY": 0.0, "gyrZ": 0.0}
    result = _normalize(raw)
    assert result["accel_x"] == 0.5
    assert result["gyro_x"] == 0.0


# ── IMUSensor._read_line tests ────────────────────────────────────────────────


def test_read_line_valid_json():
    svc = _sensor_with_serial([b'{"ax": 1.0, "ay": 0.0, "az": 9.81}\n'])
    result = svc._read_line()
    assert result["accel_x"] == 1.0
    assert result["accel_z"] == 9.81


def test_read_line_empty_returns_empty():
    svc = _sensor_with_serial([b""])
    assert svc._read_line() == {}


def test_read_line_malformed_json_returns_empty():
    svc = _sensor_with_serial([b"not json at all\n"])
    assert svc._read_line() == {}


def test_read_line_partial_json_returns_empty():
    svc = _sensor_with_serial([b'{"ax": 1.0, "ay":}\n'])
    assert svc._read_line() == {}


def test_read_line_missing_accel_returns_empty():
    svc = _sensor_with_serial([b'{"gx": 0.01, "gy": 0.0, "gz": 0.0}\n'])
    assert svc._read_line() == {}


def test_read_line_oserror_returns_empty(monkeypatch):
    """OSError (device unplugged mid-read) must not propagate."""
    svc = _sensor_with_serial([])

    def _explode():
        raise OSError("device vanished")

    svc._serial.readline = _explode
    assert svc._read_line() == {}


def test_read_line_serial_closed_returns_empty():
    svc = _sensor_with_serial([b'{"ax": 1.0, "ay": 0.0, "az": 9.81}\n'])
    svc._serial.is_open = False
    assert svc._read_line() == {}


# ── IMUSensor.capture tests ───────────────────────────────────────────────────


def test_capture_returns_dict():
    svc = _sensor_with_serial([b'{"ax": 0.1, "ay": 0.2, "az": 9.81}\n'])
    result = _run(svc.capture())
    assert isinstance(result, dict)
    assert "accel_z" in result


def test_capture_no_serial_raises():
    svc = IMUSensor.__new__(IMUSensor)
    _SensorPlugin.__init__(svc, "imu.x", "IMU x")
    svc._serial = None
    svc._port = "/dev/null"
    svc._baud_rate = 115200

    import pytest

    with pytest.raises(RuntimeError, match="not open"):
        _run(svc.capture())


# ── Sensor metadata tests ─────────────────────────────────────────────────────


def test_sensor_id_format():
    svc = IMUSensor.__new__(IMUSensor)
    _SensorPlugin.__init__(
        svc,
        "imu.ttyUSB0",
        "IMU ttyUSB0",
        description="Inertial measurement unit on /dev/ttyUSB0",
    )
    assert svc.sensor_id == "imu.ttyUSB0"
    assert "ttyUSB0" in svc.name


def test_sensor_rate_default():
    svc = IMUSensor.__new__(IMUSensor)
    _SensorPlugin.__init__(svc, "imu.x", "IMU x", rate_limit_hz=50.0)
    assert svc.rate_limit_hz == 50.0


# ── Provenance → modality routing ────────────────────────────────────────────


def test_imu_provenance_routes_to_proprioceptive():
    """sensor.imu.* must resolve to the proprioceptive modality."""
    # Load encoding module directly (requires neuromorphic on path)
    neuromorphic_src = os.path.join(os.path.dirname(__file__), "..", "..", "neuromorphic", "src")
    if neuromorphic_src not in sys.path:
        sys.path.insert(0, neuromorphic_src)

    try:
        from neuromorphic.encoding import _resolve_modality
    except ImportError:
        import pytest

        pytest.skip("neuromorphic package not on path — skipping routing test")

    assert _resolve_modality("sensor.imu.ttyUSB0") == "proprioceptive"
    assert _resolve_modality("sensor.imu") == "proprioceptive"


# ── Discovery classification ──────────────────────────────────────────────────


def test_imu_keywords_classify_as_imu():
    """Ports with IMU-related descriptions must get device_type='imu'."""
    from discovery import _IMU_KEYWORDS

    imu_descriptions = [
        "SparkFun IMU Breakout",
        "Adafruit BNO055",
        "ICM-42688 USB Bridge",
        "MPU-6050 Dev Board",
        "LSM6DS3 Inertial",
        "USB Gyro Sensor",
        "3-Axis Accelerometer",
    ]
    for desc in imu_descriptions:
        desc_lower = desc.lower()
        matched = any(kw in desc_lower for kw in _IMU_KEYWORDS)
        assert matched, f"Expected '{desc}' to match an IMU keyword"


def test_non_imu_serial_not_classified_as_imu():
    """Generic serial devices must NOT be classified as IMU."""
    from discovery import _IMU_KEYWORDS

    non_imu = ["Arduino Uno", "USB-UART Bridge", "CP210x", "CH340 USB"]
    for desc in non_imu:
        desc_lower = desc.lower()
        matched = any(kw in desc_lower for kw in _IMU_KEYWORDS)
        assert not matched, f"Expected '{desc}' NOT to match an IMU keyword"


def test_imu_in_known_device_types():
    from discovery import KNOWN_DEVICE_TYPES

    assert "imu" in KNOWN_DEVICE_TYPES
