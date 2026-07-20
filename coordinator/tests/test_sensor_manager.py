"""Tests for SensorManager hardware detection."""

import asyncio
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

_SM_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "coordinator", "sensor_manager.py")
_spec = importlib.util.spec_from_file_location("coord_sensor_manager", _SM_PATH)
sm = importlib.util.module_from_spec(_spec)
sys.modules["coord_sensor_manager"] = sm
_spec.loader.exec_module(sm)

SensorManager = sm.SensorManager
SensorType = sm.SensorType
_looks_like_imu_reading = sm._looks_like_imu_reading
_looks_like_nmea_sentence = sm._looks_like_nmea_sentence
_looks_like_ultrasonic_reading = sm._looks_like_ultrasonic_reading


def _run(coro):
    return asyncio.run(coro)


def _fake_serial_modules(
    *, ports: list[str] | None = None, serial_instance: MagicMock | None = None
):
    """Inject a fake pyserial module tree into sys.modules."""
    list_ports = MagicMock()
    list_ports.comports.return_value = [SimpleNamespace(device=port) for port in (ports or [])]
    serial_tools = ModuleType("serial.tools")
    serial_tools.list_ports = list_ports  # type: ignore[attr-defined]

    serial_mod = ModuleType("serial")
    serial_mod.tools = serial_tools  # type: ignore[attr-defined]
    serial_mod.Serial = MagicMock(return_value=serial_instance or MagicMock())

    return {
        "serial": serial_mod,
        "serial.tools": serial_tools,
        "serial.tools.list_ports": list_ports,
    }


def test_looks_like_imu_reading_accepts_short_form():
    assert _looks_like_imu_reading({"ax": 0.1, "ay": -0.05, "az": 9.81}) is True


def test_looks_like_imu_reading_accepts_long_form():
    assert _looks_like_imu_reading({"accel_x": 0.1, "accel_y": 0.0, "accel_z": 1.0}) is True


def test_looks_like_imu_reading_rejects_non_imu_payload():
    assert _looks_like_imu_reading({"temperature": 22.5, "humidity": 40}) is False


def test_looks_like_imu_reading_rejects_single_axis():
    assert _looks_like_imu_reading({"ax": 1.0}) is False


@patch.object(SensorManager, "_probe_imu_port_sync", return_value=True)
def test_detect_imu_registers_serial_device(_mock_probe):
    manager = SensorManager()
    with patch.dict(sys.modules, _fake_serial_modules(ports=["/dev/ttyUSB0"])):
        assert _run(manager._detect_imu()) is True
    sensors = manager.get_available_sensors(SensorType.IMU)
    assert len(sensors) == 1
    assert sensors[0].sensor_id == "imu_ttyUSB0"
    assert "accel" in sensors[0].capabilities


@patch.object(SensorManager, "_probe_imu_port_sync", return_value=False)
def test_detect_imu_returns_false_when_no_compatible_ports(_mock_probe):
    manager = SensorManager()
    with patch.dict(sys.modules, _fake_serial_modules(ports=["/dev/ttyUSB0"])):
        assert _run(manager._detect_imu()) is False
    assert manager.get_available_sensors(SensorType.IMU) == []


def test_detect_imu_returns_false_when_no_ports_available():
    manager = SensorManager()
    with patch.dict(sys.modules, _fake_serial_modules(ports=[])):
        assert _run(manager._detect_imu()) is False
    assert manager.get_available_sensors(SensorType.IMU) == []


def test_detect_imu_skips_when_pyserial_missing():
    manager = SensorManager()
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "serial":
            raise ImportError("no pyserial")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import):
        assert _run(manager._detect_imu()) is False


def test_probe_imu_port_sync_parses_json_line():
    fake_serial = MagicMock()
    fake_serial.readline.return_value = b'{"ax": 0.1, "ay": 0.0, "az": 9.8}\n'
    with patch.dict(sys.modules, _fake_serial_modules(serial_instance=fake_serial)):
        assert SensorManager._probe_imu_port_sync("/dev/ttyUSB0") is True
    fake_serial.close.assert_called_once()


def test_probe_imu_port_sync_ignores_malformed_json():
    fake_serial = MagicMock()
    fake_serial.readline.return_value = b"not-json\n"
    with patch.dict(sys.modules, _fake_serial_modules(serial_instance=fake_serial)):
        assert SensorManager._probe_imu_port_sync("/dev/ttyUSB0") is False


def test_probe_imu_port_sync_skips_banner_before_valid_json():
    fake_serial = MagicMock()
    fake_serial.readline.side_effect = [
        b"IMU firmware v1.0\n",
        b'{"ax": 0.1, "ay": 0.0, "az": 9.8}\n',
    ]
    with patch.dict(sys.modules, _fake_serial_modules(serial_instance=fake_serial)):
        assert SensorManager._probe_imu_port_sync("/dev/ttyUSB0") is True


def test_looks_like_nmea_sentence_accepts_gpgga():
    assert (
        _looks_like_nmea_sentence(
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
        )
        is True
    )


def test_looks_like_nmea_sentence_accepts_gnrmc():
    assert (
        _looks_like_nmea_sentence(
            "$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
        )
        is True
    )


def test_looks_like_nmea_sentence_rejects_non_nmea():
    assert _looks_like_nmea_sentence("GPS firmware v2.0") is False
    assert _looks_like_nmea_sentence('{"lat": 48.1, "lon": 11.5}') is False


@patch.object(SensorManager, "_probe_gps_port_sync", return_value=True)
def test_detect_gps_registers_serial_device(_mock_probe):
    manager = SensorManager()
    with patch.dict(sys.modules, _fake_serial_modules(ports=["/dev/ttyUSB1"])):
        assert _run(manager._detect_gps()) is True
    sensors = manager.get_available_sensors(SensorType.GPS)
    assert len(sensors) == 1
    assert sensors[0].sensor_id == "gps_ttyUSB1"
    assert "nmea" in sensors[0].capabilities


@patch.object(SensorManager, "_probe_gps_port_sync", return_value=False)
def test_detect_gps_returns_false_when_no_compatible_ports(_mock_probe):
    manager = SensorManager()
    with patch.dict(sys.modules, _fake_serial_modules(ports=["/dev/ttyUSB1"])):
        assert _run(manager._detect_gps()) is False
    assert manager.get_available_sensors(SensorType.GPS) == []


def test_detect_gps_skips_when_pyserial_missing():
    manager = SensorManager()
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "serial":
            raise ImportError("no pyserial")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import):
        assert _run(manager._detect_gps()) is False


def test_probe_gps_port_sync_parses_nmea_line():
    fake_serial = MagicMock()
    fake_serial.readline.return_value = (
        b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\n"
    )
    with patch.dict(sys.modules, _fake_serial_modules(serial_instance=fake_serial)):
        assert SensorManager._probe_gps_port_sync("/dev/ttyUSB1") is True
    fake_serial.close.assert_called_once()


def test_probe_gps_port_sync_skips_banner_before_valid_nmea():
    fake_serial = MagicMock()
    fake_serial.readline.side_effect = [
        b"u-blox AG GPS receiver\n",
        b"$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A\n",
    ]
    with patch.dict(sys.modules, _fake_serial_modules(serial_instance=fake_serial)):
        assert SensorManager._probe_gps_port_sync("/dev/ttyUSB1") is True


def test_looks_like_ultrasonic_reading_accepts_distance_cm():
    assert _looks_like_ultrasonic_reading({"distance_cm": 42.5}) is True


def test_looks_like_ultrasonic_reading_accepts_range_mm():
    assert _looks_like_ultrasonic_reading({"range_mm": 850, "unit": "mm"}) is True


def test_looks_like_ultrasonic_reading_rejects_negative_and_non_distance():
    assert _looks_like_ultrasonic_reading({"distance_cm": -1}) is False
    assert _looks_like_ultrasonic_reading({"temperature": 22.5}) is False
    assert _looks_like_ultrasonic_reading({"ax": 0.1, "ay": 0.0, "az": 9.8}) is False


@patch.object(SensorManager, "_probe_ultrasonic_port_sync", return_value=True)
def test_detect_ultrasonic_registers_serial_device(_mock_probe):
    manager = SensorManager()
    with patch.dict(sys.modules, _fake_serial_modules(ports=["/dev/ttyUSB2"])):
        assert _run(manager._detect_ultrasonic()) is True
    sensors = manager.get_available_sensors(SensorType.ULTRASONIC)
    assert len(sensors) == 1
    assert sensors[0].sensor_id == "ultrasonic_ttyUSB2"
    assert "distance" in sensors[0].capabilities


@patch.object(SensorManager, "_probe_ultrasonic_port_sync", return_value=False)
def test_detect_ultrasonic_returns_false_when_no_compatible_ports(_mock_probe):
    manager = SensorManager()
    with patch.dict(sys.modules, _fake_serial_modules(ports=["/dev/ttyUSB2"])):
        assert _run(manager._detect_ultrasonic()) is False
    assert manager.get_available_sensors(SensorType.ULTRASONIC) == []


def test_detect_ultrasonic_skips_when_pyserial_missing():
    manager = SensorManager()
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "serial":
            raise ImportError("no pyserial")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import):
        assert _run(manager._detect_ultrasonic()) is False


def test_probe_ultrasonic_port_sync_parses_json_line():
    fake_serial = MagicMock()
    fake_serial.readline.return_value = b'{"distance_cm": 37.2}\n'
    with patch.dict(sys.modules, _fake_serial_modules(serial_instance=fake_serial)):
        assert SensorManager._probe_ultrasonic_port_sync("/dev/ttyUSB2") is True
    fake_serial.close.assert_called_once()


def test_probe_ultrasonic_port_sync_skips_banner_before_valid_json():
    fake_serial = MagicMock()
    fake_serial.readline.side_effect = [
        b"HC-SR04 ready\n",
        b'{"ultrasonic_cm": 19.5}\n',
    ]
    with patch.dict(sys.modules, _fake_serial_modules(serial_instance=fake_serial)):
        assert SensorManager._probe_ultrasonic_port_sync("/dev/ttyUSB2") is True
