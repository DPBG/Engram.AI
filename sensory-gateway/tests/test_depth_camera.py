"""Tests for the DepthCameraSensor driver and normalisation pipeline.

All tests use mocked depth sources — no physical hardware required.
"""

from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup: sensory-gateway uses a flat src layout (module == "sensors").
# Add the gateway root so imports like `from sensors.depth_camera import …`
# resolve without installing the package.
# ---------------------------------------------------------------------------
_GW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GW_ROOT not in sys.path:
    sys.path.insert(0, _GW_ROOT)

# Bootstrap the activelearning SDK namespace (same trick as neuromorphic tests —
# avoid the aiohttp-pulling __init__ by registering a package stub with __path__).
import types as _types  # noqa: E402

_SDK_SRC = os.path.abspath(os.path.join(_GW_ROOT, "..", "sdk", "src"))
if "activelearning" not in sys.modules:
    _al_pkg = _types.ModuleType("activelearning")
    _al_pkg.__path__ = [os.path.join(_SDK_SRC, "activelearning")]  # type: ignore[attr-defined]
    _al_pkg.__package__ = "activelearning"
    sys.modules["activelearning"] = _al_pkg
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

from sensors.depth_camera import (  # noqa: E402
    DepthCameraSensor,
    DepthDriver,
    NullDepthDriver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


class _FakeBus:
    """Minimal event-bus stub that records published subjects."""

    def __init__(self):
        self.published: list[tuple[str, object]] = []
        self._subs: dict = {}

    async def publish(self, subject: str, data: object) -> None:
        self.published.append((subject, data))

    async def subscribe(self, subject: str, handler) -> None:
        self._subs[subject] = handler

    async def unsubscribe(self, subject: str) -> None:
        self._subs.pop(subject, None)


class _ConstantDepthDriver:
    """Returns a constant depth frame with all pixels set to ``depth_m``."""

    def __init__(self, depth_m: float = 2.5, h: int = 64, w: int = 64):
        self._depth_m = depth_m
        self._h = h
        self._w = w
        self.connected = False
        self.capture_count = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def capture_frame(self) -> np.ndarray:
        self.capture_count += 1
        return np.full((self._h, self._w), self._depth_m, dtype=np.float32)


class _HighResDepthDriver:
    """Returns a 480×640 frame to test downscaling."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def capture_frame(self) -> np.ndarray:
        return np.ones((480, 640), dtype=np.float32) * 1.0  # 1 metre everywhere


# ---------------------------------------------------------------------------
# NullDepthDriver
# ---------------------------------------------------------------------------


class TestNullDepthDriver:
    def test_is_depth_driver(self):
        assert isinstance(NullDepthDriver(), DepthDriver)

    def test_returns_zero_frame(self):
        driver = NullDepthDriver()
        driver.connect()
        frame = driver.capture_frame()
        assert frame.shape == (64, 64)
        assert frame.dtype == np.float32
        assert frame.max() == pytest.approx(0.0)

    def test_disconnect_is_idempotent(self):
        driver = NullDepthDriver()
        driver.connect()
        driver.disconnect()
        driver.disconnect()  # second call must not raise


# ---------------------------------------------------------------------------
# DepthCameraSensor — metadata
# ---------------------------------------------------------------------------


class TestDepthCameraSensorMetadata:
    def test_default_sensor_id(self):
        sensor = DepthCameraSensor()
        assert sensor.sensor_id == "depth.0"

    def test_custom_sensor_id(self):
        sensor = DepthCameraSensor(sensor_id="depth.realsense.1")
        assert sensor.sensor_id == "depth.realsense.1"

    def test_rate_limit_hz(self):
        sensor = DepthCameraSensor(fps=10.0)
        assert sensor.rate_limit_hz == pytest.approx(10.0)

    def test_capability_registered(self):
        sensor = DepthCameraSensor()
        caps = sensor.get_capabilities()
        assert len(caps) == 1
        assert caps[0].name == "depth_capture"

    def test_describe_metadata(self):
        sensor = DepthCameraSensor(sensor_id="depth.test")
        meta = sensor.describe()
        assert meta.sensor_id == "depth.test"
        assert "depth" in meta.description.lower()


# ---------------------------------------------------------------------------
# DepthCameraSensor — capture (no hardware)
# ---------------------------------------------------------------------------


class TestDepthCameraSensorCapture:
    def test_capture_returns_4096_floats(self):
        driver = _ConstantDepthDriver(depth_m=1.0)
        sensor = DepthCameraSensor(driver=driver, max_depth_m=5.0)
        sensor._bus = _FakeBus()
        result = _run(sensor.capture())
        assert len(result) == 64 * 64

    def test_normalisation_half_range(self):
        """2.5 m with max 5.0 m → all pixels ≈ 0.5."""
        driver = _ConstantDepthDriver(depth_m=2.5)
        sensor = DepthCameraSensor(driver=driver, max_depth_m=5.0)
        sensor._bus = _FakeBus()
        result = _run(sensor.capture())
        assert all(abs(v - 0.5) < 1e-5 for v in result)

    def test_normalisation_full_range(self):
        """5.0 m with max 5.0 m → all pixels == 1.0."""
        driver = _ConstantDepthDriver(depth_m=5.0)
        sensor = DepthCameraSensor(driver=driver, max_depth_m=5.0)
        sensor._bus = _FakeBus()
        result = _run(sensor.capture())
        assert all(abs(v - 1.0) < 1e-5 for v in result)

    def test_values_clipped_beyond_max(self):
        """Depth beyond max_depth_m must be clipped to 1.0."""
        driver = _ConstantDepthDriver(depth_m=100.0)
        sensor = DepthCameraSensor(driver=driver, max_depth_m=5.0)
        sensor._bus = _FakeBus()
        result = _run(sensor.capture())
        assert all(abs(v - 1.0) < 1e-5 for v in result)

    def test_null_driver_returns_zeros(self):
        sensor = DepthCameraSensor(driver=NullDepthDriver(), max_depth_m=5.0)
        sensor._bus = _FakeBus()
        result = _run(sensor.capture())
        assert all(v == pytest.approx(0.0) for v in result)

    def test_high_res_frame_downscaled(self):
        """480×640 input must be resized to 64×64 (4096 outputs)."""
        driver = _HighResDepthDriver()
        sensor = DepthCameraSensor(driver=driver, max_depth_m=5.0)
        sensor._bus = _FakeBus()
        result = _run(sensor.capture())
        assert len(result) == 4096

    def test_driver_capture_called_each_time(self):
        driver = _ConstantDepthDriver()
        sensor = DepthCameraSensor(driver=driver)
        sensor._bus = _FakeBus()
        _run(sensor.capture())
        _run(sensor.capture())
        assert driver.capture_count == 2


# ---------------------------------------------------------------------------
# DepthCameraSensor — lifecycle
# ---------------------------------------------------------------------------


class TestDepthCameraSensorLifecycle:
    def test_start_calls_driver_connect(self):
        driver = _ConstantDepthDriver()
        sensor = DepthCameraSensor(driver=driver)
        bus = _FakeBus()

        async def _start_stop():
            await sensor.start(bus)
            await sensor.stop()

        _run(_start_stop())
        assert driver.connected is False  # disconnected after stop

    def test_start_then_stop_connected_flag(self):
        driver = _ConstantDepthDriver()
        sensor = DepthCameraSensor(driver=driver)
        bus = _FakeBus()

        async def _check():
            await sensor.start(bus)
            assert driver.connected is True
            await sensor.stop()
            assert driver.connected is False

        _run(_check())

    def test_observation_published_after_capture(self):
        driver = _ConstantDepthDriver(depth_m=1.0)
        sensor = DepthCameraSensor(driver=driver, fps=100.0)
        bus = _FakeBus()

        async def _one_emit():
            sensor._bus = bus
            data = await sensor.capture()
            await sensor.emit(data)

        _run(_one_emit())
        assert len(bus.published) == 1
        subject, obs = bus.published[0]
        assert "depth.0" in subject


# ---------------------------------------------------------------------------
# Encoding provenance mapping (neuromorphic integration check)
# ---------------------------------------------------------------------------


class TestProvenanceMapping:
    def test_sensor_depth_maps_to_visual(self):
        """Load _PROVENANCE_MAP directly from encoding.py without triggering
        neuromorphic/__init__.py (which pulls in scipy/mujoco not installed here).
        """
        import importlib.util

        enc_path = os.path.abspath(
            os.path.join(_GW_ROOT, "..", "neuromorphic", "src", "neuromorphic", "encoding.py")
        )
        spec = importlib.util.spec_from_file_location("neuromorphic.encoding", enc_path)
        mod = importlib.util.module_from_spec(spec)
        # Stub out the neuromorphic package so relative imports inside encoding.py
        # resolve without triggering the full package chain.
        import types as _t

        neuro_pkg = _t.ModuleType("neuromorphic")
        neuro_pkg.__path__ = [os.path.dirname(enc_path)]
        sys.modules.setdefault("neuromorphic", neuro_pkg)
        sys.modules["neuromorphic.encoding"] = mod
        spec.loader.exec_module(mod)

        pmap = mod._PROVENANCE_MAP
        assert pmap.get("sensor.depth") == "visual", "sensor.depth must map to visual"
        assert pmap.get("sensor.realsense") == "visual", "sensor.realsense must map to visual"
