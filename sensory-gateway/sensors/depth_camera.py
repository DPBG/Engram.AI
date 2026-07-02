"""
Depth-camera sensor — captures per-pixel depth maps and publishes them to the
neuromorphic brain as ``observation.depth.<sensor_id>``.

Depth data (64×64, normalised 0–1) is routed to the *visual* modality by the
encoding pipeline so it binds temporally with concurrent RGB camera and audio
activity, enriching cross-modal binding (Invariant 4).

Two hardware backends are provided via the ``DepthDriver`` Protocol:

``RealSenseDriver``
    Intel RealSense D400 series (D415, D435, D455 …).
    Requires ``pyrealsense2`` (``pip install pyrealsense2``); import is deferred
    to ``connect()`` so the module loads without the library.

``OpenCVDepthDriver``
    Generic V4L2/DirectShow depth cameras exposed as a second OpenCV
    VideoCapture stream (e.g. structured-light cameras, ZED with OCV driver).
    Uses only ``opencv-python`` which is already a gateway dependency.

``NullDepthDriver``
    Zero-frame stub for offline development and unit tests.

Usage::

    from sensors.depth_camera import DepthCameraSensor, RealSenseDriver

    sensor = DepthCameraSensor(driver=RealSenseDriver(serial=""))
    await sensor.start(bus)
    # … brain receives observation.depth.realsense.0 →  visual modality …
    await sensor.stop()
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin

logger = logging.getLogger(__name__)

# Depth values beyond this distance (metres) are clipped before normalisation.
# 5 m is appropriate for indoor embodied scenarios; override via max_depth_m.
_DEFAULT_MAX_DEPTH_M: float = 5.0

# Output resolution — matches the RGB camera pipeline for time-alignment.
_OUT_W: int = 64
_OUT_H: int = 64


# ---------------------------------------------------------------------------
# DepthDriver Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DepthDriver(Protocol):
    """Hardware interface for a depth camera.

    All methods are synchronous — ``DepthCameraSensor`` calls them from the
    asyncio event loop via ``run_in_executor`` so they must not schedule tasks.
    """

    def connect(self) -> None:
        """Open the hardware connection and start streaming."""
        ...

    def disconnect(self) -> None:
        """Stop streaming and release hardware resources."""
        ...

    def capture_frame(self) -> np.ndarray:
        """Return one depth frame as a float32 array (H×W, metres).

        The array may be any resolution; ``DepthCameraSensor`` resizes it to
        64×64.  Values of 0.0 indicate invalid/missing depth.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete drivers
# ---------------------------------------------------------------------------


class NullDepthDriver:
    """No-op depth driver for offline development and unit tests.

    Returns a flat zero-depth frame on every ``capture_frame()`` call.
    """

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def capture_frame(self) -> np.ndarray:
        return np.zeros((_OUT_H, _OUT_W), dtype=np.float32)


class RealSenseDriver:
    """Intel RealSense D400-series depth driver.

    Requires ``pyrealsense2`` (``pip install pyrealsense2``).  The import is
    deferred to ``connect()`` so the rest of the gateway starts normally when
    the library is absent.

    Args:
        serial: Device serial number (empty string = first available device).
        width:  Depth stream width in pixels (default 640).
        height: Depth stream height in pixels (default 480).
        fps:    Streaming frame rate (default 30).
    """

    def __init__(
        self,
        serial: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline: object = None
        self._align: object = None

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required for RealSenseDriver. "
                "Install it with: pip install pyrealsense2"
            ) from exc

        config = rs.config()
        if self._serial:
            config.enable_device(self._serial)
        config.enable_stream(
            rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
        )
        pipeline = rs.pipeline()
        pipeline.start(config)
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.depth)
        logger.info(
            "RealSenseDriver: streaming %dx%d @ %d fps (serial=%r)",
            self._width, self._height, self._fps, self._serial or "auto",
        )

    def disconnect(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._align = None

    def capture_frame(self) -> np.ndarray:
        if self._pipeline is None:
            return np.zeros((_OUT_H, _OUT_W), dtype=np.float32)
        frames = self._pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return np.zeros((_OUT_H, _OUT_W), dtype=np.float32)
        # get_distance() returns metres; asanyarray gives millimetre uint16 —
        # convert to metres manually for consistency.
        import pyrealsense2 as rs  # type: ignore[import]
        data = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        scale = depth_frame.get_units()  # typically 0.001 (mm → m)
        return data * scale


class OpenCVDepthDriver:
    """Generic OpenCV depth camera driver.

    Works with any camera that exposes a depth channel via a second
    ``cv2.VideoCapture`` index (e.g. structured-light cameras, ZED SDK with
    OpenCV bridge).  The raw pixel values are divided by ``depth_scale`` to
    convert to metres before normalisation.

    Args:
        device_index: OpenCV capture index for the depth stream (default 1,
                      since index 0 is usually the colour camera).
        depth_scale:  Raw pixel → metres divisor (default 1000 for mm-scale).
    """

    def __init__(self, device_index: int = 1, depth_scale: float = 1000.0) -> None:
        self._device_index = device_index
        self._depth_scale = depth_scale
        self._cap: object = None

    def connect(self) -> None:
        import cv2  # already a gateway dependency

        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open depth camera at OpenCV device index {self._device_index}"
            )
        self._cap = cap
        logger.info("OpenCVDepthDriver: opened device index %d", self._device_index)

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def capture_frame(self) -> np.ndarray:
        if self._cap is None:
            return np.zeros((_OUT_H, _OUT_W), dtype=np.float32)
        ret, frame = self._cap.read()
        if not ret:
            return np.zeros((_OUT_H, _OUT_W), dtype=np.float32)
        # Convert to single-channel float metres
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return gray.astype(np.float32) / self._depth_scale


# ---------------------------------------------------------------------------
# DepthCameraSensor
# ---------------------------------------------------------------------------


class DepthCameraSensor(SensorPlugin[list]):
    """Depth-camera sensor that publishes 64×64 normalised depth maps.

    Each observation is a flat list of 4096 floats (64×64) in [0, 1], where
    1.0 represents ``max_depth_m`` metres and 0.0 represents the sensor minimum
    or an invalid reading.  Published with provenance ``sensor.depth`` so the
    neuromorphic encoding pipeline routes it to the *visual* modality alongside
    RGB camera data, enabling temporal correlation for cross-modal binding.

    Args:
        driver:      Hardware backend (default: ``NullDepthDriver``).
        sensor_id:   Plugin identifier (default: ``"depth.0"``).
        fps:         Capture rate in Hz (default: 5.0, matching CameraSensor).
        max_depth_m: Depth values beyond this (metres) are clipped to 1.0.
    """

    def __init__(
        self,
        driver: Optional[DepthDriver] = None,
        sensor_id: str = "depth.0",
        fps: float = 5.0,
        max_depth_m: float = _DEFAULT_MAX_DEPTH_M,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            name=f"Depth camera ({sensor_id})",
            description=(
                "Captures per-pixel depth maps (64×64, metres normalised to [0,1]). "
                "Routed to the visual modality for cross-modal binding with RGB camera "
                "and audio data (Invariant 4)."
            ),
            rate_limit_hz=fps,
            risk_class=RiskClass.LOW,
        )
        self._driver: DepthDriver = driver if driver is not None else NullDepthDriver()
        self._max_depth_m = max_depth_m

        self.add_capability(
            PluginCapability(
                name="depth_capture",
                description="Captures a normalised depth map at 64×64 resolution.",
                parameters={
                    "resolution": f"{_OUT_W}x{_OUT_H}",
                    "format": "float32_normalised",
                    "max_depth_m": str(max_depth_m),
                },
                risk_class=RiskClass.LOW,
            )
        )

    async def start(self, bus=None) -> None:
        """Connect to the depth camera, then start the emit loop."""
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._driver.connect)
        logger.info("DepthCameraSensor '%s' started (max_depth=%.1f m)", self.sensor_id, self._max_depth_m)
        try:
            await super().start(bus)
        except Exception:
            await loop.run_in_executor(None, self._driver.disconnect)
            raise

    async def stop(self) -> None:
        """Stop the emit loop and disconnect from the depth camera."""
        import asyncio

        await super().stop()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._driver.disconnect)
        logger.info("DepthCameraSensor '%s' stopped", self.sensor_id)

    async def capture(self) -> list:
        """Capture one depth frame and return as a flat normalised float list.

        Returns:
            List of 4096 floats (64×64) in [0, 1].  Values of 0.0 indicate
            missing depth; 1.0 indicates ``max_depth_m`` or beyond.

        The neuromorphic encoder handles float lists natively via
        ``_extract_features()`` — same format as ``CameraSensor``.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        raw: np.ndarray = await loop.run_in_executor(None, self._driver.capture_frame)

        # Resize to 64×64 if the driver returns a different resolution.
        # Prefer OpenCV (already a gateway dependency) for speed; fall back to
        # numpy slicing when cv2 is absent (e.g. in lightweight test environments).
        if raw.shape != (_OUT_H, _OUT_W):
            try:
                import cv2  # noqa: PLC0415
                raw = cv2.resize(raw, (_OUT_W, _OUT_H), interpolation=cv2.INTER_NEAREST)
            except ImportError:
                # Pure-numpy nearest-neighbour resize — correct but slower
                y_idx = (np.arange(_OUT_H) * raw.shape[0] / _OUT_H).astype(int)
                x_idx = (np.arange(_OUT_W) * raw.shape[1] / _OUT_W).astype(int)
                raw = raw[np.ix_(y_idx, x_idx)]

        # Clip to [0, max_depth_m] then normalise to [0, 1]
        clipped = np.clip(raw, 0.0, self._max_depth_m)
        normalised = (clipped / self._max_depth_m).astype(np.float32)

        return normalised.flatten().tolist()
