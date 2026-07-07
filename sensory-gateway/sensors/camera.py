"""
Camera sensor — captures frames from a local camera via OpenCV.

Produces 64x64 grayscale images as flattened float32 arrays,
published to observation.{sensor_id} with provenance "sensor.camera".
The neuromorphic encoding pipeline routes this to the visual modality.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin

logger = logging.getLogger(__name__)


class CameraSensor(SensorPlugin[list]):
    """Mac camera -> 64x64 grayscale -> NATS observation."""

    def __init__(self, device_index: int = 0, fps: float = 5.0, cnn: bool = False):
        super().__init__(
            sensor_id=f"camera.{device_index}",
            name=f"Camera {device_index}",
            description="Local camera capture (64x64 grayscale)",
            rate_limit_hz=fps,
            risk_class=RiskClass.LOW,
        )
        self._device_index = device_index
        self._cap: cv2.VideoCapture | None = None
        self._prev_gray: np.ndarray | None = None
        self._cnn = None
        if cnn:
            try:
                from sensors.cnn_preprocessor import get_shared_preprocessor

                self._cnn = get_shared_preprocessor()
                logger.info(f"CNN preprocessor enabled ({self._cnn.feature_dim} features)")
            except Exception as e:
                logger.warning(f"CNN preprocessing unavailable, using raw pixels: {e}")

        self.add_capability(
            PluginCapability(
                name="video_capture",
                description="Captures grayscale video frames",
                parameters={"resolution": "64x64", "format": "grayscale_float32"},
            )
        )

    async def start(self, bus=None) -> None:
        """Open the camera, then start the emit loop."""
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Cannot open camera {self._device_index}")
        logger.info(
            f"Camera {self._device_index} opened: "
            f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
        try:
            await super().start(bus)
        except Exception:
            self._cap.release()
            self._cap = None
            raise

    async def stop(self) -> None:
        """Release the camera."""
        await super().stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._prev_gray = None

    async def capture(self) -> list:
        """Capture one frame, downsample, return as flat float list.

        Returns a list of 4096 floats (64x64 grayscale, normalized 0-1).
        The neuromorphic encoder handles lists natively via _extract_features().
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera not open")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")

        # Convert to 64x64 grayscale, normalized [0, 1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0

        # Compute frame delta for motion detection (richer signal)
        if self._prev_gray is not None:
            delta = np.abs(small - self._prev_gray)
            # Combine: 70% current frame + 30% motion delta
            combined = 0.7 * small + 0.3 * delta
        else:
            combined = small

        self._prev_gray = small.copy()

        if self._cnn is not None:
            return self._cnn.extract(combined)

        # Return as plain list for JSON serialization over NATS
        return combined.flatten().tolist()
