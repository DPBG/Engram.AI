"""
Video file sensor — reads frames from a video file (MP4/WebM/AVI).

Produces 64x64 grayscale images as flattened float32 arrays,
identical to CameraSensor output. Published to observation.{sensor_id}
with provenance "sensor.videofile". The neuromorphic encoding pipeline
routes this to the visual modality.

The brain doesn't know whether frames come from a webcam or a file.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin

logger = logging.getLogger(__name__)

# Sentinel: capture() returns this when frame was read but sparse-skipped.
# Distinct from None (actual read failure / video ended).
_SPARSE_SKIP = object()


class VideoFileSensor(SensorPlugin[list]):
    """Video file -> 64x64 grayscale frames -> NATS observation."""

    def __init__(
        self,
        filepath: str,
        fps: float = 2.0,
        loop: bool = True,
        session_id: str | None = None,
        sparse_threshold: float = 0.05,
        cnn: bool = False,
        turbo: bool = False,
    ):
        stem = Path(filepath).stem[:20]
        sensor_id = f"videofile.{stem}"
        super().__init__(
            sensor_id=sensor_id,
            name=f"Video: {Path(filepath).name}",
            description="Video file capture (64x64 grayscale)",
            rate_limit_hz=fps,
            risk_class=RiskClass.LOW,
        )
        self._filepath = filepath
        self._loop = loop
        self._session_id = session_id
        self._sparse_threshold = sparse_threshold
        self._turbo = turbo
        self._cap: cv2.VideoCapture | None = None
        self._prev_gray: np.ndarray | None = None
        self._total_frames: int = 0
        self._video_fps: float = 0.0
        self._duration_s: float = 0.0
        self._loop_count: int = 0
        self._frames_emitted: int = 0
        self._frames_skipped: int = 0
        self._start_time: float = 0.0
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
                name="video_file_capture",
                description="Reads frames from a video file",
                parameters={"resolution": "64x64", "format": "grayscale_float32"},
            )
        )

    @property
    def loop_count(self) -> int:
        return self._loop_count

    @property
    def frames_emitted(self) -> int:
        return self._frames_emitted

    @property
    def duration_s(self) -> float:
        return self._duration_s

    @property
    def elapsed_s(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    @property
    def current_pos_s(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    @property
    def progress(self) -> float:
        """Current position as fraction 0-1 within current loop."""
        if self._duration_s <= 0:
            return 0.0
        return min(1.0, self.current_pos_s / self._duration_s)

    async def start(self, bus=None) -> None:
        """Open the video file, then start the emit loop."""
        if self._running:
            logger.warning(f"VideoFileSensor {self.sensor_id} already running, skipping start")
            return

        # CAP_PROP_N_THREADS=1: limit ffmpeg decoder threads per capture.
        # Without this, each capture spawns ~79 threads on a 64-core machine.
        self._cap = cv2.VideoCapture(self._filepath, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Cannot open video file: {self._filepath}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._duration_s = self._total_frames / self._video_fps if self._video_fps > 0 else 0.0
        self._loop_count = 0
        self._frames_emitted = 0
        self._start_time = time.time()
        self._prev_gray = None

        # Guard: empty video (0 frames)
        if self._total_frames == 0:
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Video file has 0 frames: {self._filepath}")

        logger.info(
            f"Video opened: {self._filepath} "
            f"({self._total_frames} frames, {self._duration_s:.1f}s, "
            f"{self._video_fps:.1f} native fps, playback at {self.rate_limit_hz} fps)"
        )
        try:
            await super().start(bus)
        except Exception:
            self._cap.release()
            self._cap = None
            raise

    async def stop(self) -> None:
        """Release the video capture."""
        await super().stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._prev_gray = None

    async def capture(self) -> list | object | None:
        """Read next frame, downsample to 64x64 grayscale + motion delta.

        Returns a list of 4096/576 floats, _SPARSE_SKIP if frame was
        read but below change threshold, or None on actual read failure.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            # End of video
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._loop_count += 1
                self._prev_gray = None
                logger.info(f"Video loop #{self._loop_count}: {Path(self._filepath).name}")
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                logger.info(f"Video ended: {Path(self._filepath).name}")
                return None

        # Same processing as CameraSensor: 64x64 grayscale, normalized [0, 1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0

        # Sparse encoding: skip frames with minimal change (saves STDP cycles)
        if self._prev_gray is not None:
            delta = np.abs(small - self._prev_gray)
            mean_diff = float(delta.mean())
            if mean_diff < self._sparse_threshold:
                self._frames_skipped += 1
                # DON'T update _prev_gray — keep the last EMITTED frame
                # as reference so motion delta stays accurate when we
                # do emit. Updating here loses accumulated motion.
                return _SPARSE_SKIP
            # Motion delta: 70% current frame + 30% motion
            combined = 0.7 * small + 0.3 * delta
        else:
            combined = small

        self._prev_gray = small.copy()
        self._frames_emitted += 1

        if self._cnn is not None:
            return self._cnn.extract(combined)

        return combined.flatten().tolist()

    async def _emit_loop(self) -> None:
        """Override emit loop to handle None (video ended) by stopping."""
        interval = 1.0 / self.rate_limit_hz if self.rate_limit_hz > 0 else 0
        consecutive_read_failures = 0
        max_read_failures = 10  # break after N actual read failures
        while self._running:
            try:
                data = await self.capture()
                if data is _SPARSE_SKIP:
                    # Frame was read successfully but sparse-skipped — not a failure
                    consecutive_read_failures = 0
                elif data is not None:
                    await self.emit(data)
                    consecutive_read_failures = 0
                elif not self._loop:
                    logger.info(f"Video sensor {self.sensor_id} finished (no loop)")
                    break
                else:
                    # Loop mode but actual read failure
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= max_read_failures:
                        logger.error(
                            f"Video sensor {self.sensor_id}: {consecutive_read_failures} consecutive "
                            f"read failures, stopping"
                        )
                        break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in video sensor {self.sensor_id}: {e}")
            if self._turbo:
                await asyncio.sleep(0)  # yield only, no delay
            elif interval > 0:
                await asyncio.sleep(interval)

    @property
    def frames_skipped(self) -> int:
        return self._frames_skipped

    def get_training_status(self) -> dict:
        """Return current training status for this video sensor."""
        total = self._frames_emitted + self._frames_skipped
        skip_pct = round(self._frames_skipped / total * 100, 1) if total > 0 else 0.0
        return {
            "sensor_id": self.sensor_id,
            "session_id": self._session_id,
            "filepath": self._filepath,
            "filename": Path(self._filepath).name,
            "duration_s": round(self._duration_s, 1),
            "current_pos_s": round(self.current_pos_s, 1),
            "progress": round(self.progress, 3),
            "loop_count": self._loop_count,
            "frames_emitted": self._frames_emitted,
            "frames_skipped": self._frames_skipped,
            "skip_pct": skip_pct,
            "elapsed_s": round(self.elapsed_s, 1),
            "fps": self.rate_limit_hz,
            "loop": self._loop,
            "running": self._running,
        }
