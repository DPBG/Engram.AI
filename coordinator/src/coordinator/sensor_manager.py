"""
Sensor Manager - Detects and manages available sensors with priority system.

Priority weights: camera(1.0) > sound(0.8) > touch(0.6) > smell(0.4)
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


def _looks_like_imu_reading(raw: dict) -> bool:
    """Return True when a parsed JSON dict contains accelerometer axes."""
    if not isinstance(raw, dict):
        return False
    accel_keys = {
        "ax", "ay", "az",
        "accel_x", "accel_y", "accel_z",
        "accx", "accy", "accz",
    }
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key.lower() in accel_keys:
            return True
    return False


class SensorType(Enum):
    """Types of sensors."""

    CAMERA = "camera"
    MICROPHONE = "microphone"
    IMU = "imu"
    GPS = "gps"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    TOUCH = "touch"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    GAS = "gas"


@dataclass
class SensorInfo:
    """Information about an available sensor."""

    sensor_id: str
    sensor_type: SensorType
    priority: float
    capabilities: list[str]
    active: bool = False


class SensorManager:
    """
    Manages available sensors with priority-based access.

    Detects sensors at startup and provides prioritized sensor lists
    for learning tasks.
    """

    # Priority weights for different sensor types
    SENSOR_PRIORITIES = {
        SensorType.CAMERA: 1.0,
        SensorType.MICROPHONE: 0.8,
        SensorType.TOUCH: 0.6,
        SensorType.IMU: 0.5,
        SensorType.GAS: 0.4,  # Smell
        SensorType.TEMPERATURE: 0.3,
        SensorType.PRESSURE: 0.3,
        SensorType.LIDAR: 0.7,
        SensorType.ULTRASONIC: 0.5,
        SensorType.GPS: 0.4,
    }

    def __init__(self):
        self._sensors: dict[str, SensorInfo] = {}
        self._learning_mode = "normal"  # normal, demonstration, exploration

    async def detect_sensors(self) -> None:
        """
        Detect available sensors on the system.

        Looks for:
        - Camera devices (USB, CSI)
        - Microphones
        - IMU sensors
        - Other hardware via GPIO, I2C, etc.
        """
        logger.info("Detecting available sensors...")

        # Try to detect camera
        detected = await self._detect_camera()
        if detected:
            logger.info("✓ Camera detected")

        # Try to detect microphone
        detected = await self._detect_microphone()
        if detected:
            logger.info("✓ Microphone detected")

        # Try to detect IMU
        detected = await self._detect_imu()
        if detected:
            logger.info("✓ IMU detected")

        # TODO: Add detection for other sensor types

        logger.info(f"Total sensors detected: {len(self._sensors)}")

    async def _detect_camera(self) -> bool:
        """Detect camera sensor."""
        try:
            import cv2

            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                self._sensors["camera_0"] = SensorInfo(
                    sensor_id="camera_0",
                    sensor_type=SensorType.CAMERA,
                    priority=self.SENSOR_PRIORITIES[SensorType.CAMERA],
                    capabilities=["rgb", "capture", "stream"],
                )
                camera.release()
                return True
        except Exception as e:
            logger.debug(f"Camera detection failed: {e}")

        return False

    async def _detect_microphone(self) -> bool:
        """Detect microphone sensor."""
        try:
            import pyaudio

            audio = pyaudio.PyAudio()
            if audio.get_device_count() > 0:
                # Find input devices
                for i in range(audio.get_device_count()):
                    info = audio.get_device_info_by_index(i)
                    if info["maxInputChannels"] > 0:
                        self._sensors[f"microphone_{i}"] = SensorInfo(
                            sensor_id=f"microphone_{i}",
                            sensor_type=SensorType.MICROPHONE,
                            priority=self.SENSOR_PRIORITIES[SensorType.MICROPHONE],
                            capabilities=["audio", "vad", "speech"],
                        )
                        audio.terminate()
                        return True
            audio.terminate()
        except Exception as e:
            logger.debug(f"Microphone detection failed: {e}")

        return False

    async def _detect_imu(self) -> bool:
        """Detect USB serial IMU sensors emitting JSON accelerometer lines."""
        try:
            import serial  # noqa: F401 — optional dependency
            from serial.tools import list_ports
        except ImportError:
            logger.debug("pyserial not available; skipping IMU detection")
            return False

        loop = asyncio.get_running_loop()
        detected_any = False
        try:
            for port_info in list_ports.comports():
                port = port_info.device
                if not await loop.run_in_executor(None, self._probe_imu_port_sync, port):
                    continue
                port_name = port.rsplit("/", 1)[-1]
                sensor_id = f"imu_{port_name}"
                self._sensors[sensor_id] = SensorInfo(
                    sensor_id=sensor_id,
                    sensor_type=SensorType.IMU,
                    priority=self.SENSOR_PRIORITIES[SensorType.IMU],
                    capabilities=["accel", "gyro", "serial"],
                )
                detected_any = True
        except Exception as e:
            logger.debug(f"IMU detection failed: {e}")

        return detected_any

    @staticmethod
    def _probe_imu_port_sync(port: str, baud_rate: int = 115200) -> bool:
        """Open a serial port and look for one valid IMU JSON line."""
        try:
            import serial

            ser = serial.Serial(port, baud_rate, timeout=0.5)
            try:
                for _ in range(5):
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if _looks_like_imu_reading(data):
                        return True
            finally:
                ser.close()
        except Exception:
            return False
        return False

    def get_available_sensors(self, sensor_type: SensorType | None = None) -> list[SensorInfo]:
        """
        Get list of available sensors, optionally filtered by type.

        Returns sensors sorted by priority (highest first).
        """
        sensors = list(self._sensors.values())

        if sensor_type:
            sensors = [s for s in sensors if s.sensor_type == sensor_type]

        # Sort by priority (highest first)
        sensors.sort(key=lambda s: s.priority, reverse=True)

        return sensors

    def get_sensor(self, sensor_id: str) -> SensorInfo | None:
        """Get info about a specific sensor."""
        return self._sensors.get(sensor_id)

    def get_sensor_ids(self) -> list[str]:
        """Get list of all sensor IDs."""
        return list(self._sensors.keys())

    def get_primary_sensor(self, sensor_type: SensorType) -> SensorInfo | None:
        """Get the primary (highest priority) sensor of a given type."""
        sensors = self.get_available_sensors(sensor_type)
        return sensors[0] if sensors else None

    def set_learning_mode(self, mode: str) -> None:
        """
        Set learning mode.

        Modes:
        - normal: Standard operation
        - demonstration: Human is demonstrating a task
        - exploration: System is exploring environment
        """
        if mode not in ["normal", "demonstration", "exploration"]:
            raise ValueError(f"Invalid mode: {mode}")

        logger.info(f"Learning mode: {mode}")
        self._learning_mode = mode

    def get_learning_mode(self) -> str:
        """Get current learning mode."""
        return self._learning_mode

    def activate_sensor(self, sensor_id: str) -> bool:
        """Mark sensor as active."""
        if sensor_id in self._sensors:
            self._sensors[sensor_id].active = True
            return True
        return False

    def deactivate_sensor(self, sensor_id: str) -> bool:
        """Mark sensor as inactive."""
        if sensor_id in self._sensors:
            self._sensors[sensor_id].active = False
            return True
        return False

    def get_active_sensors(self) -> list[SensorInfo]:
        """Get list of currently active sensors."""
        return [s for s in self._sensors.values() if s.active]

    def get_sensor_fusion_weights(self) -> dict[str, float]:
        """
        Get weights for sensor fusion based on priorities.

        Returns dict mapping sensor_id to weight (0.0 to 1.0).
        """
        active_sensors = self.get_active_sensors()
        if not active_sensors:
            return {}

        # Normalize priorities to sum to 1.0
        total_priority = sum(s.priority for s in active_sensors)
        return {s.sensor_id: s.priority / total_priority for s in active_sensors}
