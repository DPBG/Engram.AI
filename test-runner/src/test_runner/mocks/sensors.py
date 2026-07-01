"""
Mock sensor implementations for testing.

These mocks simulate hardware sensors without requiring physical devices.
"""

import asyncio
import random
import time
from typing import Any


class MockCamera:
    """
    Mock camera sensor that generates fake image data.

    Simulates a camera capturing frames at a specified FPS.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self._frame_count = 0

    async def start(self) -> None:
        """Start the mock camera."""
        self._running = True
        self._frame_count = 0

    async def stop(self) -> None:
        """Stop the mock camera."""
        self._running = False

    async def capture(self) -> dict[str, Any]:
        """
        Capture a frame from the mock camera.

        Returns fake image data with metadata.
        """
        if not self._running:
            raise RuntimeError("Camera not started")

        # Simulate capture delay
        await asyncio.sleep(1.0 / self.fps)

        self._frame_count += 1

        return {
            "frame_id": self._frame_count,
            "timestamp": int(time.time() * 1000),
            "width": self.width,
            "height": self.height,
            "format": "RGB",
            "data": f"<mock image data {self._frame_count}>",
            "metadata": {
                "exposure": random.uniform(0.01, 0.1),
                "gain": random.uniform(1.0, 4.0),
            },
        }

    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running


class MockGPIO:
    """
    Mock GPIO sensor for reading digital/analog pins.

    Simulates reading from GPIO pins without physical hardware.
    """

    def __init__(self, pin: int):
        self.pin = pin
        self._value = 0
        self._mode = "INPUT"

    def set_mode(self, mode: str) -> None:
        """Set pin mode (INPUT or OUTPUT)."""
        if mode not in ("INPUT", "OUTPUT"):
            raise ValueError(f"Invalid mode: {mode}")
        self._mode = mode

    def read(self) -> int:
        """Read digital value (0 or 1)."""
        if self._mode != "INPUT":
            raise RuntimeError("Pin not in INPUT mode")
        # Simulate random digital input
        return random.choice([0, 1])

    def write(self, value: int) -> None:
        """Write digital value (0 or 1)."""
        if self._mode != "OUTPUT":
            raise RuntimeError("Pin not in OUTPUT mode")
        if value not in (0, 1):
            raise ValueError(f"Invalid digital value: {value}")
        self._value = value

    def read_analog(self) -> float:
        """Read analog value (0.0 to 1.0)."""
        if self._mode != "INPUT":
            raise RuntimeError("Pin not in INPUT mode")
        # Simulate random analog input
        return random.uniform(0.0, 1.0)


class MockIMU:
    """
    Mock Inertial Measurement Unit (IMU) sensor.

    Simulates accelerometer, gyroscope, and magnetometer readings.
    """

    def __init__(self):
        self._running = False
        self._calibrated = False

    async def start(self) -> None:
        """Start the IMU."""
        self._running = True

    async def stop(self) -> None:
        """Stop the IMU."""
        self._running = False

    async def calibrate(self) -> None:
        """Calibrate the IMU."""
        await asyncio.sleep(0.5)  # Simulate calibration time
        self._calibrated = True

    async def read(self) -> dict[str, Any]:
        """
        Read IMU data.

        Returns accelerometer, gyroscope, and magnetometer readings.
        """
        if not self._running:
            raise RuntimeError("IMU not started")

        # Simulate sensor noise
        return {
            "timestamp": int(time.time() * 1000),
            "accelerometer": {
                "x": random.gauss(0.0, 0.1),
                "y": random.gauss(0.0, 0.1),
                "z": random.gauss(9.81, 0.1),  # Gravity
            },
            "gyroscope": {
                "x": random.gauss(0.0, 0.01),
                "y": random.gauss(0.0, 0.01),
                "z": random.gauss(0.0, 0.01),
            },
            "magnetometer": {
                "x": random.gauss(30.0, 1.0),
                "y": random.gauss(0.0, 1.0),
                "z": random.gauss(-40.0, 1.0),
            },
            "temperature": random.gauss(25.0, 2.0),
            "calibrated": self._calibrated,
        }
