"""
Mock actuator implementations for testing.

These mocks simulate hardware actuators without requiring physical devices.
"""

import asyncio


class MockServo:
    """
    Mock servo motor actuator.

    Simulates controlling a servo motor position (0-180 degrees).
    """

    def __init__(self, channel: int, min_pulse: int = 1000, max_pulse: int = 2000):
        self.channel = channel
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        self._position = 90  # Center position
        self._enabled = False

    async def enable(self) -> None:
        """Enable the servo."""
        self._enabled = True

    async def disable(self) -> None:
        """Disable the servo."""
        self._enabled = False

    async def set_position(self, angle: float) -> None:
        """
        Set servo position in degrees (0-180).

        Args:
            angle: Target angle in degrees
        """
        if not self._enabled:
            raise RuntimeError("Servo not enabled")

        if not 0 <= angle <= 180:
            raise ValueError(f"Angle must be 0-180, got {angle}")

        # Simulate servo movement time
        movement_time = abs(angle - self._position) / 180.0 * 0.5  # Max 0.5s for full range
        await asyncio.sleep(movement_time)

        self._position = angle

    def get_position(self) -> float:
        """Get current servo position."""
        return self._position

    def is_enabled(self) -> bool:
        """Check if servo is enabled."""
        return self._enabled


class MockLED:
    """
    Mock LED actuator.

    Simulates controlling an LED (on/off, brightness, RGB color).
    """

    def __init__(self, pin: int | None = None):
        self.pin = pin
        self._on = False
        self._brightness = 1.0
        self._color = (255, 255, 255)  # White

    async def turn_on(self) -> None:
        """Turn the LED on."""
        self._on = True

    async def turn_off(self) -> None:
        """Turn the LED off."""
        self._on = False

    async def set_brightness(self, brightness: float) -> None:
        """
        Set LED brightness (0.0 to 1.0).

        Args:
            brightness: Brightness level (0.0 = off, 1.0 = full)
        """
        if not 0.0 <= brightness <= 1.0:
            raise ValueError(f"Brightness must be 0.0-1.0, got {brightness}")
        self._brightness = brightness

    async def set_color(self, r: int, g: int, b: int) -> None:
        """
        Set RGB color (0-255 per channel).

        Args:
            r: Red channel (0-255)
            g: Green channel (0-255)
            b: Blue channel (0-255)
        """
        if not all(0 <= c <= 255 for c in (r, g, b)):
            raise ValueError("RGB values must be 0-255")
        self._color = (r, g, b)

    def is_on(self) -> bool:
        """Check if LED is on."""
        return self._on

    def get_brightness(self) -> float:
        """Get current brightness."""
        return self._brightness

    def get_color(self) -> tuple[int, int, int]:
        """Get current RGB color."""
        return self._color


class MockMotor:
    """
    Mock DC motor actuator.

    Simulates controlling a DC motor (speed and direction).
    """

    def __init__(self, channel: int):
        self.channel = channel
        self._speed = 0.0  # -1.0 to 1.0 (negative = reverse)
        self._enabled = False

    async def enable(self) -> None:
        """Enable the motor."""
        self._enabled = True

    async def disable(self) -> None:
        """Disable the motor and stop."""
        self._enabled = False
        self._speed = 0.0

    async def set_speed(self, speed: float) -> None:
        """
        Set motor speed (-1.0 to 1.0).

        Args:
            speed: Motor speed (-1.0 = full reverse, 0.0 = stop, 1.0 = full forward)
        """
        if not self._enabled:
            raise RuntimeError("Motor not enabled")

        if not -1.0 <= speed <= 1.0:
            raise ValueError(f"Speed must be -1.0 to 1.0, got {speed}")

        self._speed = speed

    async def stop(self) -> None:
        """Stop the motor."""
        self._speed = 0.0

    def get_speed(self) -> float:
        """Get current motor speed."""
        return self._speed

    def is_enabled(self) -> bool:
        """Check if motor is enabled."""
        return self._enabled
