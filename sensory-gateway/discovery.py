"""
Hardware discovery for local and network-attached sensors.

Probes the host machine for available cameras, microphones, serial devices,
and listens for network sensors announcing themselves over NATS.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredDevice:
    """A sensor device found during discovery."""

    device_type: str  # "camera", "mic", "serial", "network"
    device_id: str  # unique, e.g. "camera:0", "mic:2"
    name: str  # human-readable
    metadata: dict[str, Any] = field(default_factory=dict)


def discover_cameras(max_index: int = 10) -> list[DiscoveredDevice]:
    """Probe OpenCV camera indices and return available cameras."""
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed — skipping camera discovery")
        return []

    cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(DiscoveredDevice(
                    device_type="camera",
                    device_id=f"camera:{i}",
                    name=f"Camera {i} ({cap.getBackendName()})",
                    metadata={
                        "index": i,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "backend": cap.getBackendName(),
                    },
                ))
            cap.release()
    return cameras


def discover_microphones() -> list[DiscoveredDevice]:
    """List available audio input devices via sounddevice/PortAudio."""
    try:
        import sounddevice as sd
    except ImportError:
        logger.warning("sounddevice not installed — skipping microphone discovery")
        return []

    mics = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mics.append(DiscoveredDevice(
                device_type="mic",
                device_id=f"mic:{i}",
                name=dev["name"],
                metadata={
                    "index": i,
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                },
            ))
    return mics


_SERIAL_SKIP_PATTERNS = (
    "debug-console",
    "bluetooth",
    "wlan-debug",
    "/dev/cu.BLTH",
    "/dev/cu.debug",
    # Linux motherboard serial ports — no devices connected, just log spam
    "/dev/ttyS",
)


def discover_serial_devices() -> list[DiscoveredDevice]:
    """List USB serial devices (Arduino, Pi Pico, etc.).

    Skips system serial ports (debug-console, Bluetooth) that are not
    real sensor hardware.
    """
    try:
        import serial.tools.list_ports
    except ImportError:
        logger.warning("pyserial not installed — skipping serial discovery")
        return []

    devices = []
    for port in serial.tools.list_ports.comports():
        # Skip macOS system serial ports (not real sensors)
        port_lower = (port.device + " " + (port.description or "")).lower()
        if any(pat.lower() in port_lower for pat in _SERIAL_SKIP_PATTERNS):
            logger.debug(f"Skipping system serial port: {port.device}")
            continue

        devices.append(DiscoveredDevice(
            device_type="serial",
            device_id=f"serial:{port.device}",
            name=port.description or port.device,
            metadata={
                "port": port.device,
                "vid": port.vid,
                "pid": port.pid,
                "serial_number": port.serial_number,
                "manufacturer": port.manufacturer,
            },
        ))
    return devices


_IMU_DEVICES_ENV = "ENGRAM_IMU_DEVICES"


def discover_imus() -> list[DiscoveredDevice]:
    """List IMU devices declared via the ``ENGRAM_IMU_DEVICES`` env var.

    IMUs attach over many buses (I2C/SPI/serial) with no universal probe, so
    operators declare them explicitly as a comma-separated list of ``id@port``
    (or bare ``port``) entries, e.g.
    ``ENGRAM_IMU_DEVICES="waist@/dev/ttyUSB0,head@/dev/ttyUSB1"``. Returns an
    empty list when unset — a graceful no-op when no IMU is present.
    """
    raw = os.environ.get(_IMU_DEVICES_ENV, "").strip()
    if not raw:
        return []

    devices: list[DiscoveredDevice] = []
    for i, entry in enumerate(raw.split(",")):
        entry = entry.strip()
        if not entry:
            continue
        if "@" in entry:
            device_id, _, port = entry.partition("@")
        else:
            device_id, port = str(i), entry
        device_id = device_id.strip() or str(i)
        port = port.strip()
        devices.append(
            DiscoveredDevice(
                device_type="imu",
                device_id=f"imu:{device_id}",
                name=f"IMU {device_id}" + (f" ({port})" if port else ""),
                metadata={"id": device_id, "port": port},
            )
        )
    return devices


def discover_all(
    skip_cameras: bool = False,
    skip_microphones: bool = False,
    skip_serial: bool = False,
    skip_imu: bool = False,
) -> list[DiscoveredDevice]:
    """Run local hardware discovery probes.

    Probes can be skipped individually — useful on hosts where camera probing is
    slow/noisy (e.g. Windows MSMF) and you only want file-based input.
    """
    devices: list[DiscoveredDevice] = []

    if skip_cameras:
        logger.info("Skipping camera discovery (--no-camera)")
    else:
        logger.info("Discovering cameras...")
        cameras = discover_cameras()
        logger.info(f"  Found {len(cameras)} camera(s)")
        devices.extend(cameras)

    if skip_microphones:
        logger.info("Skipping microphone discovery (--no-mic)")
    else:
        logger.info("Discovering microphones...")
        mics = discover_microphones()
        logger.info(f"  Found {len(mics)} microphone(s)")
        devices.extend(mics)

    if skip_serial:
        logger.info("Skipping serial discovery")
    else:
        logger.info("Discovering serial devices...")
        serial_devs = discover_serial_devices()
        logger.info(f"  Found {len(serial_devs)} serial device(s)")
        devices.extend(serial_devs)

    if skip_imu:
        logger.info("Skipping IMU discovery")
    else:
        logger.info("Discovering IMUs...")
        imus = discover_imus()
        logger.info(f"  Found {len(imus)} IMU(s)")
        devices.extend(imus)

    return devices


# Device types the gateway knows how to handle natively.
KNOWN_DEVICE_TYPES = {"camera", "mic", "serial", "imu"}

# Keywords that indicate a device is an actuator rather than a sensor.
_ACTUATOR_KEYWORDS = ("motor", "servo", "actuator", "gripper")


def infer_plugin_type(device_name: str) -> str:
    """Infer whether a device is a sensor or actuator from its name.

    Returns ``"actuator"`` if any actuator keyword is found, else ``"sensor"``.
    """
    name_lower = device_name.lower()
    if any(kw in name_lower for kw in _ACTUATOR_KEYWORDS):
        return "actuator"
    return "sensor"


def sanitize_device_string(value: object, max_len: int = 200) -> str:
    """Sanitize a hardware-sourced string for use in NATS payloads / prompts.

    Strips newlines, carriage returns, and truncates to *max_len*.
    """
    return str(value).replace("\n", " ").replace("\r", " ")[:max_len]
