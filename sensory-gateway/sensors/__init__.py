"""Sensor adapters — each extends SensorPlugin from the SDK.

Imports are individually guarded so that a missing optional dependency (e.g.
`sounddevice` for the microphone/speech sensors) doesn't break importing the
sensors that *are* available (e.g. video-file input needs only opencv). A sensor
whose deps are missing is bound to None; using it raises a clear error at start.
"""

import logging as _logging

_log = _logging.getLogger("sensory-gateway.sensors")


def _optional(import_fn, name):
    try:
        return import_fn()
    except ImportError as exc:  # optional dependency not installed
        _log.debug("sensor %s unavailable: %s", name, exc)
        return None


def _camera():
    from sensors.camera import CameraSensor
    return CameraSensor


def _microphone():
    from sensors.microphone import MicrophoneSensor
    return MicrophoneSensor


def _serial():
    from sensors.serial_device import SerialSensor
    return SerialSensor


def _video_file():
    from sensors.video_file import VideoFileSensor
    return VideoFileSensor


def _audio_file():
    from sensors.audio_file import AudioFileSensor
    return AudioFileSensor


CameraSensor = _optional(_camera, "camera")
MicrophoneSensor = _optional(_microphone, "microphone")
SerialSensor = _optional(_serial, "serial")
VideoFileSensor = _optional(_video_file, "video_file")
AudioFileSensor = _optional(_audio_file, "audio_file")

__all__ = [
    "CameraSensor",
    "MicrophoneSensor",
    "SerialSensor",
    "VideoFileSensor",
    "AudioFileSensor",
]
