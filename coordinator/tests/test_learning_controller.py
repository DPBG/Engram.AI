"""Tests for LearningController observation feature extraction.

Loads learning_controller.py directly (mirroring test_task_coordinator.py) so
the test does not import the whole coordinator package.
"""

import asyncio
import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

_LC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "coordinator", "learning_controller.py"
)
_spec = importlib.util.spec_from_file_location("coord_learning_controller", _LC_PATH)
lc = importlib.util.module_from_spec(_spec)
sys.modules["coord_learning_controller"] = lc
_spec.loader.exec_module(lc)

LearningController = lc.LearningController
LearningPhase = lc.LearningPhase
extract_visual_features = lc.extract_visual_features
extract_audio_features = lc.extract_audio_features


def test_extract_visual_features_from_positions_colors_objects():
    obs = [
        {
            "sensor_id": "camera_0",
            "timestamp": 1000,
            "data": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.5},
                "color": "Red",
                "label": "ball",
            },
        },
        {
            "sensor_id": "camera_0",
            "timestamp": 1100,
            "data": {
                "bbox": [0, 0, 10, 20],
                "objects": [{"name": "cup", "color": "blue"}],
            },
        },
        {
            "sensor_id": "camera_0",
            "timestamp": 1200,
            "data": "not-a-dict",
        },
    ]

    features = extract_visual_features(obs)

    assert features["observation_count"] == 3
    assert features["positions"] == [
        {"x": 1.0, "y": 2.0, "z": 0.5},
        {"x": 5.0, "y": 10.0},
    ]
    assert features["colors"] == ["red", "blue"]
    assert features["objects"] == ["ball", "cup"]
    assert features["trajectory"][0] == {"timestamp": 1000, "x": 1.0, "y": 2.0, "z": 0.5}
    assert features["trajectory"][1] == {"timestamp": 1100, "x": 5.0, "y": 10.0}


def test_extract_audio_features_from_commands_and_energy():
    obs = [
        {
            "sensor_id": "microphone_0",
            "timestamp": 1,
            "data": {"transcript": "pick up the ball", "energy": 0.4},
        },
        {
            "sensor_id": "microphone_0",
            "timestamp": 2,
            "data": {"command": "Pick Up The Ball", "rms": 0.8},
        },
        {
            "sensor_id": "microphone_0",
            "timestamp": 3,
            "data": b"raw-bytes",
        },
    ]

    features = extract_audio_features(obs)

    assert features["observation_count"] == 3
    assert features["commands"] == ["pick up the ball"]
    assert features["energy"] == {
        "min": 0.4,
        "max": 0.8,
        "mean": 0.6,
        "sample_count": 2.0,
    }


def test_extract_audio_features_without_energy():
    features = extract_audio_features(
        [{"sensor_id": "microphone_0", "timestamp": 1, "data": {"text": "stop"}}]
    )
    assert features["commands"] == ["stop"]
    assert features["energy"] is None


def _make_controller(tmp_path):
    sensor_manager = MagicMock()
    sensor_manager.SensorType = SimpleNamespace(CAMERA="camera", MICROPHONE="microphone")
    sensor_manager.get_primary_sensor.return_value = None
    sensor_manager.get_sensor_fusion_weights.return_value = {}
    sensor_manager.get_active_sensors.return_value = []

    nats = MagicMock()
    nats.publish = AsyncMock()

    return (
        LearningController(
            nats_client=nats,
            sensor_manager=sensor_manager,
            tasks_root=str(tmp_path),
        ),
        nats,
    )


def test_process_observations_populates_structured_parameters(tmp_path):
    controller, _ = _make_controller(tmp_path)
    controller._current_task = {
        "trace_id": "t1",
        "task_name": "pick_ball",
        "description": "Pick up the red ball",
    }
    controller._current_phase = LearningPhase.IMITATE
    controller._observation_buffer = [
        {
            "sensor_id": "camera_0",
            "timestamp": 100,
            "data": {"x": 3.0, "y": 4.0, "color": "red", "label": "ball"},
        },
        {
            "sensor_id": "microphone_0",
            "timestamp": 110,
            "data": {"command": "grab it", "energy": 0.5},
        },
    ]

    task_data = asyncio.run(controller._process_observations())

    assert task_data["parameters"]["visual_features"] == 1
    assert task_data["parameters"]["positions"] == [{"x": 3.0, "y": 4.0}]
    assert task_data["parameters"]["colors"] == ["red"]
    assert task_data["parameters"]["objects"] == ["ball"]
    assert task_data["parameters"]["audio_features"] == 1
    assert task_data["parameters"]["voice_commands"] == ["grab it"]
    assert task_data["parameters"]["audio_energy"]["mean"] == 0.5
    assert len(task_data["trajectory"]) == 2
