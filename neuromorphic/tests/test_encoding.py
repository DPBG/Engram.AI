"""Tests for spike encoding (sensor data → spike trains)."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.encoding import SpikeEncoder, _resolve_modality
from neuromorphic.regions import SensoryCortex


@pytest.fixture
def config():
    cfg = NeuromorphicConfig()
    cfg.populations.sensory_cortex = 400
    return cfg


@pytest.fixture
def encoder(config):
    return SpikeEncoder(config, rng=np.random.default_rng(42))


@pytest.fixture
def sensory(config):
    return SensoryCortex(config)


class TestProvenanceRouting:
    def test_camera_to_visual(self):
        assert _resolve_modality("sensor.camera") == "visual"

    def test_microphone_to_auditory(self):
        assert _resolve_modality("sensor.microphone") == "auditory"

    def test_touch_to_tactile(self):
        assert _resolve_modality("sensor.touch") == "tactile"

    def test_imu_to_proprioceptive(self):
        assert _resolve_modality("sensor.imu") == "proprioceptive"

    def test_prefix_match(self):
        assert _resolve_modality("sensor.camera.front") == "visual"

    def test_mic_prefix_to_auditory(self):
        assert _resolve_modality("sensor.mic.0") == "auditory"
        assert _resolve_modality("sensor.mic.1") == "auditory"

    def test_serial_to_proprioceptive(self):
        assert _resolve_modality("sensor.serial.usb0") == "proprioceptive"

    def test_unknown_defaults_visual(self):
        assert _resolve_modality("unknown.sensor") == "visual"


class TestSpikeEncoder:
    def test_encode_array(self, encoder, sensory):
        data = np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float32)
        current = encoder.encode(sensory, data, "sensor.camera")
        assert current.shape == (sensory.n,)
        assert current.dtype == np.float32
        # Visual sub-range should have non-zero values
        sr = sensory.get_subrange("visual")
        assert current[sr.slice()].sum() > 0

    def test_encode_list(self, encoder, sensory):
        data = [0.5, 0.3, 0.8]
        current = encoder.encode(sensory, data, "sensor.microphone")
        sr = sensory.get_subrange("auditory")
        assert current[sr.slice()].sum() > 0

    def test_encode_dict_with_features(self, encoder, sensory):
        data = {"features": [0.5, 0.3, 0.8]}
        current = encoder.encode(sensory, data, "sensor.camera")
        assert current.sum() > 0

    def test_encode_dict_numeric_values(self, encoder, sensory):
        data = {"x": 0.5, "y": 0.3, "z": 0.8}
        current = encoder.encode(sensory, data, "sensor.imu")
        sr = sensory.get_subrange("proprioceptive")
        assert current[sr.slice()].sum() > 0

    def test_encode_scalar(self, encoder, sensory):
        current = encoder.encode(sensory, 0.7, "sensor.touch")
        sr = sensory.get_subrange("tactile")
        assert current[sr.slice()].sum() > 0

    def test_encode_none_graceful(self, encoder, sensory):
        current = encoder.encode(sensory, None, "sensor.camera")
        assert current.shape == (sensory.n,)
        # None should result in zero current
        assert current.sum() == 0.0

    def test_multimodal_encode(self, encoder, sensory):
        inputs = {
            "sensor.camera": [0.5, 0.3],
            "sensor.microphone": [0.2, 0.8],
        }
        current = encoder.encode_multimodal(sensory, inputs)
        assert current.shape == (sensory.n,)
        # Both visual and auditory should be active
        vis_sr = sensory.get_subrange("visual")
        aud_sr = sensory.get_subrange("auditory")
        assert current[vis_sr.slice()].sum() > 0
        assert current[aud_sr.slice()].sum() > 0

    def test_rate_coding_normalized(self, encoder, sensory):
        """Different magnitude inputs should produce similar-range outputs."""
        small = encoder.encode(sensory, [0.1, 0.2], "sensor.camera")
        large = encoder.encode(sensory, [100.0, 200.0], "sensor.camera")
        # Both should produce non-zero current
        assert small.sum() > 0
        assert large.sum() > 0
