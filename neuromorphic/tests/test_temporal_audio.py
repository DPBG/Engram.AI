"""Tests for temporal audio frame encoding (Solution 5 integration)."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.encoding import SpikeEncoder
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


def _make_temporal_frame(n_chunks: int = 5) -> dict:
    """Create a synthetic temporal audio frame (as aggregator would produce)."""
    rng = np.random.default_rng(42)
    raw = rng.random((n_chunks, 13)).astype(np.float32)
    return {
        "type": "temporal_audio_frame",
        "n_chunks": n_chunks,
        "mean_mfcc": raw.mean(axis=0).tolist(),
        "delta_mfcc": (raw[-1] - raw[0]).tolist(),
        "energy": np.sqrt((raw**2).mean(axis=1)).tolist(),
        "raw_stack": raw.tolist(),
    }


class TestTemporalAudioEncoding:
    def test_temporal_frame_encodes(self, encoder, sensory):
        """Temporal audio frame should produce non-zero current in auditory range."""
        frame = _make_temporal_frame(5)
        current = encoder.encode(sensory, frame, "sensor.audiofile")
        assert current.shape == (sensory.n,)
        sr = sensory.get_subrange("auditory")
        assert current[sr.slice()].sum() > 0

    def test_temporal_frame_richer_than_single_mfcc(self, encoder, sensory):
        """Temporal frame should produce more feature variance than a plain 13-float MFCC."""
        frame = _make_temporal_frame(5)
        # Extract features directly
        features_temporal = encoder._extract_features(frame, 400)
        features_plain = encoder._extract_features([0.5] * 13, 400)
        # Temporal frame should have more features
        assert len(features_temporal) > len(features_plain)

    def test_temporal_frame_has_mean_delta_energy(self, encoder, sensory):
        """Temporal frame features should include mean + delta + energy."""
        frame = _make_temporal_frame(5)
        features = encoder._extract_features(frame, 400)
        # mean(13) + delta(13) + energy(5) = 31 features
        assert len(features) == 31

    def test_single_chunk_temporal_frame(self, encoder, sensory):
        """Single-chunk temporal frame should still work (edge case)."""
        frame = _make_temporal_frame(1)
        current = encoder.encode(sensory, frame, "sensor.audiofile")
        sr = sensory.get_subrange("auditory")
        assert current[sr.slice()].sum() > 0

    def test_backward_compatible_plain_mfcc(self, encoder, sensory):
        """Plain 13-float MFCC list (legacy format) should still encode correctly."""
        data = [0.5, 0.3, 0.8, 0.1, 0.6, 0.2, 0.7, 0.4, 0.9, 0.3, 0.5, 0.1, 0.8]
        current = encoder.encode(sensory, data, "sensor.audiofile")
        sr = sensory.get_subrange("auditory")
        assert current[sr.slice()].sum() > 0

    def test_temporal_frame_in_multimodal(self, encoder, sensory):
        """Temporal audio frame should work alongside visual input."""
        frame = _make_temporal_frame(5)
        inputs = {
            "sensor.camera": [0.5, 0.3, 0.8, 0.1],
            "sensor.audiofile": frame,
        }
        current = encoder.encode_multimodal(sensory, inputs)
        vis_sr = sensory.get_subrange("visual")
        aud_sr = sensory.get_subrange("auditory")
        assert current[vis_sr.slice()].sum() > 0
        assert current[aud_sr.slice()].sum() > 0

    def test_different_frames_produce_different_currents(self, encoder, sensory):
        """Different temporal frames should produce distinguishable currents."""
        rng = np.random.default_rng(1)
        raw1 = rng.random((5, 13)).astype(np.float32)
        raw2 = rng.random((5, 13)).astype(np.float32) + 5.0
        frame1 = {
            "type": "temporal_audio_frame",
            "n_chunks": 5,
            "mean_mfcc": raw1.mean(axis=0).tolist(),
            "delta_mfcc": (raw1[-1] - raw1[0]).tolist(),
            "energy": np.sqrt((raw1**2).mean(axis=1)).tolist(),
            "raw_stack": raw1.tolist(),
        }
        frame2 = {
            "type": "temporal_audio_frame",
            "n_chunks": 5,
            "mean_mfcc": raw2.mean(axis=0).tolist(),
            "delta_mfcc": (raw2[-1] - raw2[0]).tolist(),
            "energy": np.sqrt((raw2**2).mean(axis=1)).tolist(),
            "raw_stack": raw2.tolist(),
        }
        c1 = encoder.encode(sensory, frame1, "sensor.audiofile")
        c2 = encoder.encode(sensory, frame2, "sensor.audiofile")
        # Should not be identical
        assert not np.allclose(c1, c2)
