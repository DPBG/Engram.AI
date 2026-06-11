"""Tests for auditory short-term memory (echoic memory buffer)."""

import numpy as np
import pytest

from neuromorphic.auditory_stm import AuditorySTM, AuditorySTMConfig


@pytest.fixture
def stm():
    return AuditorySTM(AuditorySTMConfig(window_steps=4, decay_rate=0.85))


def _make_frame(n_chunks: int = 5, base: float = 0.0) -> dict:
    """Create a synthetic temporal audio frame."""
    rng = np.random.default_rng(int(base * 100) + n_chunks)
    raw_stack = (rng.random((n_chunks, 13)) + base).astype(np.float32)
    mean_mfcc = raw_stack.mean(axis=0)
    delta_mfcc = np.zeros(13, dtype=np.float32)
    if n_chunks >= 2:
        delta_mfcc = (raw_stack[-1] - raw_stack[0]) / max(1, n_chunks - 1)
    energy = np.sqrt((raw_stack ** 2).mean(axis=1))
    return {
        "type": "temporal_audio_frame",
        "n_chunks": n_chunks,
        "mean_mfcc": mean_mfcc.tolist(),
        "delta_mfcc": delta_mfcc.tolist(),
        "energy": energy.tolist(),
        "raw_stack": raw_stack.tolist(),
    }


class TestAuditorySTMBasic:
    def test_empty_returns_none(self, stm):
        assert stm.get_features() is None

    def test_update_single_frame(self, stm):
        frame = _make_frame(5)
        stm.update(frame)
        features = stm.get_features()
        assert features is not None
        assert features.dtype == np.float32
        assert len(features) > 13  # more than just mean MFCC

    def test_n_frames_tracking(self, stm):
        assert stm.n_frames == 0
        stm.update(_make_frame(3, base=0.1))
        assert stm.n_frames == 1
        stm.update(_make_frame(3, base=0.2))
        assert stm.n_frames == 2

    def test_window_trimming(self, stm):
        """Buffer trims to window_steps (4 in fixture)."""
        for i in range(10):
            stm.update(_make_frame(3, base=i * 0.1))
        assert stm.n_frames == 4

    def test_ignores_non_temporal_frame(self, stm):
        stm.update({"type": "other", "data": [1, 2, 3]})
        assert stm.n_frames == 0
        assert stm.get_features() is None

    def test_ignores_missing_type(self, stm):
        stm.update({"mean_mfcc": [0] * 13})
        assert stm.n_frames == 0


class TestAuditorySTMFeatures:
    def test_feature_vector_content(self, stm):
        """Feature vector should contain mean + delta + energy_summary + temporal_grad + raw."""
        stm.update(_make_frame(5, base=0.5))
        features = stm.get_features()
        # With defaults: 13 (mean) + 13 (delta) + 3 (energy) + 13 (grad) + 5*13 (raw) = 107
        assert len(features) >= 42  # at minimum mean + delta + energy + gradient

    def test_features_change_with_new_data(self, stm):
        stm.update(_make_frame(5, base=0.1))
        f1 = stm.get_features().copy()
        stm.update(_make_frame(5, base=0.9))
        f2 = stm.get_features()
        # Features should be different with different input
        assert not np.allclose(f1, f2)

    def test_normalization(self, stm):
        stm.update(_make_frame(5, base=0.5))
        features = stm.get_features()
        # With normalize_output=True, features should be in [0, 1]
        assert features.min() >= 0.0
        assert features.max() <= 1.0 + 1e-6

    def test_no_normalization(self):
        cfg = AuditorySTMConfig(normalize_output=False, window_steps=4)
        stm = AuditorySTM(cfg)
        stm.update(_make_frame(5, base=10.0))
        features = stm.get_features()
        # Without normalization, values can be > 1
        assert features is not None

    def test_decay_weighting(self):
        """Recent frames should have more influence than older ones."""
        stm = AuditorySTM(AuditorySTMConfig(
            window_steps=4, decay_rate=0.5, normalize_output=False,
            include_recent_raw=False,
        ))
        # Add a frame with low values, then one with high values
        stm.update(_make_frame(3, base=0.0))
        stm.update(_make_frame(3, base=5.0))
        features = stm.get_features()
        # Mean should be pulled toward 5.0 (the recent frame) due to decay
        # The first 13 values are the weighted mean MFCC
        weighted_mean = features[:13]
        # Should be closer to 5.0 than to 0.0
        assert weighted_mean.mean() > 2.0

    def test_temporal_gradient(self):
        """Temporal gradient should detect transition from quiet to loud."""
        stm = AuditorySTM(AuditorySTMConfig(
            window_steps=4, decay_rate=1.0,  # no decay for simpler test
            normalize_output=False,
            include_delta=False, include_energy=False, include_recent_raw=False,
        ))
        stm.update(_make_frame(3, base=0.0))
        stm.update(_make_frame(3, base=0.0))
        stm.update(_make_frame(3, base=1.0))
        stm.update(_make_frame(3, base=1.0))
        features = stm.get_features()
        # gradient = features[13:26] (after mean, no delta/energy)
        gradient = features[13:26]
        # Should be positive (new_half > old_half)
        assert gradient.mean() > 0


class TestAuditorySTMPersistence:
    def test_round_trip(self, stm):
        stm.update(_make_frame(5, base=0.3))
        stm.update(_make_frame(3, base=0.7))
        original = stm.get_features()

        state = stm.get_state()
        stm2 = AuditorySTM(AuditorySTMConfig(window_steps=4, decay_rate=0.85))
        stm2.set_state(state)
        restored = stm2.get_features()

        np.testing.assert_allclose(original, restored, atol=1e-6)

    def test_state_dict_structure(self, stm):
        stm.update(_make_frame(5, base=0.5))
        state = stm.get_state()
        assert "mean_buffer" in state
        assert "delta_buffer" in state
        assert "energy_buffer" in state
        assert "raw_stack" in state
        assert "step_count" in state
        assert state["step_count"] == 1

    def test_empty_state_restore(self, stm):
        stm.set_state({})
        assert stm.n_frames == 0
        assert stm.get_features() is None

    def test_reset(self, stm):
        stm.update(_make_frame(5))
        assert stm.n_frames == 1
        stm.reset()
        assert stm.n_frames == 0
        assert stm.get_features() is None


class TestAuditorySTMConfig:
    def test_no_delta(self):
        stm = AuditorySTM(AuditorySTMConfig(
            include_delta=False, include_recent_raw=False, window_steps=4
        ))
        stm.update(_make_frame(5, base=0.5))
        f_no_delta = stm.get_features()

        stm2 = AuditorySTM(AuditorySTMConfig(
            include_delta=True, include_recent_raw=False, window_steps=4
        ))
        stm2.update(_make_frame(5, base=0.5))
        f_with_delta = stm2.get_features()

        # With delta should be 13 floats longer
        assert len(f_with_delta) == len(f_no_delta) + 13

    def test_no_energy(self):
        stm = AuditorySTM(AuditorySTMConfig(
            include_energy=False, include_recent_raw=False, window_steps=4
        ))
        stm.update(_make_frame(5, base=0.5))
        f_no_energy = stm.get_features()

        stm2 = AuditorySTM(AuditorySTMConfig(
            include_energy=True, include_recent_raw=False, window_steps=4
        ))
        stm2.update(_make_frame(5, base=0.5))
        f_with_energy = stm2.get_features()

        # With energy should be 3 floats longer
        assert len(f_with_energy) == len(f_no_energy) + 3


class TestAuditorySTMEdgeCases:
    """Edge cases found during code audit."""

    def test_fixed_length_feature_vector(self):
        """Feature vector length must be constant regardless of n_chunks in raw_stack."""
        stm = AuditorySTM(AuditorySTMConfig(
            window_steps=4, max_recent_raw_chunks=5
        ))
        # Frame with 2 chunks
        stm.update(_make_frame(2, base=0.1))
        f2 = stm.get_features()
        stm.reset()
        # Frame with 5 chunks
        stm.update(_make_frame(5, base=0.2))
        f5 = stm.get_features()
        stm.reset()
        # Frame with 8 chunks
        stm.update(_make_frame(8, base=0.3))
        f8 = stm.get_features()
        # All should have the same length
        assert len(f2) == len(f5) == len(f8)

    def test_raw_stack_padded_with_zeros(self):
        """When fewer chunks than max, raw portion should be zero-padded."""
        stm = AuditorySTM(AuditorySTMConfig(
            window_steps=4, max_recent_raw_chunks=5, normalize_output=False,
            include_delta=False, include_energy=False,
        ))
        stm.update(_make_frame(2, base=1.0))
        features = stm.get_features()
        # features: 13 (mean) + 13 (gradient) + 5*13 (raw padded) = 91
        raw_portion = features[26:]  # skip mean + gradient
        assert len(raw_portion) == 5 * 13
        # First 3*13=39 values should be zero (padding, right-aligned)
        assert np.all(raw_portion[:39] == 0.0)
        # Last 2*13=26 values should be non-zero (actual data)
        assert raw_portion[39:].sum() > 0

    def test_nan_energy_handled(self):
        """NaN in energy data should not propagate to features."""
        stm = AuditorySTM(AuditorySTMConfig(
            window_steps=4, include_recent_raw=False
        ))
        frame = _make_frame(3, base=0.5)
        # Inject NaN into energy
        frame["energy"] = [float("nan"), 0.5, 0.5]
        stm.update(frame)
        features = stm.get_features()
        assert features is not None
        assert np.all(np.isfinite(features))

    def test_constant_energy_no_crash(self):
        """All-identical energy values should not crash np.polyfit."""
        stm = AuditorySTM(AuditorySTMConfig(
            window_steps=4, include_recent_raw=False
        ))
        frame = _make_frame(5, base=0.5)
        frame["energy"] = [1.0, 1.0, 1.0, 1.0, 1.0]
        stm.update(frame)
        features = stm.get_features()
        assert features is not None
        assert np.all(np.isfinite(features))

    def test_no_raw_stack_still_fixed_length(self):
        """Even without raw_stack in the frame, feature vector is fixed length."""
        stm = AuditorySTM(AuditorySTMConfig(
            window_steps=4, max_recent_raw_chunks=5
        ))
        frame = _make_frame(3, base=0.5)
        del frame["raw_stack"]  # simulate missing raw_stack
        stm.update(frame)
        f1 = stm.get_features()

        stm2 = AuditorySTM(AuditorySTMConfig(
            window_steps=4, max_recent_raw_chunks=5
        ))
        stm2.update(_make_frame(3, base=0.5))  # with raw_stack
        f2 = stm2.get_features()

        # Both should have same length (raw portion zero-padded when missing)
        assert len(f1) == len(f2)
