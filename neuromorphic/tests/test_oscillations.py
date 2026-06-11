"""Tests for oscillatory dynamics module."""

import math

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig, OscillatoryConfig
from neuromorphic.oscillations import OscillatorBank
from neuromorphic.network import NeuromorphicNetwork


class TestOscillatorBank:
    """Unit tests for the OscillatorBank."""

    @pytest.fixture
    def bank(self):
        return OscillatorBank(OscillatoryConfig(enabled=True))

    def test_initial_phases_zero(self, bank):
        assert bank._gamma_phase == 0.0
        assert bank._theta_phase == 0.0

    def test_step_advances_phase(self, bank):
        bank.step(1.0)
        assert bank._gamma_phase > 0.0
        assert bank._theta_phase > 0.0

    def test_gamma_period(self, bank):
        """Gamma at 40 Hz should complete one cycle in 25 ms (25 steps at dt=1)."""
        for _ in range(25):
            bank.step(1.0)
        # Phase should be close to 2*pi (one full cycle)
        assert abs(bank._gamma_phase - 0.0) < 0.01 or abs(bank._gamma_phase - 2 * math.pi) < 0.01

    def test_theta_period(self, bank):
        """Theta at 6 Hz should complete one cycle in ~167 ms."""
        period_steps = round(1000.0 / 6.0)  # ~167
        for _ in range(period_steps):
            bank.step(1.0)
        # Should be near 2*pi
        assert abs(bank._theta_phase) < 0.05 or abs(bank._theta_phase - 2 * math.pi) < 0.05

    def test_gamma_gain_range(self, bank):
        """Gamma gain should be in [1-amp, 1+amp] = [0.85, 1.15]."""
        gains = []
        for _ in range(100):
            bank.step(1.0)
            gains.append(bank.gamma_gain())
        assert min(gains) >= 0.84  # allow float tolerance
        assert max(gains) <= 1.16
        # Should have variation
        assert max(gains) - min(gains) > 0.1

    def test_theta_plasticity_range(self, bank):
        """Theta factor should be in [1-amp, 1+amp] = [0.75, 1.25]."""
        factors = []
        for _ in range(200):
            bank.step(1.0)
            factors.append(bank.theta_plasticity_factor())
        assert min(factors) >= 0.74
        assert max(factors) <= 1.26
        assert max(factors) - min(factors) > 0.2

    def test_gamma_gain_always_positive(self, bank):
        """Gain must never go negative (would flip current direction)."""
        for _ in range(1000):
            bank.step(1.0)
            assert bank.gamma_gain() > 0

    def test_theta_factor_always_positive(self, bank):
        for _ in range(1000):
            bank.step(1.0)
            assert bank.theta_plasticity_factor() > 0

    def test_coherence_empty(self, bank):
        """No modalities → zero coherence."""
        assert bank.cross_modal_coherence() == 0.0

    def test_coherence_single_modality(self, bank):
        """One modality → zero coherence (need pairs)."""
        bank.update_modality_phase("visual")
        assert bank.cross_modal_coherence() == 0.0

    def test_coherence_synchronized(self, bank):
        """Two modalities locking at same phase → high coherence."""
        bank.step(10.0)
        bank.update_modality_phase("visual")
        bank.update_modality_phase("auditory")
        # Both locked to same gamma phase → perfect coherence = 1.0
        assert bank.cross_modal_coherence() > 0.99

    def test_coherence_desynchronized(self, bank):
        """Modalities locking at different phases → lower coherence."""
        bank.step(1.0)
        bank.update_modality_phase("visual")
        # Advance half a gamma cycle
        for _ in range(12):
            bank.step(1.0)
        bank.update_modality_phase("auditory")
        # ~180 degrees apart → low coherence
        assert bank.cross_modal_coherence() < 0.5

    def test_binding_boost_no_coherence(self, bank):
        """No coherence → boost = 1.0."""
        assert bank.binding_boost() == 1.0

    def test_binding_boost_high_coherence(self, bank):
        """Perfect coherence → boost > 1.0."""
        bank.update_modality_phase("visual")
        bank.update_modality_phase("auditory")
        assert bank.binding_boost() > 1.0

    def test_state_roundtrip(self, bank):
        """State save/restore preserves phases."""
        for _ in range(50):
            bank.step(1.0)
        bank.update_modality_phase("visual")
        state = bank.get_state()

        bank2 = OscillatorBank(OscillatoryConfig(enabled=True))
        bank2.set_state(state)
        assert abs(bank2._gamma_phase - bank._gamma_phase) < 1e-10
        assert abs(bank2._theta_phase - bank._theta_phase) < 1e-10
        assert bank2._modality_phases == bank._modality_phases


class TestOscillatoryNetworkIntegration:
    """Integration tests for oscillations in the network."""

    @pytest.fixture
    def osc_config(self):
        cfg = NeuromorphicConfig()
        cfg.populations.brainstem = 50
        cfg.populations.reflex_arc = 30
        cfg.populations.sensory_cortex = 100
        cfg.populations.motor_cortex = 80
        cfg.populations.cerebellum = 60
        cfg.populations.association_cortex = 100
        cfg.populations.predictive_layer = 60
        cfg.populations.working_memory = 30
        cfg.oscillatory = OscillatoryConfig(enabled=True)
        return cfg

    @pytest.fixture
    def osc_network(self, osc_config):
        return NeuromorphicNetwork(osc_config, seed=42)

    def test_oscillators_created(self, osc_network):
        assert osc_network.oscillators is not None

    def test_step_with_oscillations(self, osc_network):
        """Oscillations don't break the simulation loop."""
        obs = np.random.default_rng(0).random(100).astype(np.float32)
        result = osc_network.step(obs)
        assert result is not None
        assert osc_network.oscillators._gamma_phase > 0

    def test_metrics_include_oscillatory(self, osc_network):
        obs = np.random.default_rng(0).random(100).astype(np.float32)
        osc_network.step(obs)
        metrics = osc_network.get_metrics()
        assert "oscillatory" in metrics
        assert metrics["oscillatory"]["enabled"] is True
        assert metrics["oscillatory"]["gamma_gain"] != 0

    def test_state_persistence_with_oscillations(self, osc_network):
        """Oscillator state survives save/restore."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            osc_network.step(rng.random(100).astype(np.float32))
        state = osc_network.get_state()
        gamma_before = osc_network.oscillators._gamma_phase

        # Create new network and restore
        cfg2 = osc_network.config
        net2 = NeuromorphicNetwork(cfg2, seed=99)
        net2.set_state(state)
        assert abs(net2.oscillators._gamma_phase - gamma_before) < 1e-10

    def test_backward_compat_no_oscillations(self):
        """Default config (oscillatory.enabled=False) creates no oscillators."""
        cfg = NeuromorphicConfig()
        cfg.populations.brainstem = 50
        cfg.populations.reflex_arc = 30
        cfg.populations.sensory_cortex = 100
        cfg.populations.motor_cortex = 80
        cfg.populations.cerebellum = 60
        cfg.populations.association_cortex = 100
        cfg.populations.predictive_layer = 60
        cfg.populations.working_memory = 30
        net = NeuromorphicNetwork(cfg, seed=42)
        assert net.oscillators is None
        # Metrics should still have oscillatory section (disabled)
        metrics = net.get_metrics()
        assert metrics["oscillatory"]["enabled"] is False
        assert metrics["oscillatory"]["gamma_gain"] == 1.0

    def test_inject_observation_tracks_phase(self, osc_network):
        """inject_observation should update modality phase in oscillator."""
        osc_network.step(None)  # advance phase
        data = np.random.default_rng(0).random(10).astype(np.float32).tolist()
        osc_network.inject_observation(data, "observation.visual.camera")
        assert "visual" in osc_network.oscillators._modality_phases

    def test_inject_multimodal_tracks_phases(self, osc_network):
        """inject_multimodal should update modality phases for allocated modalities."""
        osc_network.step(None)
        data = {
            "observation.visual.camera": np.random.default_rng(0).random(10).astype(np.float32).tolist(),
            "observation.auditory.mic": np.random.default_rng(1).random(10).astype(np.float32).tolist(),
        }
        osc_network.inject_multimodal(data)
        phases = osc_network.oscillators._modality_phases
        # At least one modality should be tracked (both may be if allocator assigns ranges)
        assert len(phases) >= 1
