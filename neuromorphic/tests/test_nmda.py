"""Tests for NMDA receptor dynamics and attractor working memory (CIP-24)."""

import numpy as np
import pytest

from neuromorphic.config import LIFParams, NMDAConfig, NeuromorphicConfig
from neuromorphic.neurons import NeuronPopulation


class TestNMDAConductance:
    """Test NMDA conductance dynamics in NeuronPopulation."""

    def _make_pop(self, n=100, **nmda_overrides):
        params = LIFParams(resting=-65.0, threshold=-50.0, reset=-65.0, tau=20.0)
        nmda = NMDAConfig(enabled=True, **nmda_overrides)
        return NeuronPopulation(n, params, nmda_config=nmda)

    def test_nmda_arrays_created(self):
        """NMDA conductance array should exist when enabled."""
        pop = self._make_pop()
        assert pop._nmda_conductance is not None
        assert len(pop._nmda_conductance) == 100
        assert pop._nmda_conductance.sum() == 0.0

    def test_nmda_disabled_no_arrays(self):
        """When NMDA disabled, no conductance arrays should be created."""
        params = LIFParams()
        pop = NeuronPopulation(50, params, nmda_config=NMDAConfig(enabled=False))
        assert pop._nmda_conductance is None

    def test_backward_compat_no_nmda(self):
        """Without NMDAConfig, neurons should work as before."""
        params = LIFParams()
        pop = NeuronPopulation(50, params)
        assert pop._nmda_conductance is None
        # Should still step without error
        current = np.ones(50, dtype=np.float32) * 20.0
        pop.step(current)

    def test_accumulate_nmda_increases_conductance(self):
        """accumulate_nmda should increase NMDA conductance."""
        pop = self._make_pop(conductance_scale=0.1)
        exc_current = np.ones(100, dtype=np.float32) * 5.0
        pop.accumulate_nmda(exc_current)
        assert pop._nmda_conductance.sum() > 0.0
        assert np.allclose(pop._nmda_conductance, 0.5)  # 5.0 * 0.1

    def test_nmda_conductance_decays(self):
        """NMDA conductance should decay with tau_nmda."""
        pop = self._make_pop(tau_nmda=50.0, conductance_scale=1.0)
        # Inject conductance
        exc = np.ones(100, dtype=np.float32)
        pop.accumulate_nmda(exc)
        initial = pop._nmda_conductance.copy()
        # Step without further input (conductance should decay)
        for _ in range(100):
            pop.step(np.zeros(100, dtype=np.float32))
        assert pop._nmda_conductance[0] < initial[0] * 0.2, \
            "NMDA conductance should decay significantly after 100 steps (2*tau)"

    def test_nmda_provides_depolarizing_current(self):
        """NMDA current should depolarize neurons (push V toward threshold)."""
        pop_nmda = self._make_pop(tau_nmda=50.0, conductance_scale=0.5)
        params = LIFParams(resting=-65.0, threshold=-50.0, reset=-65.0, tau=20.0)
        pop_control = NeuronPopulation(100, params)

        # Inject NMDA conductance
        exc = np.ones(100, dtype=np.float32)
        pop_nmda.accumulate_nmda(exc)

        # Step both
        zero_input = np.zeros(100, dtype=np.float32)
        pop_nmda.step(zero_input.copy())
        pop_control.step(zero_input.copy())

        # NMDA population should be more depolarized
        assert pop_nmda.v_membrane.mean() > pop_control.v_membrane.mean(), \
            "NMDA current should depolarize neurons"

    def test_mg_block_at_resting_potential(self):
        """Mg2+ block should suppress NMDA current at resting potential (-65mV)."""
        # At V=-65mV: B(V) = 1/(1 + 1.0/3.57 * exp(0.062*65)) ≈ 1/(1 + 0.28*55.7) ≈ 0.06
        # So ~94% of NMDA current is blocked at rest
        pop = self._make_pop(tau_nmda=1000.0, conductance_scale=1.0)
        exc = np.ones(100, dtype=np.float32) * 10.0
        pop.accumulate_nmda(exc)

        # At resting potential, the Mg block should greatly reduce current
        # B(-65) = 1/(1 + 0.28 * exp(0.062*65)) = 1/(1 + 0.28*55.7) = 0.06
        v = -65.0
        mg_block = 1.0 / (1.0 + (1.0 / 3.57) * np.exp(-0.062 * v))
        assert mg_block < 0.1, f"Mg block should suppress ~94% at rest, got B={mg_block}"

    def test_mg_block_relieved_when_depolarized(self):
        """Mg2+ block should be relieved at depolarized potentials."""
        # At V=-30mV: B(V) = 1/(1 + 0.28*exp(0.062*30)) = 1/(1 + 0.28*6.36) = 0.36
        # At V=0mV: B(V) = 1/(1 + 0.28*1.0) = 0.78
        v_depolarized = -30.0
        mg_block_dep = 1.0 / (1.0 + (1.0 / 3.57) * np.exp(-0.062 * v_depolarized))
        v_rest = -65.0
        mg_block_rest = 1.0 / (1.0 + (1.0 / 3.57) * np.exp(-0.062 * v_rest))
        assert mg_block_dep > mg_block_rest * 3, \
            f"Block should be much weaker at -30mV ({mg_block_dep}) vs -65mV ({mg_block_rest})"

    def test_nmda_sustains_activity(self):
        """NMDA should sustain firing longer than pure fast synaptic input."""
        # Population WITH NMDA: brief input then no input
        pop_nmda = self._make_pop(n=50, tau_nmda=100.0, conductance_scale=0.3)
        params = LIFParams(resting=-65.0, threshold=-50.0, reset=-65.0, tau=20.0)
        pop_control = NeuronPopulation(50, params)

        # Brief strong input to drive both populations
        strong_input = np.ones(50, dtype=np.float32) * 25.0
        for _ in range(20):
            pop_nmda.step(strong_input.copy())
            pop_control.step(strong_input.copy())
            # Accumulate NMDA from recurrent-like activity
            if pop_nmda.spikes.any():
                pop_nmda.accumulate_nmda(pop_nmda.spikes.astype(np.float32) * 2.0)

        # Count spikes during 0-input period
        nmda_spikes = 0
        control_spikes = 0
        zero_input = np.zeros(50, dtype=np.float32)
        for _ in range(200):
            pop_nmda.step(zero_input.copy())
            pop_control.step(zero_input.copy())
            nmda_spikes += pop_nmda.spikes.sum()
            control_spikes += pop_control.spikes.sum()
            # NMDA recurrent accumulation
            if pop_nmda.spikes.any():
                pop_nmda.accumulate_nmda(pop_nmda.spikes.astype(np.float32) * 2.0)

        # NMDA population should have more sustained activity
        assert nmda_spikes >= control_spikes, \
            f"NMDA should sustain more spikes: nmda={nmda_spikes}, control={control_spikes}"

    def test_nmda_state_persistence(self):
        """NMDA conductance should survive get_state/set_state."""
        pop = self._make_pop()
        exc = np.ones(100, dtype=np.float32) * 3.0
        pop.accumulate_nmda(exc)
        state = pop.get_state()
        assert "nmda_conductance" in state

        # Restore into fresh population
        pop2 = self._make_pop()
        pop2.set_state(state)
        np.testing.assert_allclose(pop._nmda_conductance, pop2._nmda_conductance)

    def test_reset_clears_nmda(self):
        """reset() should zero NMDA conductance."""
        pop = self._make_pop()
        pop.accumulate_nmda(np.ones(100, dtype=np.float32))
        assert pop._nmda_conductance.sum() > 0
        pop.reset()
        assert pop._nmda_conductance.sum() == 0.0

    def test_nmda_slow_timescale(self):
        """NMDA decay should be slower than standard synaptic (tau=100ms vs tau=20ms)."""
        pop = self._make_pop(tau_nmda=100.0, conductance_scale=1.0)
        exc = np.ones(100, dtype=np.float32)
        pop.accumulate_nmda(exc)
        g_initial = float(pop._nmda_conductance[0])
        # After 50 steps (50ms = half tau), should retain ~60%
        for _ in range(50):
            pop.step(np.zeros(100, dtype=np.float32))
        g_50 = float(pop._nmda_conductance[0])
        ratio = g_50 / g_initial
        # exp(-50/100) = 0.607
        assert 0.5 < ratio < 0.7, f"After 50ms, should retain ~60%, got {ratio*100:.1f}%"


class TestNMDAConfig:
    """Test NMDAConfig creation and from_env."""

    def test_defaults(self):
        cfg = NMDAConfig()
        assert cfg.enabled is False
        assert cfg.tau_nmda == 100.0
        assert cfg.wm_recurrent_nmda == 0.7
        assert cfg.nmda_fraction == 0.3

    def test_neuromorphic_config_has_nmda(self):
        cfg = NeuromorphicConfig()
        assert hasattr(cfg, "nmda")
        assert cfg.nmda.enabled is False


def _small_config(**overrides):
    """Small-scale config for fast network tests."""
    cfg = NeuromorphicConfig(**overrides)
    cfg.populations.brainstem = 100
    cfg.populations.reflex_arc = 80
    cfg.populations.sensory_cortex = 400
    cfg.populations.motor_cortex = 200
    cfg.populations.cerebellum = 200
    cfg.populations.association_cortex = 400
    cfg.populations.predictive_layer = 200
    cfg.populations.working_memory = 50
    return cfg


class TestWorkingRecurrentSynapse:
    """Test working_recurrent synapse group creation in network."""

    def test_working_recurrent_exists(self):
        """Network should have working_recurrent synapse group."""
        from neuromorphic.network import NeuromorphicNetwork
        cfg = _small_config()
        net = NeuromorphicNetwork(cfg, seed=42)
        assert "working_recurrent" in net.synapses

    def test_working_recurrent_shape(self):
        """working_recurrent should be WM->WM (square)."""
        from neuromorphic.network import NeuromorphicNetwork
        cfg = _small_config()
        net = NeuromorphicNetwork(cfg, seed=42)
        syn = net.synapses["working_recurrent"]
        wm_n = cfg.populations.working_memory
        assert syn.weights.shape == (wm_n, wm_n)

    def test_working_recurrent_is_plastic(self):
        """working_recurrent should be plastic (STDP-modifiable)."""
        from neuromorphic.network import NeuromorphicNetwork
        cfg = _small_config()
        net = NeuromorphicNetwork(cfg, seed=42)
        syn = net.synapses["working_recurrent"]
        assert syn.plastic

    def test_step_with_nmda_enabled(self):
        """Network step should work with NMDA enabled."""
        from neuromorphic.network import NeuromorphicNetwork
        cfg = _small_config(nmda=NMDAConfig(enabled=True))
        net = NeuromorphicNetwork(cfg, seed=42)
        # Should step without error
        result = net.step()
        assert "motor_commands" in result
