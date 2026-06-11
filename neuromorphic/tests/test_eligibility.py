"""Tests for eligibility traces and BCM metaplasticity (Phases 5, 8)."""

import numpy as np
import pytest

from neuromorphic.config import STDPParams, EligibilityTraceConfig, BCMConfig
from neuromorphic.synapses import SynapseGroup


class TestEligibilityTraces:
    """Test three-factor learning with eligibility traces."""

    def _make_synapse(self, **kwargs):
        defaults = dict(
            n_pre=50, n_post=50, sparsity=0.1, init_weight=0.5,
            plastic=True, rng=np.random.default_rng(42),
            eligibility_config=EligibilityTraceConfig(),
        )
        defaults.update(kwargs)
        return SynapseGroup(**defaults)

    def test_eligibility_created(self):
        """Plastic synapse with eligibility config should have trace array."""
        sg = self._make_synapse()
        assert sg.eligibility is not None
        assert sg.eligibility.shape == (sg.nnz,)
        assert (sg.eligibility == 0.0).all()

    def test_no_eligibility_without_config(self):
        """Without eligibility config, no trace array."""
        sg = SynapseGroup(
            n_pre=50, n_post=50, sparsity=0.1, init_weight=0.5,
            plastic=True, rng=np.random.default_rng(42),
        )
        assert sg.eligibility is None

    def test_no_eligibility_non_plastic(self):
        """Non-plastic synapse should not have eligibility."""
        sg = SynapseGroup(
            n_pre=50, n_post=50, sparsity=0.1, init_weight=0.5,
            plastic=False, rng=np.random.default_rng(42),
            eligibility_config=EligibilityTraceConfig(),
        )
        assert sg.eligibility is None

    def test_stdp_updates_eligibility_not_weights(self):
        """With eligibility traces, STDP should update traces, not weights directly."""
        sg = self._make_synapse()
        weights_before = sg.weights.data.copy()

        # Create spike patterns — fire ALL pre and post so we hit some actual synapses
        pre_spikes = np.ones(50, dtype=bool)
        post_spikes = np.ones(50, dtype=bool)
        pre_times = np.full(50, 1.0, dtype=np.float32)
        post_times = np.full(50, 2.0, dtype=np.float32)  # post after pre → LTP

        sg.update_weights_stdp(pre_spikes, post_spikes, pre_times, post_times, 2.0)

        # Weights should NOT have changed (eligibility is active)
        np.testing.assert_array_equal(sg.weights.data, weights_before)
        # Eligibility should have non-zero values
        assert sg.eligibility.any()

    def test_neuromodulation_applies_traces(self):
        """apply_neuromodulation should convert eligibility to weight changes."""
        sg = self._make_synapse()
        # Manually set some eligibility
        sg.eligibility[:10] = 0.01

        weights_before = sg.weights.data[:10].copy()
        sg.apply_neuromodulation(2.0)  # strong neuromodulator

        # Weights should have changed
        expected = np.clip(weights_before + 0.01 * 2.0, sg.stdp_params.w_min, sg.stdp_params.w_max)
        np.testing.assert_allclose(sg.weights.data[:10], expected, atol=1e-6)

    def test_eligibility_decay(self):
        """Eligibility traces should decay each step."""
        sg = self._make_synapse()
        sg.eligibility[:10] = 1.0
        sg.decay_eligibility()
        assert sg.eligibility[:10].max() < 1.0
        assert sg.eligibility[:10].max() > 0.9  # decay rate 0.999

    def test_eligibility_state_roundtrip(self):
        """Eligibility should be saved and restored."""
        sg = self._make_synapse()
        sg.eligibility[:5] = 0.42

        state = sg.get_state()
        assert "eligibility" in state

        sg2 = self._make_synapse()
        sg2.set_state(state)
        np.testing.assert_allclose(sg2.eligibility[:5], 0.42, atol=1e-6)


class TestBCMMetaplasticity:
    """Test BCM per-neuron modification threshold."""

    def _make_synapse_with_bcm(self):
        sg = SynapseGroup(
            n_pre=50, n_post=50, sparsity=0.1, init_weight=0.5,
            plastic=True, rng=np.random.default_rng(42),
        )
        sg.enable_bcm(BCMConfig())
        return sg

    def test_bcm_enabled(self):
        """enable_bcm should create theta array."""
        sg = self._make_synapse_with_bcm()
        assert sg.bcm_theta is not None
        assert sg.bcm_theta.shape == (50,)
        assert (sg.bcm_theta == BCMConfig().theta_init).all()

    def test_bcm_not_on_non_plastic(self):
        """Non-plastic synapses should ignore enable_bcm."""
        sg = SynapseGroup(
            n_pre=50, n_post=50, sparsity=0.1, init_weight=0.5,
            plastic=False, rng=np.random.default_rng(42),
        )
        sg.enable_bcm(BCMConfig())
        assert sg.bcm_theta is None

    def test_bcm_threshold_adapts(self):
        """BCM threshold should increase with high firing rates."""
        sg = self._make_synapse_with_bcm()
        initial_theta = sg.bcm_theta.copy()

        # High firing rate
        high_rates = np.full(50, 0.5, dtype=np.float32)
        sg.update_bcm_threshold(high_rates)

        # Theta should have increased
        assert (sg.bcm_theta > initial_theta).all()

    def test_bcm_threshold_decreases_with_silence(self):
        """BCM threshold should decrease with low firing rates."""
        sg = self._make_synapse_with_bcm()
        # Set theta high first
        sg.bcm_theta[:] = 0.1
        initial_theta = sg.bcm_theta.copy()

        # Zero firing rate
        zero_rates = np.zeros(50, dtype=np.float32)
        sg.update_bcm_threshold(zero_rates)

        # Theta should have decreased (pulled toward 0)
        assert (sg.bcm_theta < initial_theta).all()

    def test_bcm_scaling_active_harder(self):
        """Active neurons (high theta) should have lower LTP scaling."""
        sg = self._make_synapse_with_bcm()
        sg.bcm_theta[:25] = 0.001  # quiet neurons
        sg.bcm_theta[25:] = 0.1    # active neurons

        scaling = sg._get_bcm_scaling()
        assert scaling is not None
        # Quiet neurons should have HIGHER scaling (easier to potentiate)
        assert scaling[:25].mean() > scaling[25:].mean()

    def test_bcm_state_roundtrip(self):
        """BCM theta should be saved and restored."""
        sg = self._make_synapse_with_bcm()
        sg.bcm_theta[:] = 0.05

        state = sg.get_state()
        assert "bcm_theta" in state

        sg2 = self._make_synapse_with_bcm()
        sg2.set_state(state)
        np.testing.assert_allclose(sg2.bcm_theta, 0.05, atol=1e-6)
