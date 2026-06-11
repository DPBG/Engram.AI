"""Tests for LIF neuron populations."""

import numpy as np
import pytest

from neuromorphic.config import LIFParams
from neuromorphic.neurons import NeuronPopulation


class TestNeuronPopulation:
    def test_init_resting_potential(self):
        params = LIFParams(resting=-65.0)
        pop = NeuronPopulation(100, params)
        assert pop.n == 100
        np.testing.assert_allclose(pop.v_membrane, -65.0)
        assert pop.spikes.sum() == 0

    def test_no_spikes_without_input(self):
        params = LIFParams(noise_std=0.0)
        pop = NeuronPopulation(100, params)
        spikes = pop.step(np.zeros(100, dtype=np.float32))
        assert spikes.sum() == 0

    def test_spikes_with_strong_input(self):
        params = LIFParams(threshold=-55.0, resting=-65.0, tau=20.0, noise_std=0.0)
        pop = NeuronPopulation(100, params)
        # Strong current should depolarize past threshold
        current = np.full(100, 50.0, dtype=np.float32)
        # May take a few steps to reach threshold
        for _ in range(50):
            spikes = pop.step(current)
            if spikes.any():
                break
        assert spikes.sum() > 0, "Strong input should cause spikes"

    def test_refractory_period(self):
        params = LIFParams(
            threshold=-55.0, resting=-65.0, tau=20.0,
            refractory_ms=5.0, noise_std=0.0,
        )
        pop = NeuronPopulation(10, params)
        current = np.full(10, 100.0, dtype=np.float32)

        # Force spike
        pop.v_membrane[:] = params.threshold + 1.0
        pop.step(current)

        # Neuron should be in refractory — no spike for several steps
        for _ in range(4):
            spikes = pop.step(current)
            assert not spikes.any(), "Neuron in refractory should not spike"

    def test_reset_after_spike(self):
        params = LIFParams(threshold=-55.0, reset=-70.0, noise_std=0.0)
        pop = NeuronPopulation(10, params)
        pop.v_membrane[:] = params.threshold + 1.0
        pop.step(np.zeros(10, dtype=np.float32))
        # Spiked neurons should be at reset potential
        spiked = pop.spikes
        assert spiked.any(), "At least one neuron should have spiked"
        np.testing.assert_allclose(pop.v_membrane[spiked], params.reset)

    def test_vectorized_scale(self):
        """Test with a larger population to verify vectorized ops."""
        params = LIFParams(noise_std=0.1)
        pop = NeuronPopulation(10_000, params)
        current = np.random.randn(10_000).astype(np.float32) * 20.0
        spikes = pop.step(current)
        assert spikes.shape == (10_000,)
        assert spikes.dtype == bool

    def test_spike_time_tracking(self):
        params = LIFParams(threshold=-55.0, noise_std=0.0)
        pop = NeuronPopulation(10, params)
        pop.v_membrane[:] = params.threshold + 1.0
        pop.step(np.zeros(10, dtype=np.float32))
        # All neurons should have spiked at time 0+dt
        assert np.all(pop.last_spike_time >= 0)

    def test_state_serialization(self):
        params = LIFParams()
        pop = NeuronPopulation(50, params)
        pop.step(np.random.randn(50).astype(np.float32) * 10)
        state = pop.get_state()
        assert "v_membrane" in state
        assert "refractory_timer" in state
        assert "last_spike_time" in state
        assert "time" in state

        # Restore
        pop2 = NeuronPopulation(50, params)
        pop2.set_state(state)
        np.testing.assert_allclose(pop.v_membrane, pop2.v_membrane)

    def test_reset(self):
        params = LIFParams()
        pop = NeuronPopulation(50, params)
        pop.step(np.full(50, 100.0, dtype=np.float32))
        pop.reset()
        np.testing.assert_allclose(pop.v_membrane, params.resting)
        assert pop.spikes.sum() == 0

    def test_float32_dtype(self):
        pop = NeuronPopulation(100, LIFParams())
        assert pop.v_membrane.dtype == np.float32
        assert pop.refractory_timer.dtype == np.float32
        assert pop.last_spike_time.dtype == np.float32

    def test_sfa_raises_effective_threshold(self):
        """SFA should raise effective threshold for frequently-firing neurons."""
        params = LIFParams(
            threshold=-55.0, resting=-65.0, tau=20.0, refractory_ms=2.0,
            noise_std=0.0, sfa_increment=0.5, sfa_decay=0.99,
        )
        pop = NeuronPopulation(100, params)
        current = np.full(100, 30.0, dtype=np.float32)

        # Run 500 steps — SFA should accumulate
        for _ in range(500):
            pop.step(current)

        # SFA offset should be meaningfully positive for all neurons
        assert pop._sfa_offset.min() > 0.1, (
            f"SFA offset should accumulate: min={pop._sfa_offset.min():.4f}"
        )

    def test_sfa_reduces_high_firing_rate(self):
        """Strong SFA should reduce firing rate vs no SFA under identical input."""
        # No SFA baseline
        params_none = LIFParams(
            threshold=-55.0, resting=-65.0, tau=20.0, refractory_ms=2.0,
            noise_std=0.0, sfa_increment=0.0, sfa_decay=0.99,
        )
        # Aggressive SFA (5mV per spike, fast accumulation)
        params_strong = LIFParams(
            threshold=-55.0, resting=-65.0, tau=20.0, refractory_ms=2.0,
            noise_std=0.0, sfa_increment=5.0, sfa_decay=0.99,
        )
        pop_none = NeuronPopulation(100, params_none)
        pop_strong = NeuronPopulation(100, params_strong)
        current = np.full(100, 50.0, dtype=np.float32)  # strong: v_ss = -15

        total_none = 0
        total_strong = 0
        for _ in range(500):
            total_none += int(pop_none.step(current).sum())
            total_strong += int(pop_strong.step(current).sum())

        # Strong SFA should meaningfully reduce firing
        assert total_strong < total_none * 0.8, (
            f"Strong SFA should reduce firing: none={total_none}, strong={total_strong}"
        )
        assert total_strong > 0, "SFA should not completely silence all neurons"

    def test_sfa_state_persistence(self):
        """SFA offset should survive get_state/set_state."""
        params = LIFParams(sfa_increment=0.5)
        pop = NeuronPopulation(50, params)
        # Force spikes to build up SFA
        pop.v_membrane[:] = params.threshold + 1.0
        pop.step(np.zeros(50, dtype=np.float32))
        assert pop._sfa_offset.max() > 0, "SFA offset should increase after spikes"

        state = pop.get_state()
        assert "sfa_offset" in state

        pop2 = NeuronPopulation(50, params)
        pop2.set_state(state)
        np.testing.assert_allclose(pop._sfa_offset, pop2._sfa_offset)

    def test_sfa_decays_without_spikes(self):
        """SFA offset should decay toward zero when neuron stops firing."""
        params = LIFParams(sfa_increment=1.0, sfa_decay=0.9, noise_std=0.0)
        pop = NeuronPopulation(10, params)
        # Force a spike
        pop.v_membrane[:] = params.threshold + 1.0
        pop.step(np.zeros(10, dtype=np.float32))
        offset_after_spike = pop._sfa_offset[0]
        assert offset_after_spike > 0

        # Run with zero input (no spikes) — SFA should decay
        for _ in range(50):
            pop.step(np.zeros(10, dtype=np.float32))

        assert pop._sfa_offset[0] < offset_after_spike * 0.01, (
            "SFA should decay to near-zero without spikes"
        )


class TestDualInhibition:
    """Tests for GABA-A/GABA-B dual inhibition (CIP-21)."""

    def _make_dual_pop(self, n=200, pv_fraction=0.6):
        """Helper: create a population with dual inhibition enabled."""
        from neuromorphic.config import InhibitoryConfig, DualInhibitionConfig
        params = LIFParams(threshold=-55.0, resting=-65.0, tau=20.0, noise_std=0.0)
        inh_cfg = InhibitoryConfig(inhibitory_fraction=0.2)
        dual_cfg = DualInhibitionConfig(
            enabled=True, pv_fraction=pv_fraction,
            gaba_b_tau=150.0, gaba_b_conductance_increment=0.05,
        )
        return NeuronPopulation(
            n, params, inhibitory_config=inh_cfg, dual_inhibition_config=dual_cfg,
        )

    def test_pv_sst_split(self):
        """Inhibitory neurons should be split into PV and SST subtypes."""
        pop = self._make_dual_pop(200, pv_fraction=0.6)
        n_inh = int(pop.is_inhibitory.sum())
        n_pv = int((pop.is_inhibitory & ~pop.is_sst).sum())
        n_sst = int(pop.is_sst.sum())
        assert n_inh == 40  # 20% of 200
        assert n_pv == 24   # 60% of 40
        assert n_sst == 16  # 40% of 40
        # SST neurons are a subset of inhibitory
        assert (pop.is_sst & ~pop.is_inhibitory).sum() == 0

    def test_sst_excluded_from_signed_spikes(self):
        """SST neuron spikes should not appear in signed_spikes."""
        pop = self._make_dual_pop(200)
        # Force all neurons to spike
        pop.v_membrane[:] = pop.params.threshold + 5.0
        pop.step(np.zeros(200, dtype=np.float32))
        signed = pop.signed_spikes
        # SST neurons should contribute 0 in signed_spikes
        assert signed[pop.is_sst].sum() == 0.0
        # PV neurons should contribute -1 (inhibitory)
        pv_mask = pop.is_inhibitory & ~pop.is_sst
        assert (signed[pv_mask] < 0).all()
        # Excitatory neurons should contribute +1
        exc_mask = ~pop.is_inhibitory
        assert (signed[exc_mask] > 0).all()

    def test_sst_spikes_property(self):
        """sst_spikes should return only SST neuron spikes."""
        pop = self._make_dual_pop(200)
        pop.v_membrane[:] = pop.params.threshold + 5.0
        pop.step(np.zeros(200, dtype=np.float32))
        sst_sp = pop.sst_spikes
        # Should have spikes only where is_sst is True
        assert sst_sp[pop.is_sst].all()
        assert sst_sp[~pop.is_sst].sum() == 0

    def test_gaba_b_conductance_accumulates(self):
        """accumulate_gaba_b should increase gaba_b_conductance."""
        pop = self._make_dual_pop(200)
        assert pop._gaba_b_conductance is not None
        assert pop._gaba_b_conductance.sum() == 0.0
        # Simulate SST current arriving at post-synaptic neurons
        sst_current = np.zeros(200, dtype=np.float32)
        sst_current[0:10] = 1.0  # 10 neurons receive SST input
        pop.accumulate_gaba_b(sst_current)
        assert pop._gaba_b_conductance[0:10].sum() > 0
        assert pop._gaba_b_conductance[10:].sum() == 0.0

    def test_gaba_b_conductance_decays(self):
        """GABA-B conductance should decay with tau ~150ms."""
        pop = self._make_dual_pop(100)
        # Inject GABA-B conductance
        sst_current = np.ones(100, dtype=np.float32)
        pop.accumulate_gaba_b(sst_current)
        initial_g = pop._gaba_b_conductance[0]
        assert initial_g > 0
        # Step with no input -- conductance should decay
        for _ in range(10):
            pop.step(np.zeros(100, dtype=np.float32))
        after_10 = pop._gaba_b_conductance[0]
        assert after_10 < initial_g, "GABA-B should decay each step"
        # After 150 steps (~1 tau), should be ~37% of peak
        pop2 = self._make_dual_pop(100)
        pop2.accumulate_gaba_b(np.ones(100, dtype=np.float32))
        peak = pop2._gaba_b_conductance[0]
        for _ in range(150):
            pop2.step(np.zeros(100, dtype=np.float32))
        ratio = pop2._gaba_b_conductance[0] / peak
        assert 0.2 < ratio < 0.5, f"After 1 tau, ~37% expected, got {ratio:.3f}"

    def test_gaba_b_hyperpolarizes(self):
        """GABA-B current should hyperpolarize membrane potential."""
        pop = self._make_dual_pop(100)
        # Inject strong GABA-B conductance
        sst_current = np.full(100, 10.0, dtype=np.float32)
        pop.accumulate_gaba_b(sst_current)
        # Step and check voltage drops below resting
        pop.step(np.zeros(100, dtype=np.float32))
        # Excitatory neurons should be hyperpolarized below resting
        exc_mask = ~pop.is_inhibitory
        # GABA-B reversal is -90mV, resting is -65mV, so current is negative
        assert pop.v_membrane[exc_mask].mean() < pop.params.resting, (
            f"GABA-B should hyperpolarize: mean={pop.v_membrane[exc_mask].mean():.2f}"
        )

    def test_gaba_b_suppresses_longer_than_gaba_a(self):
        """SST (GABA-B) inhibition should outlast PV (GABA-A) by 10x+."""
        pop = self._make_dual_pop(100)
        # Inject GABA-B conductance equivalent to one round of SST spikes
        sst_current = np.full(100, 5.0, dtype=np.float32)
        pop.accumulate_gaba_b(sst_current)
        # Track how many steps GABA-B conductance stays above 10% of peak
        peak = pop._gaba_b_conductance[0]
        steps_above_10pct = 0
        for _ in range(500):
            pop.step(np.zeros(100, dtype=np.float32))
            if pop._gaba_b_conductance[0] > peak * 0.1:
                steps_above_10pct += 1
        # GABA-B should persist for >100 steps (100ms+)
        # GABA-A (instant, tau ~10ms) would be gone in ~30 steps
        assert steps_above_10pct > 100, (
            f"GABA-B should persist >100ms, only lasted {steps_above_10pct} steps"
        )

    def test_backward_compat_disabled(self):
        """When dual inhibition is disabled, behavior should be unchanged."""
        from neuromorphic.config import InhibitoryConfig, DualInhibitionConfig
        params = LIFParams(threshold=-55.0, resting=-65.0, noise_std=0.0)
        inh_cfg = InhibitoryConfig(inhibitory_fraction=0.2)
        disabled_cfg = DualInhibitionConfig(enabled=False)
        pop = NeuronPopulation(
            100, params, inhibitory_config=inh_cfg,
            dual_inhibition_config=disabled_cfg,
        )
        assert pop.is_sst.sum() == 0
        assert pop._gaba_b_conductance is None
        # signed_spikes should work as before (all inhibitory neurons contribute)
        pop.v_membrane[:] = params.threshold + 5.0
        pop.step(np.zeros(100, dtype=np.float32))
        inh_signed = pop.signed_spikes[pop.is_inhibitory]
        assert (inh_signed < 0).all(), "All inhibitory should contribute when disabled"

    def test_gaba_b_state_persistence(self):
        """GABA-B conductance and is_sst should survive get_state/set_state."""
        pop = self._make_dual_pop(100)
        sst_current = np.ones(100, dtype=np.float32) * 3.0
        pop.accumulate_gaba_b(sst_current)
        state = pop.get_state()
        assert "gaba_b_conductance" in state
        assert "is_sst" in state
        # Restore into fresh population
        pop2 = self._make_dual_pop(100)
        pop2.set_state(state)
        np.testing.assert_allclose(
            pop._gaba_b_conductance, pop2._gaba_b_conductance
        )
        np.testing.assert_array_equal(pop.is_sst, pop2.is_sst)

    def test_sst_params_differ_from_pv(self):
        """SST neurons should have different tau/threshold than PV neurons."""
        pop = self._make_dual_pop(200)
        pv_mask = pop.is_inhibitory & ~pop.is_sst
        sst_mask = pop.is_sst
        # SST has tau=20ms (config default), PV has tau=10ms (InhibitoryConfig)
        assert pop._tau[pv_mask][0] == 10.0
        assert pop._tau[sst_mask][0] == 20.0
        # SST threshold differs from PV
        assert pop._threshold[sst_mask][0] != pop._threshold[pv_mask][0]
