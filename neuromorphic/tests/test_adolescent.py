"""Tests for adolescent brain phase mechanisms.

Tests cover:
- Dynamic phase transition (entry/exit criteria)
- Concept differentiation tracker
- Synaptic pruning
- Myelination (plasticity lock-in)
- Identity tagging
- STDP window widening
- Neighborhood consolidation
- Phase-dependent instinct gains
- Convergence tracking
- State persistence through adolescent phase
"""

import numpy as np
import pytest

from neuromorphic.config import (
    NeuromorphicConfig,
    STDPParams,
    PruningConfig,
    MyelinationConfig,
    NeighborhoodConsolidationConfig,
    AdolescentSTDPConfig,
    AdolescentEntryConfig,
    CriticalPeriodConfig,
    EligibilityTraceConfig,
)
from neuromorphic.synapses import SynapseGroup
from neuromorphic.neuromodulation import NeuromodulationSystem, ConceptDifferentiationTracker
from neuromorphic.instincts import OrientingInstincts
from neuromorphic.network import NeuromorphicNetwork


class TestConceptDifferentiationTracker:
    def test_empty_tracker(self):
        cfg = AdolescentEntryConfig()
        tracker = ConceptDifferentiationTracker(cfg)
        assert tracker.count_distinct() == 0
        assert not tracker.is_differentiated

    def test_single_pattern(self):
        cfg = AdolescentEntryConfig()
        tracker = ConceptDifferentiationTracker(cfg)
        tracker.record_pattern(np.random.randn(500).astype(np.float32))
        assert tracker.count_distinct() == 1

    def test_distinct_patterns(self):
        cfg = AdolescentEntryConfig(min_concept_patterns=3, concept_similarity_threshold=0.3)
        tracker = ConceptDifferentiationTracker(cfg, n_sample=100)
        rng = np.random.default_rng(42)
        # Create 5 very different patterns (orthogonal-ish random vectors)
        for i in range(5):
            pat = np.zeros(100, dtype=np.float32)
            pat[i * 20:(i + 1) * 20] = rng.random(20).astype(np.float32)
            tracker.record_pattern(pat)
        assert tracker.count_distinct() >= 3
        assert tracker.is_differentiated

    def test_similar_patterns_not_counted(self):
        cfg = AdolescentEntryConfig(min_concept_patterns=3, concept_similarity_threshold=0.3)
        tracker = ConceptDifferentiationTracker(cfg, n_sample=50)
        # Record the same pattern 10 times with small noise
        base = np.random.default_rng(42).random(50).astype(np.float32)
        for _ in range(10):
            noisy = base + np.random.default_rng().normal(0, 0.01, 50).astype(np.float32)
            tracker.record_pattern(noisy)
        # Should still be ~1 distinct pattern
        assert tracker.count_distinct() <= 2


class TestNeuromodulationAdolescent:
    def _make_config(self, **kwargs):
        entry = AdolescentEntryConfig(
            min_steps=100,
            consecutive_checks=2,
            check_interval=10,
            min_concept_patterns=2,
            sensory_stability_threshold=0.1,
            feature_stdp_decline=0.5,
            max_duration=500,
            min_duration=50,  # small for test (default 200K too large for unit tests)
        )
        cp = CriticalPeriodConfig(
            infant_end=10,
            toddler_end=30,
            juvenile_end=50,
        )
        return NeuromorphicConfig(
            critical_period=cp,
            adolescent_entry=entry,
            **kwargs,
        )

    def test_phase_progression_without_adolescent(self):
        """Phases go infant→toddler→juvenile→mature when criteria not met."""
        config = self._make_config()
        nm = NeuromodulationSystem(config)
        nm.update(5)
        assert nm.phase == "infant"
        nm.update(20)
        assert nm.phase == "toddler"
        nm.update(40)
        assert nm.phase == "juvenile"
        nm.update(100)
        assert nm.phase == "mature"
        assert not nm.is_adolescent

    def test_adolescent_entry_requires_all_criteria(self):
        """Adolescent entry requires concept differentiation + sensory stability + feature decline."""
        config = self._make_config()
        nm = NeuromodulationSystem(config)

        # Get past juvenile end
        for step in range(60):
            nm.update(step)

        # Record distinct patterns
        rng = np.random.default_rng(42)
        for i in range(5):
            pat = np.zeros(200, dtype=np.float32)
            pat[i * 40:(i + 1) * 40] = rng.random(40).astype(np.float32)
            nm.concept_tracker.record_pattern(pat)

        # Set external signals: sensory stable, feature STDP declined
        nm.update_external_signals(
            sensory_rate_variance=0.01,  # < 0.1 threshold
            feature_stdp_current=0.1,
            feature_stdp_peak=1.0,  # 0.1 < 0.5 * 1.0
            myelination_fraction=0.0,
        )

        # Run enough checks to trigger entry (check_interval=10, consecutive_checks=2)
        for step in range(100, 130):
            nm.update(step)

        assert nm.is_adolescent
        assert nm.phase == "adolescent"

    def test_adolescent_exit_on_max_duration(self):
        """Adolescent exits after max_duration steps."""
        config = self._make_config()
        nm = NeuromodulationSystem(config)

        # Force adolescent entry
        nm._adolescent_active = True
        nm._adolescent_start_step = 100
        nm._current_phase = "adolescent"

        # Run past max_duration (500)
        nm.update_external_signals(myelination_fraction=0.0)
        for step in range(601, 620):
            nm.update(step)

        assert not nm.is_adolescent
        assert nm.phase == "mature"

    def test_adolescent_exit_on_myelination(self):
        """Adolescent exits when myelination target reached."""
        config = self._make_config()
        nm = NeuromodulationSystem(config)

        nm._adolescent_active = True
        nm._adolescent_start_step = 100
        nm._current_phase = "adolescent"

        nm.update_external_signals(myelination_fraction=0.30)  # > 0.25 target
        nm.update(200)

        assert not nm.is_adolescent

    def test_adolescent_baselines_are_peak(self):
        """During adolescent phase, DA/NE should be at peak values."""
        config = self._make_config()
        nm = NeuromodulationSystem(config)

        nm._adolescent_active = True
        nm._current_phase = "adolescent"
        nm.update(200)

        assert nm.da == config.critical_period.adolescent_da
        assert nm.ne == config.critical_period.adolescent_ne

    def test_state_roundtrip(self):
        """Adolescent state survives get_state/set_state."""
        config = self._make_config()
        nm = NeuromodulationSystem(config)
        nm._adolescent_active = True
        nm._adolescent_start_step = 42
        nm._current_phase = "adolescent"
        nm._feature_stdp_peak = 0.5

        state = nm.get_state()
        nm2 = NeuromodulationSystem(config)
        nm2.set_state(state)

        assert nm2._adolescent_active == True
        assert nm2._adolescent_start_step == 42
        assert nm2._current_phase == "adolescent"
        assert nm2._feature_stdp_peak == 0.5


class TestSynapticPruning:
    def _make_synapse(self, n=100, init_weight=0.5):
        return SynapseGroup(
            n_pre=n, n_post=n, sparsity=0.1,
            init_weight=init_weight, plastic=True,
            stdp_params=STDPParams(w_min=0.0, w_max=1.0),
            rng=np.random.default_rng(42), name="test",
        )

    def test_prune_removes_weak_synapses(self):
        syn = self._make_synapse(init_weight=0.5)
        syn.init_adolescent_arrays()
        # Manually set some weights below threshold
        n_below = min(50, syn.weights.nnz)
        syn.weights.data[:n_below] = 0.02  # below threshold 0.05
        original_nnz = syn.nnz

        config = PruningConfig(weight_threshold=0.05, max_prune_fraction=0.1)
        pruned = syn.prune(config)

        assert pruned > 0
        assert syn.nnz < original_nnz

    def test_prune_respects_max_fraction(self):
        syn = self._make_synapse(init_weight=0.02)
        syn.init_adolescent_arrays()
        original_nnz = syn.nnz

        config = PruningConfig(weight_threshold=0.5, max_prune_fraction=0.05)
        pruned = syn.prune(config)

        assert pruned <= int(original_nnz * 0.05) + 1

    def test_prune_skips_myelinated(self):
        syn = self._make_synapse(init_weight=0.02)
        syn.init_adolescent_arrays()
        # Mark all as myelinated
        syn.myelinated[:] = True

        config = PruningConfig(weight_threshold=0.5)
        pruned = syn.prune(config)
        assert pruned == 0

    def test_prune_skips_identity(self):
        syn = self._make_synapse(init_weight=0.02)
        syn.init_adolescent_arrays()
        syn.identity[:] = True

        config = PruningConfig(weight_threshold=0.5)
        pruned = syn.prune(config)
        assert pruned == 0


class TestMyelination:
    def _make_synapse(self):
        syn = SynapseGroup(
            n_pre=50, n_post=50, sparsity=0.2,
            init_weight=0.7, plastic=True,
            stdp_params=STDPParams(w_min=0.01, w_max=1.0),
            rng=np.random.default_rng(42), name="test",
        )
        syn.init_adolescent_arrays()
        return syn

    def test_myelinate_stable_high_weight(self):
        syn = self._make_synapse()
        config = MyelinationConfig(stability_window=10, weight_threshold=0.5)
        # Set high stability
        syn.stability_counter[:] = 100
        n_myel = syn.myelinate(config)
        assert n_myel > 0
        assert syn.myelination_fraction > 0

    def test_myelinate_skip_low_weight(self):
        syn = self._make_synapse()
        config = MyelinationConfig(stability_window=10, weight_threshold=0.99)
        syn.stability_counter[:] = 100
        # Set all weights below threshold
        syn.weights.data[:] = 0.1
        n_myel = syn.myelinate(config)
        assert n_myel == 0

    def test_myelinate_skip_unstable(self):
        syn = self._make_synapse()
        config = MyelinationConfig(stability_window=100, weight_threshold=0.5)
        # stability_counter is 0 (freshly initialized)
        n_myel = syn.myelinate(config)
        assert n_myel == 0

    def test_plasticity_mask(self):
        syn = self._make_synapse()
        config = MyelinationConfig(plasticity_reduction=0.1)
        mask = syn.get_adolescent_plasticity_mask(config)
        # All normal initially
        assert np.allclose(mask, 1.0)

        # Mark some as myelinated
        syn.myelinated[:10] = True
        mask = syn.get_adolescent_plasticity_mask(config)
        assert np.allclose(mask[:10], 0.1)
        assert np.allclose(mask[10:], 1.0)

        # Mark some as identity
        syn.identity[:5] = True
        mask = syn.get_adolescent_plasticity_mask(config)
        assert np.allclose(mask[:5], 0.01)


class TestIdentityTagging:
    def test_tag_identity(self):
        syn = SynapseGroup(
            n_pre=50, n_post=50, sparsity=0.2,
            init_weight=0.7, plastic=True,
            rng=np.random.default_rng(42), name="test",
        )
        syn.init_adolescent_arrays()
        # Mark some as myelinated and survived pruning
        syn.myelinated[:20] = True
        syn.prune_survival_count[:20] = 5

        n_ident = syn.tag_identity(min_survival_rounds=3)
        assert n_ident == 20
        assert syn.identity_fraction > 0


class TestNeighborhoodConsolidation:
    def test_da_burst_rescues_traces(self):
        syn = SynapseGroup(
            n_pre=30, n_post=30, sparsity=0.3,
            init_weight=0.5, plastic=True,
            eligibility_config=EligibilityTraceConfig(),
            rng=np.random.default_rng(42), name="test",
        )
        # Set some eligibility traces
        syn.eligibility[:] = 0.01
        original = syn.eligibility.copy()

        # Create recent spike times
        current_time = 100.0
        spike_pre = np.full(30, 99.8, dtype=np.float32)  # very recent
        spike_post = np.full(30, 99.5, dtype=np.float32)

        config = NeighborhoodConsolidationConfig(
            da_burst_threshold=3.0,
            rescue_radius_ms=500.0,
            rescue_strength=0.5,
        )

        # DA below threshold — no rescue
        syn.apply_neighborhood_consolidation(2.0, config, current_time, spike_pre, spike_post)
        assert np.allclose(syn.eligibility, original)

        # DA above threshold — rescues nearby traces
        syn.apply_neighborhood_consolidation(4.0, config, current_time, spike_pre, spike_post)
        assert (syn.eligibility > original).any()


class TestInstinctPhaseScaling:
    def test_adolescent_amplifies_novelty(self):
        config = NeuromorphicConfig()
        instincts = OrientingInstincts(config)
        n = config.populations.sensory_cortex
        current = np.ones(n, dtype=np.float32)
        active = {"visual": (0, int(n * 0.4))}

        # Baseline gain (infant)
        instincts.set_phase("infant")
        gain_infant = instincts.compute_gain(current, active, step=10000)

        # Reset habituation
        instincts._habituation.clear()
        instincts._last_active_step.clear()

        # Adolescent gain (should be higher novelty)
        instincts.set_phase("adolescent")
        gain_adolescent = instincts.compute_gain(current, active, step=10000)

        # Adolescent novelty should be > infant
        vis_end = int(n * 0.4)
        assert gain_adolescent[:vis_end].mean() >= gain_infant[:vis_end].mean()

    def test_mature_dampens_novelty(self):
        config = NeuromorphicConfig()
        instincts = OrientingInstincts(config)
        n = config.populations.sensory_cortex
        current = np.ones(n, dtype=np.float32)
        active = {"visual": (0, int(n * 0.4))}

        # Baseline (infant = 1.0 scale)
        instincts.set_phase("infant")
        gain_infant = instincts.compute_gain(current, active, step=10000)

        instincts._habituation.clear()
        instincts._last_active_step.clear()

        # Mature gain (should be lower)
        instincts.set_phase("mature")
        gain_mature = instincts.compute_gain(current, active, step=10000)

        vis_end = int(n * 0.4)
        assert gain_mature[:vis_end].mean() <= gain_infant[:vis_end].mean()


class TestConvergenceTracking:
    def test_network_convergence_detection(self):
        """Network tracks convergence via STDP delta moving average."""
        config = NeuromorphicConfig(
            populations=NeuromorphicConfig().populations.__class__(
                brainstem=100, reflex_arc=50, sensory_cortex=200,
                motor_cortex=100, cerebellum=100, association_cortex=200,
                predictive_layer=100, working_memory=50,
            ),
        )
        net = NeuromorphicNetwork(config, seed=42)
        assert not net.is_converged

        # Run steps — without input, deltas should drop to zero
        for _ in range(200):
            net.step()

        # After many empty steps, convergence should be detected
        # (all deltas should be 0 since no spikes → no STDP)
        assert net.is_converged or net._convergence_stable_count > 0

    def test_reset_convergence(self):
        config = NeuromorphicConfig(
            populations=NeuromorphicConfig().populations.__class__(
                brainstem=50, reflex_arc=30, sensory_cortex=100,
                motor_cortex=50, cerebellum=50, association_cortex=100,
                predictive_layer=50, working_memory=30,
            ),
        )
        net = NeuromorphicNetwork(config, seed=42)
        net._convergence_stable_count = 100
        net.reset_convergence()
        assert net._convergence_stable_count == 0
        assert len(net._convergence_history) == 0


class TestAdolescentStateRoundtrip:
    def test_synapse_state_with_adolescent_arrays(self):
        rng = np.random.default_rng(42)
        syn = SynapseGroup(
            n_pre=50, n_post=50, sparsity=0.2,
            init_weight=0.5, plastic=True,
            rng=rng, name="test",
        )
        syn.init_adolescent_arrays()
        syn.stability_counter[:10] = 42
        syn.myelinated[:5] = True
        syn.identity[:3] = True
        syn.prune_survival_count[:8] = 7

        state = syn.get_state()

        # Create a new synapse with same shape and SAME nnz by using identical init
        # (set_state replaces the weight matrix so nnz will match)
        syn2 = SynapseGroup(
            n_pre=50, n_post=50, sparsity=0.2,
            init_weight=0.3, plastic=True,
            rng=np.random.default_rng(99), name="test",
        )
        syn2.init_adolescent_arrays()
        syn2.set_state(state)

        # After set_state, nnz matches the saved state
        assert syn2.weights.nnz == syn.weights.nnz
        # Adolescent arrays should be restored
        assert (syn2.stability_counter[:10] == 42).all()
        assert syn2.myelinated[:5].all()
        assert syn2.identity[:3].all()
        assert (syn2.prune_survival_count[:8] == 7).all()
