"""Tests for sparse synapse connections and STDP learning."""

import numpy as np
from scipy import sparse

from neuromorphic.config import RSTDPParams, STDPParams
from neuromorphic.synapses import SynapseGroup


class TestSynapseGroup:
    def test_sparse_construction(self):
        syn = SynapseGroup(
            n_pre=1000,
            n_post=500,
            sparsity=0.01,
            init_weight=0.3,
            rng=np.random.default_rng(42),
        )
        assert syn.weights.shape == (500, 1000)
        assert sparse.issparse(syn.weights)
        assert syn.nnz > 0
        # Sparsity should be approximately correct
        actual_sparsity = syn.nnz / (1000 * 500)
        assert actual_sparsity < 0.02  # within 2x

    def test_compute_current(self):
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.1,
            init_weight=0.5,
            rng=np.random.default_rng(42),
        )
        pre_spikes = np.zeros(100, dtype=bool)
        pre_spikes[:10] = True
        current = syn.compute_current(pre_spikes)
        assert current.shape == (50,)
        assert current.dtype == np.float32
        # Some post neurons should receive current
        assert current.sum() > 0

    def test_no_current_no_spikes(self):
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.1,
            init_weight=0.5,
            rng=np.random.default_rng(42),
        )
        pre_spikes = np.zeros(100, dtype=bool)
        current = syn.compute_current(pre_spikes)
        assert current.sum() == 0.0

    def test_non_plastic_frozen(self):
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.1,
            init_weight=0.5,
            plastic=False,
            rng=np.random.default_rng(42),
        )
        original_data = syn.weights.data.copy()

        # Try STDP update — should be no-op
        pre_spikes = np.zeros(100, dtype=bool)
        pre_spikes[:10] = True
        post_spikes = np.zeros(50, dtype=bool)
        post_spikes[:5] = True
        pre_times = np.full(100, 10.0, dtype=np.float32)
        post_times = np.full(50, 11.0, dtype=np.float32)

        syn.update_weights_stdp(pre_spikes, post_spikes, pre_times, post_times, 11.0)
        np.testing.assert_array_equal(syn.weights.data, original_data)

    def test_stdp_ltp(self):
        """Post fires after pre → weights should increase (LTP)."""
        stdp = STDPParams(a_plus=0.05, a_minus=0.06, w_min=0.0, w_max=1.0)
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.5,
            init_weight=0.3,
            plastic=True,
            stdp_params=stdp,
            rng=np.random.default_rng(42),
        )
        original_mean = syn.weights.data.mean()

        pre_spikes = np.zeros(100, dtype=bool)
        pre_spikes[:50] = True
        post_spikes = np.zeros(50, dtype=bool)
        post_spikes[:25] = True

        # Pre fires at t=10, post at t=12 (dt > 0 → LTP)
        pre_times = np.full(100, 10.0, dtype=np.float32)
        post_times = np.full(50, 12.0, dtype=np.float32)

        syn.update_weights_stdp(pre_spikes, post_spikes, pre_times, post_times, 12.0)

        # On average, weights should have increased (LTP)
        new_mean = syn.weights.data.mean()
        assert (
            new_mean > original_mean
        ), f"LTP should increase mean weight: {original_mean:.6f} → {new_mean:.6f}"

    def test_stdp_ltd(self):
        """Pre fires after post → weights should decrease (LTD)."""
        stdp = STDPParams(a_plus=0.05, a_minus=0.06, w_min=0.0, w_max=1.0)
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.5,
            init_weight=0.5,
            plastic=True,
            stdp_params=stdp,
            rng=np.random.default_rng(42),
        )
        original_mean = syn.weights.data.mean()

        pre_spikes = np.zeros(100, dtype=bool)
        pre_spikes[:50] = True
        post_spikes = np.zeros(50, dtype=bool)
        post_spikes[:25] = True

        # Post fires at t=10, pre at t=12 (dt < 0 → LTD)
        pre_times = np.full(100, 12.0, dtype=np.float32)
        post_times = np.full(50, 10.0, dtype=np.float32)

        syn.update_weights_stdp(pre_spikes, post_spikes, pre_times, post_times, 12.0)
        new_mean = syn.weights.data.mean()
        assert (
            new_mean < original_mean
        ), f"LTD should decrease mean weight: {original_mean:.6f} → {new_mean:.6f}"

    def test_weight_bounds(self):
        """Weights should stay within [w_min, w_max]."""
        stdp = STDPParams(a_plus=0.5, a_minus=0.5, w_min=0.0, w_max=1.0)
        syn = SynapseGroup(
            n_pre=50,
            n_post=50,
            sparsity=0.5,
            init_weight=0.9,
            plastic=True,
            stdp_params=stdp,
            rng=np.random.default_rng(42),
        )

        pre_spikes = np.ones(50, dtype=bool)
        post_spikes = np.ones(50, dtype=bool)
        pre_times = np.full(50, 10.0, dtype=np.float32)
        post_times = np.full(50, 10.5, dtype=np.float32)

        # Many LTP updates
        for _ in range(20):
            syn.update_weights_stdp(pre_spikes, post_spikes, pre_times, post_times, 10.5)
            post_times += 1.0
            pre_times += 1.0

        assert syn.weights.data.max() <= 1.0
        assert syn.weights.data.min() >= 0.0

    def test_rstdp_modulation(self):
        """R-STDP should scale weight changes by modulation signal."""
        rstdp = RSTDPParams(a_plus=0.05, a_minus=0.06, w_min=0.0, w_max=1.0)
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.5,
            init_weight=0.3,
            plastic=True,
            rstdp_params=rstdp,
            rng=np.random.default_rng(42),
        )

        pre_spikes = np.zeros(100, dtype=bool)
        pre_spikes[:50] = True
        post_spikes = np.zeros(50, dtype=bool)
        post_spikes[:25] = True
        pre_times = np.full(100, 10.0, dtype=np.float32)
        post_times = np.full(50, 12.0, dtype=np.float32)

        # High modulation (surprise) → larger weight change
        syn_copy_data = syn.weights.data.copy()
        syn.update_weights_rstdp(
            pre_spikes, post_spikes, pre_times, post_times, 12.0, modulation=3.0
        )
        high_mod_change = abs(syn.weights.data.mean() - syn_copy_data.mean())
        assert high_mod_change > 0

    def test_normalize_weights(self):
        syn = SynapseGroup(
            n_pre=50,
            n_post=50,
            sparsity=0.5,
            init_weight=0.5,
            plastic=True,
            rng=np.random.default_rng(42),
        )
        # Artificially inflate weights beyond w_max
        syn.weights = syn.weights.multiply(100.0).tocsr()
        syn.normalize_weights()
        # Hard bounds still enforced
        assert syn.weights.data.max() <= syn.stdp_params.w_max + 1e-5
        assert syn.weights.data.min() >= syn.stdp_params.w_min - 1e-5

    def test_synaptic_scaling_reduces_saturated_weights(self):
        """Repeated scaling should pull saturated weights toward the target."""
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.3,
            init_weight=0.99,
            plastic=True,
            rng=np.random.default_rng(42),
        )
        mean_before = syn.weights.data.mean()
        # Apply scaling multiple times (simulates many homeostasis intervals)
        for _ in range(100):
            syn.normalize_weights()
        mean_after = syn.weights.data.mean()
        # Saturated weights should decrease toward target
        assert mean_after < mean_before

    def test_synaptic_scaling_preserves_within_row_order(self):
        """Multiplicative scaling preserves relative weight ordering within each row."""
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.3,
            init_weight=0.5,
            plastic=True,
            rng=np.random.default_rng(42),
        )
        indptr = syn.weights.indptr
        data_before = syn.weights.data.copy()
        for _ in range(10):
            syn.normalize_weights()
        # Check per-row: multiplicative scaling preserves within-row ordering
        preserved = 0
        total = 0
        for i in range(syn.n_post):
            s, e = indptr[i], indptr[i + 1]
            if e - s < 2:
                continue
            order_b = np.argsort(data_before[s:e])
            order_a = np.argsort(syn.weights.data[s:e])
            preserved += (order_b == order_a).sum()
            total += e - s
        assert total > 0
        assert preserved / total > 0.95

    def test_state_serialization(self):
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.1,
            init_weight=0.4,
            rng=np.random.default_rng(42),
        )
        state = syn.get_state()
        assert "weights_data" in state
        assert "weights_indices" in state
        assert "weights_indptr" in state
        assert "shape" in state

        syn2 = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.0,
            init_weight=0.0,
            rng=np.random.default_rng(0),
        )
        syn2.set_state(state)
        np.testing.assert_array_equal(syn.weights.data, syn2.weights.data)

    def test_float32_storage(self):
        syn = SynapseGroup(
            n_pre=100,
            n_post=50,
            sparsity=0.1,
            init_weight=0.3,
            rng=np.random.default_rng(42),
        )
        assert syn.weights.dtype == np.float32

    def test_homeostasis_phase_dependent(self):
        """Claim 1e: higher plasticity_multiplier -> gentler homeostasis (H1 rule).

        During infant phase (high plasticity), homeostasis should produce
        smaller corrections so STDP can "win". During mature phase (low
        plasticity), homeostasis should be stronger to maintain stability.
        """

        # Create two identical synapse groups with saturated weights
        def make_syn():
            syn = SynapseGroup(
                n_pre=100,
                n_post=50,
                sparsity=0.3,
                init_weight=0.9,
                plastic=True,
                rng=np.random.default_rng(42),
            )
            return syn

        syn_infant = make_syn()
        syn_mature = make_syn()

        mean_before = syn_infant.weights.data.mean()

        # Infant: high plasticity_multiplier -> gentle correction (H1)
        syn_infant.normalize_weights(
            target_frac=0.5,
            base_rate=0.01,
            plasticity_multiplier=2.0,
            adolescent_bypass=False,
        )
        infant_correction = abs(syn_infant.weights.data.mean() - mean_before)

        # Mature: low plasticity_multiplier -> strong correction (H1)
        syn_mature.normalize_weights(
            target_frac=0.5,
            base_rate=0.01,
            plasticity_multiplier=0.5,
            adolescent_bypass=False,
        )
        mature_correction = abs(syn_mature.weights.data.mean() - mean_before)

        assert mature_correction > infant_correction, (
            f"Mature correction ({mature_correction:.6f}) should exceed "
            f"infant ({infant_correction:.6f}) due to H1 inverse scaling"
        )

    def test_homeostasis_adolescent_bypass(self):
        """When adolescent_bypass=True, base_rate is used directly (H2).

        This produces stronger correction than the default H1 path would
        at the same plasticity_multiplier, counteracting widened STDP.
        """

        def make_syn():
            return SynapseGroup(
                n_pre=100,
                n_post=50,
                sparsity=0.3,
                init_weight=0.9,
                plastic=True,
                rng=np.random.default_rng(42),
            )

        syn_h1 = make_syn()
        syn_h2 = make_syn()
        mean_before = syn_h1.weights.data.mean()

        # H1 path: inverse scaling with high plasticity (adolescent DA=2.5)
        syn_h1.normalize_weights(
            target_frac=0.5,
            base_rate=0.01,
            plasticity_multiplier=2.5,
            adolescent_bypass=False,
        )
        h1_correction = abs(syn_h1.weights.data.mean() - mean_before)

        # H2 path: bypass, use base_rate directly
        syn_h2.normalize_weights(
            target_frac=0.5,
            base_rate=0.01,
            plasticity_multiplier=2.5,
            adolescent_bypass=True,
        )
        h2_correction = abs(syn_h2.weights.data.mean() - mean_before)

        assert h2_correction > h1_correction, (
            f"H2 bypass ({h2_correction:.6f}) should produce stronger "
            f"correction than H1 ({h1_correction:.6f}) during adolescent"
        )
