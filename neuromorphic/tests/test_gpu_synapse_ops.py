"""Tests for the experimental GPU synapse-op backend (issue #135 spike).

Covers the acceptance criteria from the issue:
1. Functional equivalence with the existing CPU implementation
   (SynapseGroup.compute_current).
2. (Performance is measured separately — see scripts/benchmark_gpu_synapse.py
   and docs/GPU-SYNAPSE-BACKEND-FEASIBILITY.md — not asserted in unit tests,
   since CI hardware has no GPU and timing assertions would be flaky.)
3. CPU fallback works without GPU hardware, and without jax installed at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from neuromorphic.gpu_synapse_ops import (
    _compute_current_numpy,
    compute_current_jax,
    gpu_backend_available,
)
from neuromorphic.synapses import SynapseGroup

pytest.importorskip("jax", reason="jax not installed (optional 'gpu' extra)")


def _random_synapse_group(n_pre, n_post, sparsity, seed=0) -> SynapseGroup:
    rng = np.random.default_rng(seed)
    return SynapseGroup(
        n_pre=n_pre, n_post=n_post, sparsity=sparsity, init_weight=0.3, rng=rng,
    )


class TestFunctionalEquivalence:
    """Criterion 1: spike branch shows functional equivalence with the CPU path."""

    @pytest.mark.parametrize(
        "n_pre,n_post,sparsity,fire_rate",
        [
            (200, 200, 0.05, 0.05),
            (1000, 500, 0.02, 0.1),
            (2000, 2000, 0.01, 0.03),
        ],
    )
    def test_matches_cpu_compute_current(self, n_pre, n_post, sparsity, fire_rate):
        group = _random_synapse_group(n_pre, n_post, sparsity)
        rng = np.random.default_rng(1)
        pre_spikes = rng.random(n_pre) < fire_rate

        expected = group.compute_current(pre_spikes)
        actual = compute_current_jax(group.weights, pre_spikes)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual.dtype == np.float32

    def test_matches_cpu_with_ei_sign(self):
        group = _random_synapse_group(500, 400, 0.03)
        rng = np.random.default_rng(2)
        pre_spikes = rng.random(500) < 0.1
        sign = rng.choice([-1.0, 1.0], size=500).astype(np.float32)

        expected = group.compute_current(pre_spikes, pre_output_sign=sign)
        actual = compute_current_jax(group.weights, pre_spikes, pre_output_sign=sign)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_no_spikes_returns_zeros(self):
        group = _random_synapse_group(300, 200, 0.05)
        pre_spikes = np.zeros(300, dtype=bool)

        actual = compute_current_jax(group.weights, pre_spikes)

        assert actual.shape == (200,)
        assert np.all(actual == 0.0)

    def test_empty_weights_returns_zeros(self):
        empty = sparse.csr_matrix((50, 30), dtype=np.float32)
        pre_spikes = np.ones(30, dtype=bool)

        actual = compute_current_jax(empty, pre_spikes)

        assert actual.shape == (50,)
        assert np.all(actual == 0.0)

    def test_repeated_calls_reuse_jit_cache_and_stay_correct(self):
        """The per-shape JIT cache must not leak stale compiled state across
        different weight *values* at the same (n_post, n_pre) shape."""
        group = _random_synapse_group(300, 300, 0.05)
        rng = np.random.default_rng(3)
        pre_spikes = rng.random(300) < 0.1

        first = compute_current_jax(group.weights, pre_spikes)

        # Mutate weights in-place (same sparsity structure/shape) and re-run —
        # the cached jit fn is keyed by shape, not by data, so it must still
        # reflect the new weight values.
        group.weights.data *= 2.0
        second = compute_current_jax(group.weights, pre_spikes)

        assert not np.allclose(first, second)
        np.testing.assert_allclose(second, group.compute_current(pre_spikes), rtol=1e-5, atol=1e-6)


class TestCPUFallback:
    """Criterion 3: CPU fallback operates successfully without GPU hardware."""

    def test_jax_runs_on_cpu_device_in_this_environment(self):
        """This test environment has no GPU/TPU — confirms jax itself is
        exercising its CPU backend, not silently skipping computation."""
        import jax
        assert all(d.platform == "cpu" for d in jax.devices())

    def test_gpu_backend_available_true_when_jax_installed(self):
        assert gpu_backend_available() is True

    def test_pure_numpy_fallback_matches_cpu_compute_current(self):
        """The no-jax-at-all fallback path (_compute_current_numpy) must also
        stay correct, independent of whether jax happens to be installed."""
        group = _random_synapse_group(400, 300, 0.04)
        rng = np.random.default_rng(4)
        pre_spikes = rng.random(400) < 0.08

        expected = group.compute_current(pre_spikes)
        actual = _compute_current_numpy(group.weights, pre_spikes)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_compute_current_jax_uses_numpy_fallback_when_backend_unavailable(self, monkeypatch):
        """Simulates jax not being installed — compute_current_jax must still
        return a correct result via the numpy fallback, not raise."""
        import neuromorphic.gpu_synapse_ops as gpu_ops

        monkeypatch.setattr(gpu_ops, "gpu_backend_available", lambda: False)

        group = _random_synapse_group(250, 200, 0.05)
        rng = np.random.default_rng(5)
        pre_spikes = rng.random(250) < 0.1

        expected = group.compute_current(pre_spikes)
        actual = gpu_ops.compute_current_jax(group.weights, pre_spikes)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
