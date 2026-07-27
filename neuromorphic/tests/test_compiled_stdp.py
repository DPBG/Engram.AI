"""Equivalence tests for the compiled STDP + eligibility-trace kernels.

Invariant 1 (CLAUDE.md §2): "Any change to those intervals MUST preserve
equivalence and be covered by an equivalence test."

This file verifies that:
  1. The compiled ``stdp_delta`` matches the NumPy reference to float32 precision.
  2. The compiled ``neuromod_decay_sparse`` and ``neuromod_decay_full`` match
     the NumPy reference to float32 precision.
  3. End-to-end: ``SynapseGroup.apply_neuromodulation_and_decay`` with the
     compiled path active produces the same weight changes as the NumPy-only
     path — confirming the compiled kernel is a correct drop-in replacement.

Tests skip when Numba is not installed or NEURO_COMPILED_STDP=0.
Run with the compiled extra: uv run --extra dev --extra compiled python -m pytest tests/test_compiled_stdp.py
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.equivalence import assert_kernel_equivalent, compiled_only_mark

import neuromorphic.compiled_kernels as _ck_mod
from neuromorphic.compiled_kernels import (
    COMPILED_STDP_ENABLED,
    neuromod_decay_full,
    neuromod_decay_sparse,
    stdp_delta,
)

# Applied at class level to tests that require the Numba path to be active.
# The NumPy-fallback tests (TestNumPyFallbackKernels) carry no such mark.
_COMPILED_ONLY = compiled_only_mark(
    COMPILED_STDP_ENABLED,
    reason="Numba not installed or NEURO_COMPILED_STDP=0 — compiled kernel tests skipped",
)

_RNG = np.random.default_rng(0)


# ── stdp_delta equivalence ─────────────────────────────────────────────────


@_COMPILED_ONLY
class TestCompiledStdpDelta:
    """Compiled stdp_delta output must match NumPy to float32 tolerance."""

    _A_PLUS = 0.012
    _A_MINUS = 0.010
    _TAU_PLUS = 20.0
    _TAU_MINUS = 20.0

    def _numpy_ref(self, dt: np.ndarray) -> np.ndarray:
        dw = np.empty(len(dt), dtype=np.float32)
        ltp = dt >= 0.0
        if ltp.any():
            dw[ltp] = np.float32(self._A_PLUS) * np.exp(-dt[ltp] / self._TAU_PLUS).astype(
                np.float32
            )
        ltd = ~ltp
        if ltd.any():
            dw[ltd] = np.float32(-self._A_MINUS) * np.exp(dt[ltd] / self._TAU_MINUS).astype(
                np.float32
            )
        return dw

    @pytest.mark.parametrize("n", [1, 100, 10_000])
    def test_ltp_only(self, n):
        dt = _RNG.uniform(1.0, 50.0, size=n).astype(np.float32)
        assert_kernel_equivalent(
            stdp_delta(dt, self._A_PLUS, self._TAU_PLUS, self._A_MINUS, self._TAU_MINUS),
            self._numpy_ref(dt),
            err_msg="LTP-only mismatch: compiled vs NumPy",
        )

    @pytest.mark.parametrize("n", [1, 100, 10_000])
    def test_ltd_only(self, n):
        dt = -_RNG.uniform(1.0, 50.0, size=n).astype(np.float32)
        assert_kernel_equivalent(
            stdp_delta(dt, self._A_PLUS, self._TAU_PLUS, self._A_MINUS, self._TAU_MINUS),
            self._numpy_ref(dt),
            err_msg="LTD-only mismatch: compiled vs NumPy",
        )

    def test_mixed_ltp_ltd(self):
        dt = np.array([-20.0, -5.0, 0.0, 5.0, 20.0], dtype=np.float32)
        assert_kernel_equivalent(
            stdp_delta(dt, self._A_PLUS, self._TAU_PLUS, self._A_MINUS, self._TAU_MINUS),
            self._numpy_ref(dt),
            err_msg="mixed LTP/LTD mismatch: compiled vs NumPy",
        )

    def test_sign_conventions(self):
        """LTP (dt >= 0) must be positive; LTD (dt < 0) must be negative."""
        dt = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        dw = stdp_delta(dt, self._A_PLUS, self._TAU_PLUS, self._A_MINUS, self._TAU_MINUS)
        assert dw[0] < 0.0, "LTD should produce negative delta"
        assert dw[1] >= 0.0, "Simultaneous should produce non-negative delta (Hebbian)"
        assert dw[2] > 0.0, "LTP should produce positive delta"


# ── neuromod_decay_sparse equivalence ─────────────────────────────────────


@_COMPILED_ONLY
class TestCompiledNeuromodDecaySparse:
    """Compiled neuromod_decay_sparse must match NumPy to float32 tolerance."""

    _NNZ = 5_000
    _W_MIN = np.float32(0.01)
    _W_MAX = np.float32(1.0)
    _PRUNE_THR = np.float32(1e-6)

    def _make_arrays(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        eligibility = (rng.random(self._NNZ).astype(np.float32) - 0.5) * 0.02
        data = rng.uniform(0.1, 0.9, self._NNZ).astype(np.float32)
        return eligibility, data

    def _numpy_ref(self, elig, data, idx, modulator, interval, d_per_step, mask=None):
        """Per-entry effective gain reference — matches the corrected kernel semantics."""
        elig_slice = elig[idx].copy()
        abs_e = np.abs(elig_slice)
        eff_gain = np.ones(len(idx), dtype=np.float32)
        step_pow = np.float32(d_per_step)
        e_check = abs_e * step_pow
        for _ in range(1, interval):
            alive_k = e_check > np.float32(self._PRUNE_THR)
            if not alive_k.any():
                break
            eff_gain[alive_k] += step_pow
            step_pow *= np.float32(d_per_step)
            e_check *= np.float32(d_per_step)
        dw = np.float32(modulator) * elig_slice * eff_gain
        if mask is not None:
            dw *= mask[idx]
        data[idx] = np.clip(data[idx] + dw, self._W_MIN, self._W_MAX)
        decay_total = np.float32(d_per_step**interval)
        elig_slice *= decay_total
        elig[idx] = elig_slice
        return np.abs(elig_slice) > self._PRUNE_THR

    @pytest.mark.parametrize("interval", [1, 3, 10])
    def test_no_mask(self, interval):
        d = 0.999
        modulator = 0.5
        idx = np.arange(self._NNZ // 2, dtype=np.int32)

        elig_np, data_np = self._make_arrays(seed=interval)
        elig_c, data_c = elig_np.copy(), data_np.copy()

        alive_ref = self._numpy_ref(elig_np, data_np, idx, modulator, interval, d)
        alive_c = neuromod_decay_sparse(
            elig_c,
            data_c,
            idx,
            modulator,
            interval,
            d,
            self._W_MIN,
            self._W_MAX,
            self._PRUNE_THR,
        )

        assert_kernel_equivalent(
            data_c[idx],
            data_np[idx],
            err_msg=f"weight mismatch (no mask, interval={interval})",
        )
        assert_kernel_equivalent(
            elig_c[idx],
            elig_np[idx],
            err_msg=f"eligibility mismatch (no mask, interval={interval})",
        )
        np.testing.assert_array_equal(
            alive_c, alive_ref, err_msg=f"alive mask mismatch (interval={interval})"
        )

    @pytest.mark.parametrize("interval", [1, 3])
    def test_with_mask(self, interval):
        mask = np.random.default_rng(99).uniform(0.01, 1.0, self._NNZ).astype(np.float32)
        d = 0.999
        modulator = 0.3
        idx = np.arange(self._NNZ // 2, dtype=np.int32)

        elig_np, data_np = self._make_arrays(seed=7)
        elig_c, data_c = elig_np.copy(), data_np.copy()

        self._numpy_ref(elig_np, data_np, idx, modulator, interval, d, mask=mask)
        neuromod_decay_sparse(
            elig_c,
            data_c,
            idx,
            modulator,
            interval,
            d,
            self._W_MIN,
            self._W_MAX,
            self._PRUNE_THR,
            plasticity_mask=mask,
        )

        assert_kernel_equivalent(
            data_c[idx],
            data_np[idx],
            err_msg=f"weight mismatch (masked, interval={interval})",
        )
        assert_kernel_equivalent(
            elig_c[idx],
            elig_np[idx],
            err_msg=f"eligibility mismatch (masked, interval={interval})",
        )

    def test_weight_clipping_enforced(self):
        """Compiled kernel must honour w_min / w_max the same as NumPy clip."""
        elig = np.full(100, 1.0, dtype=np.float32)  # large traces → big dw
        data = np.full(100, 0.95, dtype=np.float32)  # weights near w_max
        idx = np.arange(100, dtype=np.int32)
        # After update data would exceed w_max=1.0 without clipping
        neuromod_decay_sparse(
            elig.copy(),
            data,
            idx,
            5.0,
            1,
            0.999,
            np.float32(0.01),
            np.float32(1.0),
            np.float32(1e-6),
        )
        assert data.max() <= 1.0 + 1e-6, "w_max violated by compiled kernel"
        assert data.min() >= 0.01 - 1e-6, "w_min violated by compiled kernel"


# ── neuromod_decay_full equivalence ───────────────────────────────────────


@_COMPILED_ONLY
class TestCompiledNeuromodDecayFull:
    """Compiled neuromod_decay_full must match NumPy to float32 tolerance."""

    _NNZ = 5_000
    _W_MIN = np.float32(0.01)
    _W_MAX = np.float32(1.0)

    def _make_arrays(self, seed: int = 1):
        rng = np.random.default_rng(seed)
        elig = (rng.random(self._NNZ).astype(np.float32) - 0.5) * 0.02
        data = rng.uniform(0.1, 0.9, self._NNZ).astype(np.float32)
        return elig, data

    @pytest.mark.parametrize("interval", [1, 3, 10])
    def test_no_mask(self, interval):
        d = 0.999
        decay = np.float32(d**interval)
        interval_gain = float((1.0 - d**interval) / (1.0 - d))
        modulator = 0.4

        elig_np, data_np = self._make_arrays(seed=interval)
        elig_c, data_c = elig_np.copy(), data_np.copy()

        dw = np.float32(modulator) * elig_np * np.float32(interval_gain)
        data_np[:] = np.clip(data_np + dw, self._W_MIN, self._W_MAX)
        elig_np *= decay

        neuromod_decay_full(
            elig_c, data_c, modulator, interval_gain, decay, self._W_MIN, self._W_MAX
        )

        assert_kernel_equivalent(
            data_c,
            data_np,
            err_msg=f"full-array weight mismatch (interval={interval})",
        )
        assert_kernel_equivalent(
            elig_c,
            elig_np,
            err_msg=f"full-array eligibility mismatch (interval={interval})",
        )

    def test_with_mask(self):
        mask = np.random.default_rng(5).uniform(0.01, 1.0, self._NNZ).astype(np.float32)
        d, modulator, interval = 0.999, 0.6, 3
        decay = np.float32(d**interval)
        interval_gain = float((1.0 - d**interval) / (1.0 - d))

        elig_np, data_np = self._make_arrays()
        elig_c, data_c = elig_np.copy(), data_np.copy()

        dw = np.float32(modulator) * elig_np * np.float32(interval_gain) * mask
        data_np[:] = np.clip(data_np + dw, self._W_MIN, self._W_MAX)
        elig_np *= decay

        neuromod_decay_full(
            elig_c,
            data_c,
            modulator,
            interval_gain,
            decay,
            self._W_MIN,
            self._W_MAX,
            plasticity_mask=mask,
        )

        assert_kernel_equivalent(data_c, data_np)
        assert_kernel_equivalent(elig_c, elig_np)


# ── End-to-end SynapseGroup integration ───────────────────────────────────


@_COMPILED_ONLY
class TestEndToEndCompiledEquivalence:
    """SynapseGroup.apply_neuromodulation_and_decay compiled == NumPy.

    Invariant 1 enforcement: the compiled path must produce weight changes
    within float32 tolerance of the every-step NumPy reference.
    """

    def _make_sg(self, rng_seed: int = 7):
        from neuromorphic.config import EligibilityTraceConfig
        from neuromorphic.synapses import SynapseGroup

        sg = SynapseGroup(
            n_pre=100,
            n_post=100,
            sparsity=0.1,
            init_weight=0.5,
            plastic=True,
            rng=np.random.default_rng(rng_seed),
            eligibility_config=EligibilityTraceConfig(),
        )
        elig_seed = np.random.default_rng(rng_seed + 1)
        sg.eligibility[:] = elig_seed.uniform(-0.01, 0.01, sg.nnz).astype(np.float32)
        sg._elig_active = np.arange(sg.nnz, dtype=np.int32)
        return sg

    @pytest.mark.parametrize("interval", [1, 3])
    @pytest.mark.parametrize("sparse_path", [True, False], ids=["sparse", "full-array"])
    def test_compiled_matches_numpy_per_step(self, monkeypatch, interval, sparse_path):
        """Compiled batched call == interval × every-step NumPy call.

        Runs the batched compiled path and the per-step NumPy path on
        identical initial state, then asserts weight-change agreement.
        """
        sg_c = self._make_sg()
        sg_np = self._make_sg()

        if not sparse_path:
            sg_c._elig_active = None
            sg_np._elig_active = None

        # ── Compiled path (batched) ──
        w_before = sg_c.weights.data.copy()
        sg_c.apply_neuromodulation_and_decay(0.5, interval=interval)
        dw_compiled = sg_c.weights.data - w_before

        # ── NumPy path (per-step) ── patch compiled_kernels so dispatch falls back
        monkeypatch.setattr(_ck_mod, "COMPILED_STDP_ENABLED", False)
        w_before = sg_np.weights.data.copy()
        for _ in range(interval):
            sg_np.apply_neuromodulation_and_decay(0.5, interval=1)
        dw_numpy = sg_np.weights.data - w_before

        assert_kernel_equivalent(
            dw_compiled,
            dw_numpy,
            rtol=1e-4,
            err_msg=(
                f"compiled vs NumPy weight-change mismatch "
                f"(interval={interval}, sparse_path={sparse_path})"
            ),
        )

    def test_stdp_update_compiled_matches_numpy(self, monkeypatch):
        """STDP delta from compiled kernel matches NumPy for non-BCM synapse groups."""
        from neuromorphic.config import EligibilityTraceConfig
        from neuromorphic.synapses import SynapseGroup

        def make():
            sg = SynapseGroup(
                n_pre=50,
                n_post=50,
                sparsity=0.2,
                init_weight=0.5,
                plastic=True,
                rng=np.random.default_rng(3),
                eligibility_config=EligibilityTraceConfig(),
            )
            return sg

        pre_sp = np.ones(50, dtype=bool)
        post_sp = np.ones(50, dtype=bool)
        pre_t = np.full(50, 1.0, dtype=np.float32)
        post_t = np.full(50, 3.0, dtype=np.float32)  # post after pre → LTP

        sg_c = make()
        sg_c.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=3.0)
        elig_compiled = sg_c.eligibility.copy()

        monkeypatch.setattr(_ck_mod, "COMPILED_STDP_ENABLED", False)
        sg_np = make()
        sg_np.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=3.0)
        elig_numpy = sg_np.eligibility.copy()

        assert_kernel_equivalent(
            elig_compiled,
            elig_numpy,
            err_msg="STDP eligibility trace mismatch: compiled vs NumPy",
        )


# ── NumPy fallback coverage ────────────────────────────────────────────────
# These tests carry no skip mark — they run regardless of Numba availability
# and explicitly force the NumPy path by patching compiled_kernels directly.


class TestNumPyFallbackKernels:
    """Public API of compiled_kernels.py exercises its NumPy fallback correctly.

    Patches _ck_mod.COMPILED_STDP_ENABLED = False so the dispatch inside each
    public function takes the NumPy branch regardless of Numba presence.
    """

    def test_stdp_delta_fallback_sign_conventions(self, monkeypatch):
        monkeypatch.setattr(_ck_mod, "COMPILED_STDP_ENABLED", False)
        dt = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        dw = stdp_delta(dt, 0.012, 20.0, 0.010, 20.0)
        assert dw[0] < 0.0, "LTD should be negative"
        assert dw[1] >= 0.0, "dt=0 should be non-negative (LTP boundary)"
        assert dw[2] > 0.0, "LTP should be positive"

    def test_stdp_delta_fallback_values(self, monkeypatch):
        monkeypatch.setattr(_ck_mod, "COMPILED_STDP_ENABLED", False)
        dt = np.array([10.0], dtype=np.float32)
        dw = stdp_delta(dt, 0.012, 20.0, 0.010, 20.0)
        expected = np.float32(0.012) * np.exp(np.float32(-10.0 / 20.0))
        np.testing.assert_allclose(dw[0], expected, rtol=1e-5)

    def test_neuromod_decay_sparse_fallback(self, monkeypatch):
        monkeypatch.setattr(_ck_mod, "COMPILED_STDP_ENABLED", False)
        elig = np.array([0.05, -0.03, 0.0], dtype=np.float32)
        data = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        idx = np.array([0, 1], dtype=np.int32)
        alive = neuromod_decay_sparse(
            elig,
            data,
            idx,
            1.0,
            1,
            np.float32(0.999),
            np.float32(0.01),
            np.float32(1.0),
            np.float32(1e-6),
        )
        assert alive.dtype == bool
        assert len(alive) == 2
        # large traces survive decay
        assert alive[0] and alive[1]
        # weights should have shifted
        assert data[0] != 0.5

    def test_neuromod_decay_full_fallback(self, monkeypatch):
        monkeypatch.setattr(_ck_mod, "COMPILED_STDP_ENABLED", False)
        elig = np.array([0.05, -0.03], dtype=np.float32)
        data = np.array([0.5, 0.6], dtype=np.float32)
        data_before = data.copy()
        neuromod_decay_full(
            elig, data, 1.0, 1.0, np.float32(0.999), np.float32(0.01), np.float32(1.0)
        )
        assert not np.array_equal(data, data_before), "weights must change"
        assert data.min() >= 0.01 - 1e-6
        assert data.max() <= 1.0 + 1e-6

    def test_neuromod_decay_sparse_fallback_with_mask(self, monkeypatch):
        monkeypatch.setattr(_ck_mod, "COMPILED_STDP_ENABLED", False)
        elig_orig = np.array([0.05, -0.03, 0.0], dtype=np.float32)
        data = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        data_no_mask = data.copy()
        idx = np.array([0, 1], dtype=np.int32)
        mask = np.array([2.0, 0.5, 1.0], dtype=np.float32)  # per-synapse scale
        # Both calls receive independent copies of elig so neither mutates the
        # other's baseline (neuromod_decay_sparse writes back eligibility[idx]).
        neuromod_decay_sparse(
            elig_orig.copy(),
            data_no_mask,
            idx,
            1.0,
            1,
            np.float32(0.999),
            np.float32(0.01),
            np.float32(1.0),
            np.float32(1e-6),
        )
        neuromod_decay_sparse(
            elig_orig.copy(),
            data,
            idx,
            1.0,
            1,
            np.float32(0.999),
            np.float32(0.01),
            np.float32(1.0),
            np.float32(1e-6),
            plasticity_mask=mask,
        )
        # mask=2.0 on idx[0] should amplify the weight update; 0.5 attenuates idx[1]
        delta_no_mask_0 = data_no_mask[0] - 0.5
        delta_masked_0 = data[0] - 0.5
        assert (
            abs(delta_masked_0) > abs(delta_no_mask_0) - 1e-7
        ), "mask=2.0 must amplify the weight change on idx[0]"
        delta_no_mask_1 = data_no_mask[1] - 0.6
        delta_masked_1 = data[1] - 0.6
        assert (
            abs(delta_masked_1) < abs(delta_no_mask_1) + 1e-7
        ), "mask=0.5 must attenuate the weight change on idx[1]"
        # idx[2] is not in active set — must be untouched
        assert data[2] == 0.7

    def test_neuromod_decay_full_fallback_with_mask(self, monkeypatch):
        monkeypatch.setattr(_ck_mod, "COMPILED_STDP_ENABLED", False)
        elig = np.array([0.05, -0.03], dtype=np.float32)
        data = np.array([0.5, 0.6], dtype=np.float32)
        data_zero_mask = data.copy()
        mask_zero = np.zeros(2, dtype=np.float32)
        neuromod_decay_full(
            elig.copy(),
            data_zero_mask,
            1.0,
            1.0,
            np.float32(0.999),
            np.float32(0.01),
            np.float32(1.0),
            plasticity_mask=mask_zero,
        )
        # zero mask means dw is zeroed → weights stay at their initial values
        np.testing.assert_array_equal(
            data_zero_mask,
            np.array([0.5, 0.6], dtype=np.float32),
            err_msg="zero plasticity_mask must produce no weight change",
        )
        data_full_mask = data.copy()
        mask_double = np.full(2, 2.0, dtype=np.float32)
        neuromod_decay_full(
            elig.copy(),
            data_full_mask,
            1.0,
            1.0,
            np.float32(0.999),
            np.float32(0.01),
            np.float32(1.0),
            plasticity_mask=mask_double,
        )
        data_no_mask = data.copy()
        neuromod_decay_full(
            elig.copy(),
            data_no_mask,
            1.0,
            1.0,
            np.float32(0.999),
            np.float32(0.01),
            np.float32(1.0),
        )
        np.testing.assert_allclose(
            np.abs(data_full_mask - np.array([0.5, 0.6], dtype=np.float32)),
            np.abs(data_no_mask - np.array([0.5, 0.6], dtype=np.float32)) * 2.0,
            rtol=1e-5,
            err_msg="mask=2.0 must double the weight delta vs no mask",
        )
