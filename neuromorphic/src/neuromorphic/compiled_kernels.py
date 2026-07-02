"""Compiled (Numba-JIT) kernels for the STDP + eligibility-trace hot path.

When Numba is available and ``NEURO_COMPILED_STDP`` is not set to ``0``, this
module exposes JIT-compiled versions of the two most expensive operations in
the learning pipeline:

  1. ``stdp_delta``             — fused LTP/LTD kernel (single pass, no temps)
  2. ``neuromod_decay_sparse``  — fused neuromod + decay + clip + prune over the
                                  active sparse eligibility set
  3. ``neuromod_decay_full``    — same, but operating on the full array (used
                                  when active-set tracking is abandoned at >80%)

When Numba is unavailable (or ``NEURO_COMPILED_STDP=0``), the same public
functions fall back to NumPy-equivalent implementations so ``synapses.py``
needs no branching of its own.

Performance note (Invariant 1):
  Numba JIT-compiles lazily on the first call with ~100–300 ms overhead per
  signature.  ``cache=True`` persists compiled artifacts to ``__pycache__``
  so subsequent process starts reuse the cached binary.  ``fastmath=True``
  lets LLVM reassociate floating-point operations; the resulting values remain
  within float32 rounding of the NumPy reference (verified by the equivalence
  tests in ``test_compiled_stdp.py``).

Feature flag:
  ``NEURO_COMPILED_STDP=0``  — disable compiled kernels (use NumPy fallback)
  ``NEURO_COMPILED_STDP=1``  — enable when Numba is importable (default)
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ── Availability check ─────────────────────────────────────────────────────

NUMBA_AVAILABLE: bool = False
try:
    import numba  # type: ignore[import-untyped]  # noqa: F401
    NUMBA_AVAILABLE = True
except ImportError:
    pass

_env = os.environ.get("NEURO_COMPILED_STDP", "1").lower().strip()
COMPILED_STDP_ENABLED: bool = NUMBA_AVAILABLE and _env not in ("0", "false", "no")

if COMPILED_STDP_ENABLED:
    logger.debug("neuromorphic: Numba compiled STDP kernels active (cache=True)")
else:
    logger.debug(
        "neuromorphic: Numba compiled STDP kernels disabled "
        "(NUMBA_AVAILABLE=%s, NEURO_COMPILED_STDP=%r)",
        NUMBA_AVAILABLE,
        _env,
    )


# ── Numba JIT kernels (only defined when Numba is available) ───────────────

if COMPILED_STDP_ENABLED:
    import numba as _nb  # type: ignore[import-untyped]

    @_nb.njit(cache=True, fastmath=True)
    def _stdp_delta_njit(
        dt_spike: np.ndarray,
        a_plus: float,
        tau_plus: float,
        a_minus: float,
        tau_minus: float,
    ) -> np.ndarray:
        """Fused LTP/LTD delta: single pass, no ltp_mask/ltd_mask temp arrays."""
        n = len(dt_spike)
        dw = np.empty(n, np.float32)
        for i in range(n):
            dt = dt_spike[i]
            if dt >= 0.0:
                dw[i] = np.float32(a_plus) * np.exp(np.float32(-dt / tau_plus))
            else:
                dw[i] = np.float32(-a_minus) * np.exp(np.float32(dt / tau_minus))
        return dw

    @_nb.njit(cache=True, fastmath=True)
    def _neuromod_decay_sparse_njit(
        eligibility: np.ndarray,
        data: np.ndarray,
        idx: np.ndarray,
        modulator: float,
        interval: int,
        d_per_step: float,
        w_min: float,
        w_max: float,
        prune_threshold: float,
    ) -> np.ndarray:
        """Fused neuromod + decay + clip over sparse active set; returns alive mask.

        Uses per-entry effective gain so that a trace which decays below
        prune_threshold mid-interval only accumulates weight change for the steps
        it would have been alive — matching the per-step reference exactly.
        """
        n = len(idx)
        alive = np.empty(n, _nb.boolean)
        mod = np.float32(modulator)
        d = np.float32(d_per_step)
        pt = np.float32(prune_threshold)
        # Full-interval decay applied to eligibility.
        decay_total = np.float32(1.0)
        for _ in range(interval):
            decay_total *= d
        for i in range(n):
            k = idx[i]
            e = eligibility[k]
            # Step 0 always contributes; step j (j>0) contributes only if
            # |e0 * d^j| > pt (entry was still alive at end of step j-1).
            e_abs = e if e >= np.float32(0.0) else -e
            step_pow = np.float32(1.0)
            eff_gain = np.float32(0.0)
            for j in range(interval):
                eff_gain += step_pow
                e_abs *= d
                step_pow *= d
                if e_abs <= pt:
                    break  # pruned after step j; remaining steps don't contribute
            dw = mod * e * eff_gain
            w = data[k] + dw
            if w < w_min:
                w = w_min
            elif w > w_max:
                w = w_max
            data[k] = np.float32(w)
            e *= decay_total
            eligibility[k] = e
            alive[i] = e > pt or e < -pt
        return alive

    @_nb.njit(cache=True, fastmath=True)
    def _neuromod_decay_sparse_masked_njit(
        eligibility: np.ndarray,
        data: np.ndarray,
        idx: np.ndarray,
        modulator: float,
        interval: int,
        d_per_step: float,
        w_min: float,
        w_max: float,
        plasticity_mask: np.ndarray,
        prune_threshold: float,
    ) -> np.ndarray:
        """Fused sparse neuromod + decay + clip with per-synapse plasticity mask."""
        n = len(idx)
        alive = np.empty(n, _nb.boolean)
        mod = np.float32(modulator)
        d = np.float32(d_per_step)
        pt = np.float32(prune_threshold)
        decay_total = np.float32(1.0)
        for _ in range(interval):
            decay_total *= d
        for i in range(n):
            k = idx[i]
            e = eligibility[k]
            e_abs = e if e >= np.float32(0.0) else -e
            step_pow = np.float32(1.0)
            eff_gain = np.float32(0.0)
            for j in range(interval):
                eff_gain += step_pow
                e_abs *= d
                step_pow *= d
                if e_abs <= pt:
                    break
            dw = mod * e * eff_gain * plasticity_mask[k]
            w = data[k] + dw
            if w < w_min:
                w = w_min
            elif w > w_max:
                w = w_max
            data[k] = np.float32(w)
            e *= decay_total
            eligibility[k] = e
            alive[i] = e > pt or e < -pt
        return alive

    @_nb.njit(cache=True, fastmath=True)
    def _neuromod_decay_full_njit(
        eligibility: np.ndarray,
        data: np.ndarray,
        modulator: float,
        interval_gain: float,
        decay: float,
        w_min: float,
        w_max: float,
    ) -> None:
        """Fused neuromod + decay + clip over the complete eligibility array."""
        mod_gain = np.float32(modulator * interval_gain)
        d = np.float32(decay)
        for i in range(len(eligibility)):
            dw = mod_gain * eligibility[i]
            w = data[i] + dw
            if w < w_min:
                w = w_min
            elif w > w_max:
                w = w_max
            data[i] = np.float32(w)
            eligibility[i] *= d

    @_nb.njit(cache=True, fastmath=True)
    def _neuromod_decay_full_masked_njit(
        eligibility: np.ndarray,
        data: np.ndarray,
        modulator: float,
        interval_gain: float,
        decay: float,
        w_min: float,
        w_max: float,
        plasticity_mask: np.ndarray,
    ) -> None:
        """Full-array fused neuromod + decay + clip with per-synapse plasticity mask."""
        mod_gain = np.float32(modulator * interval_gain)
        d = np.float32(decay)
        for i in range(len(eligibility)):
            dw = mod_gain * eligibility[i] * plasticity_mask[i]
            w = data[i] + dw
            if w < w_min:
                w = w_min
            elif w > w_max:
                w = w_max
            data[i] = np.float32(w)
            eligibility[i] *= d


# ── Public interface ───────────────────────────────────────────────────────
# Identical signature whether compiled (Numba) or fallback (NumPy).
# synapses.py calls these unconditionally — no if/else at the call site.


def stdp_delta(
    dt_spike: np.ndarray,
    a_plus: float,
    tau_plus: float,
    a_minus: float,
    tau_minus: float,
) -> np.ndarray:
    """Compute STDP weight delta for a batch of spike-time differences.

    Args:
        dt_spike:  float32 array of ``post_time - pre_time`` values.
        a_plus:    LTP amplitude (scalar; BCM-scaled path is handled by caller).
        tau_plus:  LTP time constant (ms).
        a_minus:   LTD amplitude.
        tau_minus: LTD time constant (ms).

    Returns:
        float32 dw array, same length as dt_spike.
        Positive (LTP) for dt >= 0; negative (LTD) for dt < 0.
    """
    if COMPILED_STDP_ENABLED:
        return _stdp_delta_njit(
            dt_spike,
            float(a_plus),
            float(tau_plus),
            float(a_minus),
            float(tau_minus),
        )
    # NumPy fallback — equivalent to the original synapses.py implementation
    dw = np.empty(len(dt_spike), dtype=np.float32)
    ltp = dt_spike >= 0.0
    if ltp.any():
        dw[ltp] = np.float32(a_plus) * np.exp(-dt_spike[ltp] / tau_plus).astype(np.float32)
    ltd = ~ltp
    if ltd.any():
        dw[ltd] = np.float32(-a_minus) * np.exp(dt_spike[ltd] / tau_minus).astype(np.float32)
    return dw


def neuromod_decay_sparse(
    eligibility: np.ndarray,
    data: np.ndarray,
    idx: np.ndarray,
    modulator: float,
    interval: int,
    d_per_step: float,
    w_min: float,
    w_max: float,
    prune_threshold: float,
    plasticity_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply neuromodulation + eligibility decay in-place on the active sparse set.

    Mutates ``data[idx]`` and ``eligibility[idx]`` in-place.

    Args:
        eligibility:      Full nnz eligibility array.
        data:             CSR weights.data (full nnz).
        idx:              int32 active-set indices (subset of [0, nnz)).
        modulator:        Neuromodulatory signal (averaged over interval).
        interval:         Number of logical steps being batched.
        d_per_step:       Per-step trace decay (``trace_decay`` from config).
        w_min, w_max:     Weight bounds for hard clipping.
        prune_threshold:  Entries with |e| <= this after decay are considered dead.
        plasticity_mask:  Optional float32 per-synapse scaling (myelination/identity).

    Returns:
        Boolean array of length ``len(idx)`` — True for entries still alive
        after decay.  Caller uses this to prune ``_elig_active``.
    """
    if COMPILED_STDP_ENABLED:
        if plasticity_mask is not None:
            return _neuromod_decay_sparse_masked_njit(
                eligibility,
                data,
                idx,
                float(modulator),
                int(interval),
                float(d_per_step),
                float(w_min),
                float(w_max),
                plasticity_mask,
                float(prune_threshold),
            )
        return _neuromod_decay_sparse_njit(
            eligibility,
            data,
            idx,
            float(modulator),
            int(interval),
            float(d_per_step),
            float(w_min),
            float(w_max),
            float(prune_threshold),
        )
    # NumPy fallback — per-entry effective gain accounting for mid-interval pruning.
    # Step 0 always contributes; step j (j>0) contributes only if the trace was
    # still alive at the end of step j-1 (|e0 * d^j| > prune_threshold).
    elig_slice = eligibility[idx]
    abs_e = np.abs(elig_slice)
    eff_gain = np.ones(len(idx), dtype=np.float32)
    step_pow = np.float32(d_per_step)
    e_check = abs_e * step_pow  # |e0 * d^1| — alive check after step 0
    for _ in range(1, interval):
        contributes = e_check > np.float32(prune_threshold)
        if not contributes.any():
            break
        eff_gain[contributes] += step_pow
        step_pow *= np.float32(d_per_step)
        e_check *= np.float32(d_per_step)
    dw = np.float32(modulator) * elig_slice * eff_gain
    if plasticity_mask is not None:
        dw *= plasticity_mask[idx]
    data[idx] = np.clip(data[idx] + dw, w_min, w_max)
    decay_total = np.float32(d_per_step ** interval)
    elig_slice *= decay_total
    eligibility[idx] = elig_slice
    return np.abs(elig_slice) > prune_threshold


def neuromod_decay_full(
    eligibility: np.ndarray,
    data: np.ndarray,
    modulator: float,
    interval_gain: float,
    decay: float,
    w_min: float,
    w_max: float,
    plasticity_mask: np.ndarray | None = None,
) -> None:
    """Apply neuromodulation + decay in-place on the entire eligibility array.

    Used when active-set tracking was abandoned (>80% of nnz active).
    Mutates both ``data`` and ``eligibility`` in-place.
    """
    if COMPILED_STDP_ENABLED:
        if plasticity_mask is not None:
            _neuromod_decay_full_masked_njit(
                eligibility,
                data,
                float(modulator),
                float(interval_gain),
                float(decay),
                float(w_min),
                float(w_max),
                plasticity_mask,
            )
        else:
            _neuromod_decay_full_njit(
                eligibility,
                data,
                float(modulator),
                float(interval_gain),
                float(decay),
                float(w_min),
                float(w_max),
            )
        return
    # NumPy fallback
    dw = np.float32(modulator) * eligibility * np.float32(interval_gain)
    if plasticity_mask is not None:
        dw *= plasticity_mask
    data[:] = np.clip(data + dw, w_min, w_max)
    eligibility *= np.float32(decay)
