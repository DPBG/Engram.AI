"""Compiled (Numba-JIT) kernels for neuromorphic hot paths.

When Numba is available and ``NEURO_COMPILED_STDP`` is not set to ``0``, this
module exposes JIT-compiled versions of the most expensive operations in
the learning pipeline:

  1. ``stdp_delta``             — fused LTP/LTD kernel (single pass, no temps)
  2. ``neuromod_decay_sparse``  — fused neuromod + decay + clip + prune over the
                                  active sparse eligibility set
  3. ``neuromod_decay_full``    — same, but operating on the full array (used
                                  when active-set tracking is abandoned at >80%)

Separately, when Numba is available and ``NEURO_COMPILED_DEDUP`` is not set to
``0``, this module also exposes:

  4. ``dedup_indices``          — O(n) active-synapse-index dedup (issue #435,
                                  ADR 0002 bottleneck #5), replacing
                                  ``synapses.py``'s ``np.unique``-based dedup
                                  in ``_gather_active_synapses``. Open-addressing
                                  hash set over just the active-index count (n),
                                  not the full nnz range -- true O(n), no sort.

When Numba is unavailable (or the relevant flag is ``0``), the same public
functions fall back to NumPy-equivalent implementations so ``synapses.py``
needs no branching of its own.

Performance note (Invariant 1):
  Numba JIT-compiles lazily on the first call with ~100–300 ms overhead per
  signature.  ``cache=True`` persists compiled artifacts to ``__pycache__``
  so subsequent process starts reuse the cached binary.  ``fastmath=True``
  lets LLVM reassociate floating-point operations; the resulting values remain
  within float32 rounding of the NumPy reference (verified by the equivalence
  tests in ``test_compiled_stdp.py`` and ``test_compiled_dedup.py``).
  ``dedup_indices`` carries no ``fastmath`` (integer dedup, not float math) and
  returns values in a different order than ``np.unique`` -- callers must not
  depend on sorted output (verified: neither call site in ``synapses.py`` does).

Feature flags:
  ``NEURO_COMPILED_STDP=0``   — disable STDP/neuromod compiled kernels
  ``NEURO_COMPILED_STDP=1``   — enable when Numba is importable (default)
  ``NEURO_COMPILED_DEDUP=0``  — disable the compiled dedup kernel
  ``NEURO_COMPILED_DEDUP=1``  — enable when Numba is importable (default)
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

# Independently toggleable: dedup (issue #435) is unrelated numerics to STDP/
# neuromod, so it gets its own flag rather than piggybacking on
# NEURO_COMPILED_STDP's name (which would be misleading for a dedup kernel).
_env_dedup = os.environ.get("NEURO_COMPILED_DEDUP", "1").lower().strip()
COMPILED_DEDUP_ENABLED: bool = NUMBA_AVAILABLE and _env_dedup not in ("0", "false", "no")

if COMPILED_DEDUP_ENABLED:
    logger.debug("neuromorphic: Numba compiled dedup kernel active (cache=True)")
else:
    logger.debug(
        "neuromorphic: Numba compiled dedup kernel disabled "
        "(NUMBA_AVAILABLE=%s, NEURO_COMPILED_DEDUP=%r)",
        NUMBA_AVAILABLE,
        _env_dedup,
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
    decay_total = np.float32(d_per_step**interval)
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


# ── active-synapse dedup (issue #435, ADR 0002 bottleneck #5) ──────────────
#
# synapses.py's _gather_active_synapses "large matrix" path (100K < nnz <
# 10M) concatenates the CSR-post and CSC-pre active-index arrays and calls
# np.unique() to dedup them -- an O(n log n) sort, profiled at 5.6% of step
# time. The smaller-matrix path already gets O(1)-per-entry dedup for free
# via a full nnz-sized boolean mask, but a full-nnz mask is exactly what the
# large-matrix path exists to avoid (allocating up to 10M+ bools every step).
#
# An open-addressing hash set sized to O(n) (n = the *active* index count,
# not nnz) gets true O(n) expected-time dedup without an nnz-sized
# allocation. Order of the result is scan order, not sorted -- both call
# sites in synapses.py use the result purely as fancy-index arrays
# (rows[active_idx], eligibility[active_idx] += dw, etc.), so sort order was
# never semantically required; only "each active index appears exactly
# once" is (duplicate indices under += would silently under-count via
# NumPy's last-write-wins fancy-indexing behavior).

if COMPILED_DEDUP_ENABLED:
    import numba as _nb_dedup  # type: ignore[import-untyped]

    @_nb_dedup.njit(cache=True)
    def _dedup_indices_njit(combined: np.ndarray) -> np.ndarray:
        """Open-addressing hash-set dedup, O(n) expected time, O(n) space."""
        n = combined.shape[0]
        if n == 0:
            return combined.copy()
        # Table size: next power of two >= 4n (load factor <= 0.25 keeps
        # expected probes-per-op low), minimum 16 to avoid degenerate small
        # tables.
        table_size = 16
        while table_size < n * 4:
            table_size *= 2
        table_mask = np.int64(table_size - 1)
        table = np.full(table_size, -1, dtype=np.int64)
        output = np.empty(n, dtype=combined.dtype)
        count = 0
        for i in range(n):
            val = np.int64(combined[i])
            # Knuth multiplicative hash; uint64 arithmetic sidesteps signed
            # overflow, then masked down to the table's index range.
            h = np.int64((np.uint64(val) * np.uint64(2654435761)) & np.uint64(table_mask))
            found = False
            while table[h] != -1:
                if table[h] == val:
                    found = True
                    break
                h = (h + 1) & table_mask
            if not found:
                table[h] = val
                output[count] = combined[i]
                count += 1
        return output[:count]


def dedup_indices(combined: np.ndarray) -> np.ndarray:
    """Deduplicate an array of active-synapse indices.

    Semantically equivalent to ``np.unique(combined)`` for this codebase's
    use -- same *set* of values, exactly one occurrence each -- but does NOT
    guarantee sorted output. Callers must only rely on set membership /
    uniqueness, never on ordering. (Verified for synapses.py's two call
    sites: both use the result purely as a fancy-index array.)

    Args:
        combined: int array of possibly-duplicated indices.

    Returns:
        Array of the same dtype containing each unique value from
        ``combined`` exactly once, in first-occurrence scan order (not
        sorted).
    """
    if COMPILED_DEDUP_ENABLED:
        return _dedup_indices_njit(combined)
    # NumPy fallback -- equivalent to the original synapses.py implementation.
    # np.unique's sorted output is a strict superset of this function's
    # contract (sorted output is *also* a valid dedup), so this fallback
    # remains a correct implementation even though compiled path doesn't sort.
    return np.unique(combined)
