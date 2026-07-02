"""Experimental GPU-accelerated CSR synapse ops (issue #135 research spike).

Scope: this module ports ONLY the sparse-dense matrix-vector product in
``SynapseGroup.compute_current`` (spike propagation) to JAX. STDP/R-STDP/
BCM/eligibility-trace updates are deliberately NOT ported here — they use
data-dependent-shape sparse gathering and in-place CSR-data mutation
(``synapses.py``'s ``_gather_active_synapses`` / ``update_weights_stdp``),
which fights JAX's static-shape JIT model and would need a substantial
masked/dense-shadow rewrite to fit. See docs/GPU-SYNAPSE-BACKEND-FEASIBILITY.md
for the full writeup, including why this module is NOT wired into
``SynapseGroup`` as a live runtime option: CPU-only benchmarking here found
it consistently slower than the existing hand-tuned scipy CPU path, so it
is kept as a standalone, independently-tested module pending real GPU
hardware validation.

Design intent (per the issue's acceptance criteria):
- Functional equivalence with the CPU implementation (see test_gpu_synapse_ops.py).
- "GPU when available, CPU otherwise" comes from JAX's own device model —
  the same code runs on whatever ``jax.devices()`` resolves to. No hand-rolled
  CuPy-style try/except-GPU-then-numpy dance is needed.
- No new hard dependency: jax/jaxlib are only imported inside functions here,
  never at module import time, so the rest of neuromorphic works unmodified
  without the optional ``gpu`` extra installed (``pip install .[gpu]``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def gpu_backend_available() -> bool:
    """True if jax is importable. Does not imply an actual GPU/TPU device —
    JAX transparently falls back to its CPU backend when none is visible."""
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def _compute_current_numpy(
    weights: sparse.csr_matrix,
    pre_spikes: np.ndarray,
    pre_output_sign: np.ndarray | None = None,
) -> np.ndarray:
    """Plain CPU fallback — the same computation as SynapseGroup.compute_current's
    default (non-CSC-gather) path, used when jax isn't installed at all."""
    n_post = weights.shape[0]
    if weights.nnz == 0:
        return np.zeros(n_post, dtype=np.float32)
    if pre_output_sign is not None:
        spike_vec = pre_spikes.astype(np.float32) * pre_output_sign
    else:
        spike_vec = pre_spikes.astype(np.float32)
    return np.asarray(weights @ spike_vec).flatten().astype(np.float32)


def compute_current_jax(
    weights: sparse.csr_matrix,
    pre_spikes: np.ndarray,
    pre_output_sign: np.ndarray | None = None,
) -> np.ndarray:
    """JAX sparse matrix-vector product — functional equivalent of
    SynapseGroup.compute_current's default CSR path.

    Runs on GPU/TPU if jax.devices() resolves one, CPU otherwise — the
    same code path either way (no separate GPU/CPU implementations to
    keep in sync). Falls back to the pure-numpy path if jax is not
    installed, so callers never need their own try/except around this.

    Args:
        weights: (n_post, n_pre) scipy CSR sparse weight matrix.
        pre_spikes: boolean/float array (n_pre,) of which pre neurons spiked.
        pre_output_sign: optional +1/-1 array (n_pre,) for E/I balance.

    Returns:
        float32 array (n_post,) of input currents — bit-for-bit equivalent
        (within float32 summation-order tolerance) to the CPU implementation.
    """
    if not gpu_backend_available():
        return _compute_current_numpy(weights, pre_spikes, pre_output_sign)

    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    n_post, n_pre = weights.shape
    if weights.nnz == 0:
        return np.zeros(n_post, dtype=np.float32)

    if pre_output_sign is not None:
        spike_vec = pre_spikes.astype(np.float32) * pre_output_sign
    else:
        spike_vec = pre_spikes.astype(np.float32)

    bcoo = jsparse.BCOO.from_scipy_sparse(weights)
    out = _jit_spmv(bcoo.data, bcoo.indices, jnp.asarray(spike_vec), n_post, n_pre)
    return np.asarray(out, dtype=np.float32)


_JIT_CACHE: dict[tuple[int, int], Any] = {}


def _jit_spmv(data, indices, spike_vec, n_post: int, n_pre: int):
    """JIT-compiled BCOO @ dense-vector matvec, cached per (n_post, n_pre)
    shape so repeated calls at the same network size reuse the compiled
    kernel instead of re-tracing every step."""
    import jax
    from jax.experimental import sparse as jsparse

    key = (n_post, n_pre)
    fn = _JIT_CACHE.get(key)
    if fn is None:

        def _mv(d, idx, s):
            m = jsparse.BCOO((d, idx), shape=(n_post, n_pre))
            return m @ s

        fn = jax.jit(_mv)
        _JIT_CACHE[key] = fn
    return fn(data, indices, spike_vec)
