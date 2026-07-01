"""
GPU synapse-op backend benchmark (issue #135 research spike).

Compares SynapseGroup.compute_current's existing scipy CPU path against
the experimental JAX backend (gpu_synapse_ops.compute_current_jax) at
representative neuron/synapse counts, and verifies functional equivalence
at each scale. Results feed docs/GPU-SYNAPSE-BACKEND-FEASIBILITY.md.

This machine has no GPU/CUDA — jax runs on its CPU backend here, so these
numbers show "JAX-on-CPU vs scipy-on-CPU", not a real GPU comparison. See
the feasibility doc for how to read that honestly.

Usage:
    cd neuromorphic && uv run --extra gpu python scripts/benchmark_gpu_synapse.py
    cd neuromorphic && uv run --extra gpu python scripts/benchmark_gpu_synapse.py --output benchmarks/gpu_synapse.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuromorphic.gpu_synapse_ops import compute_current_jax, gpu_backend_available
from neuromorphic.synapses import SynapseGroup

# (n_pre, n_post, sparsity, fire_rate) — chosen to span SynapseGroup.compute_current's
# own adaptive-strategy thresholds (100K / 10M nnz) so results are meaningful
# relative to the code being compared against, not arbitrary round numbers.
SCALES = [
    ("small", 2_000, 2_000, 0.02, 0.05),      # ~80K nnz — below the 100K CSC-gather threshold
    ("medium", 20_000, 20_000, 0.01, 0.05),   # ~4M nnz — inside the CSC-gather range
    ("large", 200_000, 200_000, 0.0005, 0.02),  # ~20M nnz — above the CSC-gather range
]


def _time_calls(fn, calls: int) -> float:
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    return (time.perf_counter() - t0) / calls


def run_scale(name, n_pre, n_post, sparsity, fire_rate, calls=20) -> dict:
    rng = np.random.default_rng(0)
    group = SynapseGroup(n_pre=n_pre, n_post=n_post, sparsity=sparsity, init_weight=0.3, rng=rng)
    pre_spikes = rng.random(n_pre) < fire_rate

    cpu_result = group.compute_current(pre_spikes)
    t_cpu = _time_calls(lambda: group.compute_current(pre_spikes), calls)

    if gpu_backend_available():
        jax_result = compute_current_jax(group.weights, pre_spikes)  # warmup / jit compile
        t_jax = _time_calls(lambda: compute_current_jax(group.weights, pre_spikes), calls)
        max_abs_diff = float(np.max(np.abs(jax_result - cpu_result)))
    else:
        t_jax = None
        max_abs_diff = None

    return {
        "scale": name,
        "n_pre": n_pre,
        "n_post": n_post,
        "sparsity": sparsity,
        "fire_rate": fire_rate,
        "nnz": int(group.weights.nnz),
        "cpu_ms_per_call": t_cpu * 1e3,
        "jax_ms_per_call": t_jax * 1e3 if t_jax is not None else None,
        "slowdown_factor": (t_jax / t_cpu) if t_jax is not None else None,
        "max_abs_diff": max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--calls", type=int, default=20, help="Timed calls per scale")
    args = parser.parse_args()

    print(f"jax available: {gpu_backend_available()}")
    if gpu_backend_available():
        import jax
        print(f"jax devices: {jax.devices()}")
    print()

    results = [run_scale(*scale, calls=args.calls) for scale in SCALES]

    print(f"{'scale':<8} {'nnz':>10} {'scipy(ms)':>10} {'jax(ms)':>10} {'slowdown':>9} {'max|diff|':>10}")
    for r in results:
        jax_ms = f"{r['jax_ms_per_call']:.3f}" if r["jax_ms_per_call"] is not None else "n/a"
        slow = f"{r['slowdown_factor']:.1f}x" if r["slowdown_factor"] is not None else "n/a"
        diff = f"{r['max_abs_diff']:.2e}" if r["max_abs_diff"] is not None else "n/a"
        print(f"{r['scale']:<8} {r['nnz']:>10} {r['cpu_ms_per_call']:>10.3f} {jax_ms:>10} {slow:>9} {diff:>10}")

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "gpu_backend_available": gpu_backend_available(),
        "results": results,
    }
    if gpu_backend_available():
        import jax
        output["jax_version"] = jax.__version__
        output["jax_devices"] = [str(d) for d in jax.devices()]

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
