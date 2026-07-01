# GPU Backend Feasibility for CSR Synapse Operations

> **Status**: Research spike + feasibility report (issue #135, Phase 4 —
> Real-Time Performance). Code: `neuromorphic/src/neuromorphic/gpu_synapse_ops.py`,
> `neuromorphic/scripts/benchmark_gpu_synapse.py`, `neuromorphic/tests/test_gpu_synapse_ops.py`.
>
> **Recommendation: do not adopt yet.** On CPU, the JAX backend is
> consistently 12–37× *slower* than the existing scipy implementation at
> every scale tested. That doesn't prove a real GPU would also lose — no
> GPU hardware was available to test — but it does mean this can't be
> merged into the default runtime path on the evidence available today.
> See [§5](#5-recommendation) for exactly what would change that.

## 1. Context

README's Known Limitations section already names the problem this issue
is scoping:

> The core simulation is Python + NumPy/SciPy. It runs and learns, but it
> is not real-time at the ~1M-neuron scale on commodity hardware. Real-time
> embodied use will require GPU/compiled kernels or neuromorphic hardware.

`neuromorphic/src/neuromorphic/synapses.py`'s `SynapseGroup` already uses
`scipy.sparse` CSR matrices with several rounds of CPU-specific hand
optimization (see [§3](#3-why-only-compute_current-was-ported)) — this
report evaluates whether CuPy or JAX can accelerate that further.

**Environment note**: this spike was done on a machine with no GPU, no
CUDA toolkit, and no `nvidia-smi` — confirmed at the start of this work.
Every number below is CPU-only. That shapes both the methodology (§4) and
why CuPy specifically could not be evaluated at all (§2).

## 2. CuPy vs. JAX

The issue proposed either CuPy or JAX. Only JAX was actually testable here:

| | CuPy | JAX |
|---|---|---|
| Requires CUDA to run at all | **Yes** — no CPU mode exists | No — runs on CPU automatically when no GPU/TPU is visible |
| Testable in this environment | No | Yes (`jax.devices()` → `[CpuDevice(id=0)]`) |
| API | NumPy-compatible drop-in, separate from CPU code | Single `jax.numpy` API, same code runs on CPU/GPU/TPU |
| "CPU fallback" (acceptance criterion 3) | Would require hand-writing a second numpy code path and switching between them | Comes for free from JAX's device model |

Criterion 3 ("CPU fallback operates successfully without GPU hardware")
is a much more natural fit for JAX than CuPy: with CuPy, "fallback" means
maintaining two implementations and an `if gpu_available: ... else: ...`
branch (the classic pattern that rots — the CPU branch silently drifts
out of sync because it's rarely exercised in GPU-equipped dev environments).
JAX's device-agnostic array API sidesteps that: `gpu_synapse_ops.py`
has exactly one implementation, and it's the same code on CPU and GPU.
For that reason alone, if a GPU backend is pursued later, **JAX is the
better fit** — independent of the CPU performance results below.

## 3. Why only `compute_current` was ported

`SynapseGroup`'s hot paths fall into two categories:

**Good GPU/JIT fit — ported (`gpu_synapse_ops.compute_current_jax`):**
`compute_current` is a pure function: `weights @ spike_vector` (sparse-dense
matvec), called every simulation step, with static input/output shapes for
a given network size. This is exactly the shape JAX's `jax.jit` wants.

**Poor GPU/JIT fit — deliberately NOT ported:**
`update_weights_stdp`, `update_weights_rstdp`, `apply_neuromodulation_and_decay`,
and `_gather_active_synapses` all:
- **Mutate `weights.data` / `eligibility` in place.** JAX arrays are
  immutable; this would need to become copy-on-write or move to a
  functional-update pattern throughout `SynapseGroup`, not just these methods.
- **Produce data-dependent output shapes.** `_gather_active_synapses`
  returns an array sized by how many synapses are connected to neurons
  that spiked *this step* — that count varies every call. `jax.jit`
  requires static shapes; a data-dependent-size gather forces either
  re-tracing on every distinct shape (defeating JIT) or a rewrite to a
  fixed-size masked representation (dense boolean mask over all `nnz`,
  which throws away the sparse-gather speedup the CPU code was written
  to get in the first place — see the `nnz < 100_000` / `> 10_000_000`
  branching in `_gather_active_synapses` and `compute_current`).
- **Branch on runtime array sizes in Python** (`if nnz < 100_000 or ...`)
  to pick between CSR-SpMV and CSC-column-gather strategies. `jax.jit`
  traces Python control flow once per input *shape*; a strategy switch
  keyed off `nnz`/firing-rate would need `jax.lax.cond`/`jax.lax.select`
  rewrites throughout.

Porting these properly is a substantially larger rewrite than this spike's
scope — and per [§4](#4-methodology--results), there's no performance
evidence yet that justifies taking that on.

## 4. Methodology & results

`scripts/benchmark_gpu_synapse.py` builds a real `SynapseGroup` (same class
used in production, not a synthetic reimplementation) at three sizes chosen
to straddle `compute_current`'s own CSR/CSC strategy thresholds (100K and
10M nnz), and times `n` repeated calls to the existing CPU
`compute_current()` against `compute_current_jax()` at each size, using the
identical weight matrix and spike vector for both. Correctness is checked
via `max(|jax_result - cpu_result|)`.

Run with `cd neuromorphic && uv run --extra gpu python scripts/benchmark_gpu_synapse.py`.
Raw output: `neuromorphic/benchmarks/gpu_synapse_spike.json`. Reproduced here:

| Scale | nnz | scipy CPU (ms/call) | JAX CPU (ms/call) | Slowdown | Max \|diff\| |
|---|---:|---:|---:|---:|---:|
| small (2K×2K, 5% fire) | 79,218 | 0.041 | 1.528 | **37.5×** | 0.0 |
| medium (20K×20K, 5% fire) | 3,980,284 | 3.289 | 40.208 | **12.2×** | 1.4×10⁻⁶ |
| large (200K×200K, 2% fire) | 19,994,970 | 18.551 | 296.036 | **16.0×** | 0.0 |

System: `Linux-6.17.0-35-generic-x86_64`, JAX 0.10.2, `jax.devices() == [CpuDevice(id=0)]`.

**Correctness (acceptance criterion 1 — functional equivalence):** exact
match or within float32 summation-order tolerance (~1e-6) at every scale,
confirmed both here and in `tests/test_gpu_synapse_ops.py`'s parametrized
equivalence suite (11 tests, including empty-weights, no-spikes, E/I-sign,
and a JIT-cache-reuse-with-changed-weights regression check).

**Performance (acceptance criterion 2 — measurable gains):** the opposite
of a gain. JAX-on-CPU is 12–37× slower than `scipy.sparse`'s CSR/CSC SpMV
at every tested scale, worsening at small scale rather than improving —
`compute_current_jax` reconstructs a `jax.experimental.sparse.BCOO` from
the scipy CSR matrix on every call (unavoidable, since weights actually
change between STDP updates in production — caching that conversion
would silently serve stale weights), and that conversion plus JIT
dispatch overhead dominates at small problem sizes where scipy's C SpMV
is already near-instant.

**This does not, by itself, tell us what a real GPU would do.** GPU
sparse SpMV wins (when it does) come from massive core-count parallelism
on very large matrices, which this CPU-only run cannot exercise. But it
does mean the current evidence gives no basis to expect a win — see §5.

**CPU fallback (acceptance criterion 3):** confirmed two ways —
(a) JAX's own device selection: `jax.devices()` returns `[CpuDevice(id=0)]`
automatically since no GPU/TPU is visible, no code branch required;
(b) `gpu_synapse_ops.py` also handles jax not being *installed at all*
(`gpu_backend_available() == False` → `_compute_current_numpy` fallback),
verified by a test that monkeypatches jax's absence and by manually
simulating an `ImportError` on `import jax` end-to-end.

## 5. Dependency overhead & cross-platform compatibility (criterion 4)

**Install size.** CPU-only `jax` + `jaxlib` + `ml-dtypes`: **~90MB**
(`jax` 3.1MB + `ml-dtypes` 4.8MB + `jaxlib` 81.5MB), measured via
`uv pip install jax` in this environment. A CUDA-enabled install
(`jax[cuda12]` or similar) pulls a CUDA-specific `jaxlib` wheel plus
matching cuDNN/cuBLAS libraries — commonly several hundred MB to ~1GB,
though this could not be measured directly here (no CUDA toolkit present,
and this sandbox's disk was already at 97% full mid-spike — installing a
multi-hundred-MB CUDA wheel speculatively wasn't a responsible use of the
shared environment). Either way, this is added as an **optional** extra
(`pip install .[gpu]`, `neuromorphic/pyproject.toml`'s `gpu` group) —
never a default dependency — so it doesn't affect the base install size
for anyone not opting in.

**Cross-platform.** `scipy.sparse` ships prebuilt wheels for essentially
every platform Engram already targets (Linux x86_64/ARM64, macOS
x86_64/ARM64, Windows). JAX's platform matrix is narrower: full CUDA GPU
support is Linux x86_64-only; macOS has CPU support with limited/no Metal
GPU acceleration; Windows GPU support requires WSL; ARM (relevant for
Raspberry Pi / Jetson embedded targets — see README's hardware-integration
limitation) has more limited official wheel coverage than x86_64. Since
`neuromorphic` already declares `mujoco>=3.0.0` as a hard dependency with
its own platform constraints, this isn't a new category of risk, but it
does compound: an optional `gpu` extra should stay optional, not become
a stealth requirement for any single code path.

**Deployment footprint.** `docs/ARCHITECTURE.md` picked NATS partly for its
"single 10MB binary, no dependencies" footprint. A ~90MB (CPU) to
~1GB (CUDA) ML framework dependency is a different order of magnitude —
fine as an opt-in extra for a workstation/cloud deployment profile, a poor
fit for the embedded/robotics targets `sensory-gateway`/actuator work
(#138) is oriented toward.

**Fork-safety interaction, observed directly in this spike.** Running the
full `neuromorphic` test suite with jax importable in the process (from
`test_gpu_synapse_ops.py`) surfaced a real warning from unrelated tests:
`persistence.py:397`'s background-save path calls `os.fork()`, and jax's
multithreaded runtime is present in the parent process at that point —
Python emits `RuntimeWarning: os.fork() was called ... JAX is
multithreaded ... this will likely lead to a deadlock`. It didn't fail
any test here, but it's a real, environment-independent risk (not
specific to this sandbox) that would need addressing — e.g. skipping
fork-based background save under `_FORK_BGSAVE` when jax is loaded, or
isolating GPU-backend code to a separate process — before `jax` could
safely become anything more than an opt-in, rarely-imported extra.

## 6. Recommendation

**Do not wire `gpu_synapse_ops.py` into `SynapseGroup.compute_current` as
a live runtime option yet.** It is intentionally *not* imported or called
anywhere outside its own tests/benchmark script in this PR — zero risk to
the existing, already-tuned CPU hot path or to Invariant 1's "logically
every step" equivalence guarantees, since nothing existing was modified.

Revisit when any of the following changes:
1. **Real CUDA GPU hardware is available** to benchmark against — the one
   input this spike structurally could not provide. Re-run
   `benchmark_gpu_synapse.py` on it before drawing conclusions either way.
2. **The simulation moves toward GPU-resident state across steps**, rather
   than converting scipy CSR → JAX BCOO and transferring host↔device on
   every single step. Per-step round-trips are the classic reason
   promising-on-paper GPU ops lose in practice; amortizing that requires
   keeping weights/spikes on-device for many consecutive steps, which is
   a materially bigger architectural change than this spike's scope.
3. **STDP/eligibility-trace updates get a JIT-friendly (masked/dense)
   rewrite** — until then, even a fast GPU `compute_current` would leave
   the learning-rule updates (arguably the more expensive path at high
   firing rates) on CPU, capping the achievable end-to-end speedup.

## 7. Artifacts

- `neuromorphic/src/neuromorphic/gpu_synapse_ops.py` — the spike implementation
- `neuromorphic/tests/test_gpu_synapse_ops.py` — equivalence + fallback tests (skips cleanly without the `gpu` extra)
- `neuromorphic/scripts/benchmark_gpu_synapse.py` — reproducible benchmark
- `neuromorphic/benchmarks/gpu_synapse_spike.json` — raw results backing §4
- `neuromorphic/pyproject.toml` — `gpu` optional extra (`jax>=0.4.20`)
