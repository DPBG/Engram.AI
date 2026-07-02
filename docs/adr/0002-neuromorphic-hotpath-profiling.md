# 0002 — Neuromorphic hot-path profiling & optimization strategy

- **Status:** Proposed
- **Date:** 2026-07-01
- **Tracking issue:** [#133](https://github.com/DPBG/Engram.AI/issues/133) (M4)
- **Milestone:** M4 — Real-Time Performance
- **Phase:** 4 (Real-Time Performance)
- **Related:** issue [#135](https://github.com/DPBG/Engram.AI/issues/135) (GPU backend
  feasibility for CSR synapse ops — a separate, not-yet-merged PR reached a
  complementary conclusion: on CPU, a JAX backend for `compute_current` was
  12–37× *slower* than the existing scipy implementation at every scale
  tested, so it isn't a near-term fix for the bottleneck this ADR identifies
  in the same function. See that issue's PR for the full writeup.)

---

## Context

README's Known Limitations section already names the problem this issue
is scoping:

> The core simulation is Python + NumPy/SciPy. It runs and learns, but it
> is not real-time at the ~1M-neuron scale on commodity hardware. Real-time
> embodied use will require GPU/compiled kernels or neuromorphic hardware.

CLAUDE.md §6 requires the neuromorphic hot path to be vectorized (CSR
sparse matrices, `float32`, no per-neuron Python loops), and Invariant 1's
enforcement note requires any change to update intervals to preserve
mathematical equivalence with running "logically every step." Until now,
no profiling data existed to say **which** operations actually dominate
step time, so any optimization work would have been guesswork. This ADR
is that missing data, plus a concrete strategy built on it.

---

## Methodology

Profiled with `cProfile` (stdlib) at a **representative scale**: the exact
config `launcher/registry.py`'s `_NEURO_SMALL` dict uses for `python run.py`'s
default **core** profile — i.e. what a contributor's machine actually runs
locally, not an arbitrary size:

```
brainstem=2000, reflex_arc=1500, sensory_cortex=12000, motor_cortex=6000,
cerebellum=6000, association_cortex=12000, predictive_layer=6000,
working_memory=2000, feature_layer=5000, concept_layer=1500, meta_controller=1000
→ 55,000 neurons total, cognitive layer enabled
```

20 warmup steps (excluded from the profile, so allocator-range settling
and lazy initialization don't skew the numbers) followed by 200 profiled
steps, injecting synchronized visual+auditory input every step (`network.
inject_multimodal` → `network.step`) so STDP, dendritic integration, and
plasticity all actually run rather than idling. Raw profile:
`neuromorphic/profiling/hotpath_20260701.pstats` (load with
`pstats.Stats(path)`; script that produced it: `neuromorphic/scripts/
profile_hotpath.py`).

**Environment note**: single run, one machine, no GPU present (consistent
with #135's finding). This is enough to rank bottlenecks and validate the
threading finding below (which was confirmed with a direct A/B comparison,
not just profiler inference) — it is not a multi-machine statistical study.
Re-running this exact script is how to validate any optimization proposed
here.

---

## Findings

### Top 5 bottlenecks by self-time (% of 4.322s profiled step time)

Numbers below are from the exact committed artifact
(`neuromorphic/profiling/hotpath_20260701.pstats`) — reproduce with
`uv run python scripts/profile_hotpath.py --steps 200 --top 20`.

| # | Bottleneck | Self time | % of step time |
|---|---|---:|---:|
| 1 | `SynapseGroup.compute_current` (`synapses.py:197`) | 0.690s | **16.0%** |
| 2 | Thread-pool dispatch overhead — `ThreadPoolExecutor` futures machinery (`queue.get`/`put`, `_thread.lock.acquire`/`release`) inside `_route_parallel` (`network.py:749`) | 0.607s | **14.0%** |
| 3 | `NeuronPopulation.step` — LIF/AdEx integration (`neurons.py:255`) | 0.389s | **9.0%** |
| 4 | `step_dendrites` — multi-compartment dendritic integration, Invariant 5 (`neurons.py:171`) | 0.368s | **8.5%** |
| 5 | `np.unique`/`_unique_hash` — active-synapse dedup in `_gather_active_synapses` and CSC index building (`synapses.py:294`, `:123`) | 0.242s | **5.6%** |

**Together these five account for ~53% of step time** (consistent across
repeated runs — a first pass measured 16.7/13.9/8.9/8.3/5.5%, within ~1
point of the numbers above on every dimension). Honorable mentions just
below the cut: `_ensure_csc` CSC-rebuild (`synapses.py:123`, partially
overlapping with #5 above), the `network._compute_current` dispatch
wrapper (3.2%), and the actual C-level `scipy.sparse._sparsetools.
csr_matvec` kernel itself (2.2% — notably *cheap*; most of #1's cost is
Python-level orchestration around the matvec, not the matvec itself, see
below).

Bottlenecks #3 and #4 are expected and not easily removable: Invariant 1
(all 6 learning mechanisms always run) and Invariant 5 (every cortical
synapse targets a specific dendritic compartment) mandate this work happen
every step. They're listed here as an honest accounting of where time
goes, not as things to cut.

### The surprising finding: the default thread pool makes the default config *slower*

`network.py:200`'s `_route_parallel` uses a `ThreadPoolExecutor`
(`NEURO_STDP_THREADS`, default 8) on the stated assumption that "SciPy CSR
SpMV releases the GIL, so threads give genuine parallelism." That's true
in principle, but profiling shows thread-pool *dispatch* overhead (#2
above, 14.0%) is nearly as large as the compute it's meant to parallelize
(#1, 16.0%). A direct A/B measurement (`NEURO_STDP_THREADS=1` vs. the
default `=8`, `neuromorphic/scripts/profile_hotpath.py --compare-threading`)
confirms this isn't a profiler artifact:

| Scale | Serial (`THREADS=1`) | Parallel (`THREADS=8`, default) | Result |
|---|---:|---:|---|
| 55K neurons (dev-scale default) | 9.80 ms/step (102.1 steps/s) | 15.58 ms/step (64.2 steps/s) | **Parallel is 1.59× *slower*** |
| 220K neurons (4× dev-scale) | 229.97 ms/step (4.35 steps/s) | 199.48 ms/step (5.01 steps/s) | Parallel is 1.15× faster |

There's a genuine scale-dependent crossover somewhere between 55K and
220K neurons: below it, per-task work (`compute_current` for one synapse
group) is too small relative to the fixed cost of `executor.submit()` +
future resolution + lock-guarded queue access, so parallelizing makes
things worse. Above it, there's enough compute per task to amortize that
cost. **The default profile (`_NEURO_SMALL`, what every contributor runs
locally via `python run.py`) sits below the crossover** — meaning the
thread pool that exists to make things faster is currently making the
single most common way to run Engram slower, by ~37% (`(15.58-9.80)/15.58`).

---

## Decision

Adopt a two-track strategy: a low-risk, immediately-actionable fix for the
threading regression, plus documented next steps for the two Python-level
hot spots.

### 1. Make thread-pool usage scale-adaptive (near-term, low risk)

`NEURO_STDP_THREADS` should not default to a fixed 8 regardless of network
size. Concretely: gate `_route_parallel`'s parallel path on the total
`nnz` (or neuron count) it's about to process, falling back to serial
below an empirically-set threshold (somewhere between 55K and 220K
neurons per the measurement above — needs a few more data points in that
range to pin down, not guessed). This is a config/dispatch-logic change,
not a numerics change — it doesn't touch STDP, eligibility, or any
learning-rule math, so it carries no risk to Invariant 1's equivalence
guarantee. Expected win: ~37% faster steps at the default dev scale, for
free, with no accuracy trade-off.

### 2. `compute_current`'s Python-level orchestration (medium-term)

The actual sparse matvec (`csr_matvec`) is only 2.2% of step time; the
`compute_current` method wrapping it is 16.0%. That gap is Python-level
work: the CSR-vs-CSC strategy selection, spike-vector construction, and
(on the CSC path) the flat-index-gather machinery in `_gather_active_synapses`
(itself contributing bottleneck #5 above, `np.unique` calls). Two
candidate directions, in order of how well they fit this codebase:

- **Re-examine the CSR/CSC strategy thresholds** (`synapses.py:226`,
  `nnz < 100_000 or firing_rate > 0.3 or nnz > 10_000_000`) against this
  profiling data — those thresholds were tuned before any profiler ever
  ran against them. It's plausible the crossover point is wrong for
  today's typical `nnz` distribution.
- **Compiled kernels for the CSC-gather path** (Numba `@njit`, not
  Cython — no build-step/compiler-toolchain addition beyond a pure-Python
  dependency, unlike Cython). This is the "compiled kernels" option
  README's Known Limitations section names, and unlike #135's GPU-backend
  finding, a CPU JIT avoids the two problems that sank GPU here: no new
  ~90MB+ dependency tier, and no host↔device transfer per step. Scope as
  a follow-up spike with its own before/after profiling run using this
  same methodology, not assumed to work.

**GPU offloading is explicitly not recommended as a near-term direction**
for this specific function — #135 already tested it and found JAX-on-CPU
12–37× slower than the current implementation, with no GPU hardware
available to test the actual GPU case. Revisit only if real GPU hardware
becomes available and the simulation moves toward keeping state
GPU-resident across steps (avoiding per-step host↔device transfer) — see
issue #135's PR for the full reasoning.

### 3. Dendritic and neuron integration (#3, #4) — no action proposed here

These are core, invariant-mandated computation, not incidental overhead.
Any future optimization here (e.g. a compiled kernel for the LIF/AdEx
update) must ship with an equivalence test per CLAUDE.md's Invariant 1
enforcement note — out of scope for this ADR, noted for a future one.

---

## Consequences

### Positive

- A concrete, data-backed optimization backlog exists for the first time —
  future perf work has a baseline to compare against (re-run
  `profile_hotpath.py`) instead of guessing.
- The threading fix (§1) is a genuine, measured, near-zero-risk win at the
  most common deployment scale.
- Clarifies that GPU offloading (#135) and thread-pool tuning (this ADR)
  are two independent findings that both point away from "add more
  parallelism" and toward "reduce Python-level dispatch overhead" as the
  more promising direction for `compute_current` specifically.

### Negative / costs

- The scale-adaptive threading threshold needs a handful of additional
  profiling runs between 55K and 220K neurons to place correctly — not
  done in this ADR, flagged as the first follow-up.
- Compiled-kernel work (§2) is a new build-time dependency (Numba) and a
  nontrivial implementation effort; this ADR scopes it as a spike, not a
  commitment.
- This profiling run is single-machine, single-scale-pair evidence. It's
  sufficient to justify the threading fix (the A/B gap is large and
  consistent in direction) but a smaller optimization might need more
  repeated measurement to separate signal from noise.

### Follow-up (unblocked by this ADR)

1. Implement scale-adaptive `_route_parallel` dispatch; re-run this ADR's
   profiling script before/after to confirm the ~37% dev-scale win and
   verify no regression at large scale.
2. Locate the actual serial/parallel crossover point with a few more
   scale data points (e.g. 80K, 120K, 160K neurons).
3. Spike: Numba-JIT the CSC-gather path in `_gather_active_synapses` /
   `compute_current`; profile before/after with this same methodology.
4. Re-examine the hardcoded CSR/CSC strategy thresholds in
   `synapses.py:226` against real `nnz`/firing-rate distributions from a
   running system, not just this synthetic benchmark's inputs.

---

## References

- CLAUDE.md §6 (coding standards — vectorized hot paths, CSR, float32) and
  Invariant 1's equivalence requirement.
- README.md Known Limitations (performance/scale, GPU/compiled-kernels).
- `neuromorphic/src/neuromorphic/network.py:749` (`_route_parallel`),
  `:200` (`_stdp_executor` construction).
- `neuromorphic/src/neuromorphic/synapses.py:197` (`compute_current`),
  `:294` (`_gather_active_synapses`), `:226` (CSR/CSC strategy threshold).
- `launcher/registry.py` (`_NEURO_SMALL` — the profiled config).
- Issue [#135](https://github.com/DPBG/Engram.AI/issues/135) (GPU backend
  feasibility for the same `compute_current` function).
- `neuromorphic/scripts/profile_hotpath.py` (this ADR's profiling script).
