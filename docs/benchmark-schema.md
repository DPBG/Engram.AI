# Benchmark JSON Schema — Stable Public Contract

> Documents `neuromorphic.benchmarks.BenchmarkSuite.run_all()`'s output — the
> JSON `save_results()` writes to `neuromorphic/benchmarks/benchmarks_*.json`
> — as a contract the dashboard and any other consumer can code against
> without re-reading `benchmarks.py`. Tracking issue:
> [#325](https://github.com/DPBG/Engram.AI/issues/325) (M3.40).

## Two schemas share `neuromorphic/benchmarks/`, and that's the real ambiguity

Before the field reference: the issue that requested this doc describes "two
consumers independently reading undocumented subsets of the same JSON" —
checking that against the code, this is **half right and worth correcting
precisely**, because the imprecise version is itself a trap for the next
person who reads it.

**`neuromorphic/benchmarks/` holds output from two unrelated scripts:**

| Producer | Filename pattern | Top-level keys |
|---|---|---|
| `BenchmarkSuite.save_results()` (this doc) | `benchmarks_YYYYMMDD_HHMMSS.json` (plural) | `timestamp`, `step_count`, `total_neurons`, `elapsed_s`, `cross_modal_recall`, `novelty_detection`, `association_strength`, `energy_efficiency`, `concept_separability`, `cross_modal_binding_accuracy` |
| `scripts/benchmark.py` (a separate step-timing/speed benchmark, unrelated to `BenchmarkSuite`) | `benchmark_YYYYMMDD_HHMMSS.json` (singular) | `timestamp`, `system`, `config`, `init`, `speed`, `learning`, `memory`, `final_state` |

`dashboard/src/dashboard/learning_evidence.py`'s `is_benchmark_result_file()`
matches **both** filenames with one regex (`^benchmarks?_\d{8}_\d{6}\.json$` —
the `s?` makes "benchmark_" and "benchmarks_" both match), and
`extract_learning_metrics()` reads a single JSON blob defensively across
**both** schemas at once (e.g. it tries `association_strength.concept_count`
first, then falls back to `learning.concept_count`; tries `step_count` at the
top level, then `final_state.step_count`). This is why the dashboard's
parser looks like it's guessing — it genuinely doesn't know in advance which
of the two schemas a given file has, and both can be present in the same
directory from different tooling runs.

**`neuromorphic/scripts/benchmark_ci_gate.py` reads `scripts/benchmark.py`'s
`speed`/`learning` keys — not `BenchmarkSuite`'s schema at all.** It invokes
`scripts/benchmark.py` directly and checks `result["speed"]["steps_per_sec"]`
and `result["learning"]["steps_per_sec"]` against a committed baseline. It
never touches `cross_modal_recall`, `concept_separability`, or any other key
this doc covers. So "the CI gate reads a subset of the same JSON the
dashboard reads" is not accurate — it reads a **different script's** JSON
that happens to land in the same directory with a confusingly similar
filename. The rest of this document is the schema `benchmark_ci_gate.py`
does *not* use; `scripts/benchmark.py`'s own schema is out of scope here.

## Top-level `BenchmarkSuite.run_all()` result

Always present, regardless of any individual benchmark's internal state:

| Key | Type | Notes |
|---|---|---|
| `timestamp` | `str` | `%Y-%m-%dT%H:%M:%S`, suite run time. |
| `step_count` | `int` | `network.step_count` at the time `run_all()` was called. |
| `total_neurons` | `int` | `network.config.populations.total`. |
| `elapsed_s` | `float` | Wall-clock seconds for the whole suite. |
| `cross_modal_recall` | `dict` | Benchmark 1 — see below. Always full shape. |
| `novelty_detection` | `dict` | Benchmark 2 — see below. Always full shape. |
| `association_strength` | `dict` | Benchmark 3 — see below. Always full shape; nested dicts can be empty. |
| `energy_efficiency` | `dict` | Benchmark 4 — see below. Always full shape. |
| `concept_separability` | `dict` | Benchmark 5 — see below. **Two distinct shapes** — see its section. |
| `cross_modal_binding_accuracy` | `dict` | Benchmark 6 — see below. Always full shape. |

Five of the six per-benchmark dicts (1–4, 6) **always** return their full,
fixed key set — a degenerate input (empty patterns, no plastic synapses,
etc.) produces zero/default-valued fields, never a reduced shape and never
an `error` key. **Benchmark 5 (`concept_separability`) is the one exception**
and is the specific ambiguity issue #308 (item #23) flags — see its section.

### 1. `cross_modal_recall` (`CrossModalRecallBenchmark`)

Always this exact key set:

| Key | Type |
|---|---|
| `visual_to_auditory_recall` | `float` (0.0–1.0) |
| `auditory_to_visual_recall` | `float` (0.0–1.0) |
| `binding_strength_before` | `float` |
| `binding_strength_after` | `float` |
| `binding_strength_delta` | `float` |
| `n_cross_modal_before` | `int` |
| `n_cross_modal_after` | `int` |
| `patterns_tested` | `int` |

### 2. `novelty_detection` (`NoveltyDetectionBenchmark`)

Always this exact key set:

| Key | Type |
|---|---|
| `familiar_pred_error` | `float`, `>= 0.0` |
| `novel_pred_error` | `float`, `>= 0.0` |
| `discrimination_ratio` | `float` |
| `familiarization_steps` | `int` |
| `firing_rate_shift` | `dict[str, float]` — **keys are region names, not fixed**; see below |

`firing_rate_shift`'s keys come from `network.get_firing_rates()`, i.e. one
entry per currently-populated brain region (`brainstem`, `sensory_cortex`,
`concept`, etc., depending on which regions have `n > 0` in the running
network's config). A consumer must iterate this dict rather than index into
it by a hardcoded region name, since which regions exist is a network
configuration choice, not part of this schema.

### 3. `association_strength` (`AssociationStrengthBenchmark`)

Top-level key set is always present:

| Key | Type |
|---|---|
| `weight_changes` | `dict[str, dict]` — **conditional entries**, see below |
| `myelination` | `dict[str, float]` — **conditional entries**, see below |
| `concept_count` | `int` |
| `patterns_trained` | `int` |
| `training_reps` | `int` |

`weight_changes` and `myelination` are always present as dicts, but their
**per-group entries are conditional**, keyed by
`AssociationStrengthBenchmark.BINDING_GROUPS = ("sensory_association",
"sensory_feature", "feature_association")`:

- A group appears in `weight_changes` only if that synapse group had
  `nnz > 0` at both the pre- and post-training snapshot. Each present entry
  has `initial_mean`, `final_mean`, `delta_mean`, `initial_max`, `final_max`,
  `delta_max`, `initial_std`, `final_std` (all `float`).
- A group appears in `myelination` only if it exists, is `plastic`, and has
  a non-`None` `myelinated` array. Each present entry is a `float` fraction
  in `[0, 1]`.

Both dicts can be `{}` (empty) if no binding group qualifies — this is not
an error, just means no measurable groups existed at benchmark time.

### 4. `energy_efficiency` (`EnergyEfficiencyBenchmark`)

Always this exact key set:

| Key | Type |
|---|---|
| `mean_spikes_per_step` | `float` |
| `global_firing_rate` | `float` |
| `region_firing_rates` | `dict[str, float]` — keys are region names present in the network, same caveat as `firing_rate_shift` above |
| `approx_energy_units` | `float` |
| `spikes_per_association` | `float` |
| `total_steps` | `int` |
| `total_neurons` | `int` |

### 5. `concept_separability` (`ConceptSeparabilityBenchmark`) — two shapes

**This is the field the issue was opened to pin down.** Unlike every other
benchmark, this one has an `error` key that, when present, means the rest of
the dict is a **reduced, degenerate shape** — not the full one. A consumer
that only checks for the presence of `silhouette_score` cannot tell a real
low score apart from a benchmark that didn't run at all (this exact gap is
issue [#308](https://github.com/DPBG/Engram.AI/issues/308), item #23 — a CI
check for this is proposed there, not yet implemented as of this doc).

**Error shape** — returned when `network.concept is None` (no concept layer
configured) or when there are fewer than 2 classes / fewer than 4 samples
(`"insufficient patterns for separability"`):

| Key | Type | Value |
|---|---|---|
| `error` | `str` | `"no concept layer"` or `"insufficient patterns for separability"` |
| `silhouette_score` | `float` | Always `0.0` — **not a measured score** |
| `linear_probe_accuracy` | `float` | Always `0.0` — **not a measured score** |
| `n_patterns` | `int` | |

**Success shape** — no `error` key, full set of 11 keys:

| Key | Type |
|---|---|
| `silhouette_score` | `float`, `[-1, 1]` |
| `linear_probe_accuracy` | `float`, `[0, 1]` |
| `mean_intra_class_distance` | `float`, `>= 0` |
| `mean_inter_class_distance` | `float`, `>= 0` |
| `separation_ratio` | `float`, `>= 0` |
| `n_patterns` | `int` |
| `n_samples` | `int` |
| `concept_neurons` | `int` |
| `top_neurons_per_pattern` | `list[list[int]]` |
| `training_reps` | `int` |
| `probe_reps` | `int` |

**Contract for any consumer of this key:** check `"error" in result` first.
If present, treat `silhouette_score`/`linear_probe_accuracy` as *absent*
(they are placeholder zeros, not measurements) — never plot, average, or
gate on them. `BenchmarkSuite.summary()` already does this correctly
(`if "error" not in cs: ... else: ... (skipped — {cs['error']})`); use it as
the reference implementation for this check.

### 6. `cross_modal_binding_accuracy` (`CrossModalBindingAccuracyBenchmark`)

Always this exact key set (no error path):

| Key | Type |
|---|---|
| `precision` | `float`, `[0, 1]` |
| `recall` | `float`, `[0, 1]` |
| `f1` | `float`, `[0, 1]` |
| `true_positives` | `int` |
| `false_positives` | `int` |
| `false_negatives` | `int` |
| `binding_strength_before` | `float` |
| `binding_strength_after` | `float` |
| `binding_strength_delta` | `float` |
| `n_cross_modal_before` | `int` |
| `n_cross_modal_after` | `int` |
| `matched_coupling_mean` | `float` |
| `decoy_coupling_mean` | `float` |
| `matched_to_decoy_ratio` | `float` |
| `pairs_tested` | `int` |
| `training_reps` | `int` |
| `fixture_seed` | `int` |
| `coupling_matrix` | `list[list[float]]`, `n_pairs × n_pairs` |

## What each real consumer actually reads today

**`dashboard/src/dashboard/learning_evidence.py` (`extract_learning_metrics`)**
reads, from this schema: `association_strength.concept_count`,
`cross_modal_binding_accuracy.{f1,precision,recall,matched_to_decoy_ratio}`,
`cross_modal_recall.{visual_to_auditory_recall,auditory_to_visual_recall,
binding_strength_after,binding_strength_delta}`,
`concept_separability.{silhouette_score,linear_probe_accuracy}`,
top-level `step_count`/`total_neurons`. It also reads `scripts/benchmark.py`'s
`learning.concept_count`, `final_state.step_count`, `config.total_neurons` as
fallbacks for files from that other producer — see the two-schemas section
above.

`silhouette_score` is mapped to the dashboard scalar `concept_separability`
(trend series + summary stat). As of issue
[#287](https://github.com/DPBG/Engram.AI/issues/287), `linear_probe_accuracy`
is also surfaced as its own reported scalar / trend series
(`linear_probe_accuracy`) — an independent nearest-centroid clustering
cross-check, not only a fallback when silhouette is absent.
`BenchmarkSuite.summary()` already printed both; the dashboard gap was the
missing second series.

**It does not currently apply the `concept_separability` error-shape
contract above** (it calls `.get("silhouette_score")` / `.get("linear_probe_accuracy")`
unconditionally, which are `0.0` in both shapes) — worth fixing separately if
issue [#308](https://github.com/DPBG/Engram.AI/issues/308) lands, since a
degenerate run would otherwise plot as a real zero score on the dashboard's
trend line.

**`neuromorphic/scripts/benchmark_ci_gate.py`** does not read this schema —
see the two-schemas section above.

## Stability contract

- **Additive changes are safe.** Adding a new top-level benchmark or a new
  field to an existing benchmark's dict does not break either consumer
  above, since both index by key name, not by iterating/asserting the full
  key set.
- **Renaming or removing an existing key is a breaking change** for the
  consumers listed above by name. Grep for the key across
  `dashboard/src/dashboard/` and `neuromorphic/scripts/` before renaming one.
- **Changing `concept_separability`'s error-vs-success shape distinction**
  (e.g. adding a new error path, or changing which keys the error shape
  carries) must be reflected in this doc's "two shapes" section above, since
  it's the one benchmark whose shape isn't fixed.
