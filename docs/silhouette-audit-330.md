# Silhouette / Linear-Probe Verification — Issue #330

> Verification of `ConceptSeparabilityBenchmark`'s hand-rolled silhouette
> score and nearest-centroid linear-probe accuracy
> (`neuromorphic/src/neuromorphic/benchmarks.py`) against
> `sklearn.metrics.silhouette_score` on synthetic fixtures, per the issue's
> acceptance criteria. Comparison stays a dev-time/CI-time-only step —
> `sklearn` is not a dependency of the shipped benchmark code, only of the
> `verify` optional extra used to run the comparison.

## Verification command

```bash
cd neuromorphic && uv run --extra verify python scripts/verify_silhouette_score.py
```

See `scripts/verify_silhouette_score.py` for the full comparison harness and
its module docstring for exactly what it checks and why (10 synthetic
fixtures: well-separated clusters, overlapping/near-random clusters,
imbalanced class sizes, a singleton cluster, the minimum-allowed 2 classes,
and 5 randomized trials).

## Result: one unintentional divergence found and fixed

**Silhouette score — singleton clusters.** The hand-rolled implementation
computed a(i) (mean intra-cluster distance) as `0.0` when a sample's cluster
has no other members, then ran that through the general formula
`s(i) = (b(i) - a(i)) / max(a(i), b(i))`, producing `1.0` — "perfectly
separated" — for a singleton cluster.

`sklearn.metrics.silhouette_samples`'s own source comments this exact case:
`# nan values are for clusters of size 1, and should be 0`. Rousseeuw's
original 1987 definition treats a(i) as **undefined**, not 0, for a
singleton cluster, and the documented convention is to define the whole
score as 0 (neither well- nor poorly-clustered — the question isn't
answerable, not maximally answered "yes"). Deriving 1.0 from the general
formula was an unintentional divergence from that convention, not a
documented design choice.

Confirmed via `scripts/verify_silhouette_score.py`'s `singleton_cluster`
fixture: `sil_samples[0]` (the singleton) was `1.0` from the hand-rolled
code vs `0.0` from sklearn; every other sample in that fixture already
matched exactly.

**Fix:** `silhouette_scores_from_distance_matrix()` now special-cases a
singleton cluster's sample to score `0.0` directly, matching sklearn/the
textbook convention, instead of routing through the general formula.

**Practical impact:** low. The benchmark's default configuration
(`probe_reps=3`) gives every class the same sample count, so a singleton
cluster cannot occur in a normal benchmark run — it only arises if a future
caller passes `probe_reps=1` for some subset of patterns, or otherwise
produces an unbalanced sample count per class. No conclusion the benchmark
suite has been used to support depended on this path. Fixed anyway because
the cost was trivial and leaving a known-wrong edge case in a function named
after a textbook statistic is exactly the risk this issue was opened to
close off.

**Linear-probe accuracy (nearest-centroid, leave-one-out):** no divergence
found. Verified against a ground-truth built from `sklearn.metrics.pairwise.
cosine_similarity` + `sklearn.preprocessing.normalize` (not
`sklearn.neighbors.NearestCentroid`, which only supports
`metric ∈ {"euclidean", "manhattan"}` in scikit-learn≥1.3 — using it would
have silently substituted a different algorithm than the one
`benchmarks.py` implements). All 10 fixtures matched within float64
precision (`1e-9`), including the `overlapping` fixture where accuracy is a
non-trivial ~29%, not a degenerate 0%/100%.

## What changed

- `neuromorphic/src/neuromorphic/benchmarks.py` — extracted the inline
  silhouette and leave-one-out nearest-centroid logic out of
  `ConceptSeparabilityBenchmark.run()` into two standalone, independently
  testable functions (`silhouette_scores_from_distance_matrix`,
  `nearest_centroid_loo_accuracy`), so the verification script and the unit
  tests below exercise the exact shipped code, not a reimplementation of it.
  Fixed the singleton-cluster case in the former.
- `neuromorphic/scripts/verify_silhouette_score.py` — the dev-only
  comparison harness (new).
- `neuromorphic/pyproject.toml` — added a `verify` optional extra
  (`scikit-learn>=1.3.0`), used only by the script above. Not part of `dev`,
  which the whole test suite depends on, and not a dependency of
  `benchmarks.py` itself.
- `neuromorphic/tests/test_benchmarks.py` — direct unit tests for both
  extracted functions (`TestSilhouetteScoresFromDistanceMatrix`,
  `TestNearestCentroidLooAccuracy`), including a regression test for the
  singleton-cluster fix so this stays caught by the normal CI suite without
  needing scikit-learn installed.

## References

- Issue [#330](https://github.com/DPBG/Engram.AI/issues/330) (M3.45).
- `sklearn.metrics.silhouette_samples` source (scikit-learn 1.3+): the
  `# nan values are for clusters of size 1, and should be 0` comment this
  audit's fix now matches.
- Rousseeuw, P.J. (1987). "Silhouettes: a Graphical Aid to the Interpretation
  and Validation of Cluster Analysis." *Computational and Applied
  Mathematics*, 20, 53–65.
