"""
Ground-truth verification for ConceptSeparabilityBenchmark's hand-rolled
silhouette score and nearest-centroid linear-probe accuracy (issue #330).

benchmarks.py deliberately avoids an sklearn dependency (see its module
docstring), so this comparison stays a dev-time/CI-time verification step,
never a runtime one: sklearn is only ever imported here, in a script under
neuromorphic/scripts/, gated behind the `verify` optional extra. If this
script ever needs to become a pytest test, it must stay opt-in (skip when
sklearn is absent) rather than adding sklearn to the shared `dev` extra,
which the whole test suite depends on.

What this checks:
  1. Silhouette: neuromorphic.benchmarks.silhouette_scores_from_distance_matrix
     (the exact function ConceptSeparabilityBenchmark.run() calls) against
     sklearn.metrics.silhouette_samples, on the SAME precomputed cosine
     distance matrix — isolating the aggregation formula from any distance-
     metric choice.
  2. Linear probe: neuromorphic.benchmarks.nearest_centroid_loo_accuracy
     against a leave-one-out (sklearn.model_selection.LeaveOneOut) nearest-
     unit-centroid classifier built from sklearn's own normalize() and
     cosine_similarity() primitives. sklearn.neighbors.NearestCentroid only
     supports metric in {"euclidean", "manhattan"} (checked against
     scikit-learn>=1.3.0) — using it would silently swap in a different
     algorithm (nearest raw centroid by Euclidean distance) instead of
     checking the one benchmarks.py actually implements (nearest
     *unit-normalized* centroid by cosine similarity), so the ground truth
     here is built from cosine_similarity()/normalize() directly instead.

Fixtures cover: well-separated clusters, overlapping/near-random clusters,
a singleton cluster (tests the divide-by-zero convention), and randomized
trials for statistical coverage — not just one hand-picked "nice" case.

Usage:
    cd neuromorphic && uv run --extra verify python scripts/verify_silhouette_score.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuromorphic.benchmarks import (  # noqa: E402
    nearest_centroid_loo_accuracy,
    silhouette_scores_from_distance_matrix,
)

try:
    from sklearn.metrics import silhouette_samples
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import normalize
except ImportError:
    print(
        "scikit-learn not installed. Run with:\n"
        "  cd neuromorphic && uv run --extra verify python scripts/verify_silhouette_score.py",
        file=sys.stderr,
    )
    sys.exit(1)

# Silhouette scores can legitimately differ by a few ULPs between the
# hand-rolled float64 aggregation and sklearn's — this is a numerical-noise
# floor, not a correctness threshold to relax if a real divergence shows up.
TOLERANCE = 1e-9


def _cosine_distance_matrix(mat: np.ndarray) -> np.ndarray:
    """Mirror ConceptSeparabilityBenchmark.run()'s exact distance computation:
    L2-normalise rows, then D[i,j] = 1 - cosine_similarity, clipped to
    [0, 2], zero diagonal."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_unit = mat / norms
    sim = mat_unit @ mat_unit.T
    dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    return dist, mat_unit


def _make_fixture(
    rng: np.random.Generator,
    n_per_class: list[int],
    n_features: int,
    separation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic class-clustered feature matrix.

    Each class gets a random centroid scaled by `separation`; larger
    `separation` -> more separated clusters -> silhouette closer to 1.
    `separation=0.0` collapses all centroids to the origin, so classes only
    differ by noise -> silhouette closer to 0.
    """
    labels = []
    rows = []
    for class_id, n in enumerate(n_per_class):
        centroid = rng.normal(size=n_features) * separation
        for _ in range(n):
            rows.append(centroid + rng.normal(size=n_features) * 0.5)
            labels.append(class_id)
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int32)


def _compare_silhouette(name: str, mat: np.ndarray, labels: np.ndarray) -> float:
    dist, _mat_unit = _cosine_distance_matrix(mat)
    ours = silhouette_scores_from_distance_matrix(dist, labels)
    theirs = silhouette_samples(dist, labels, metric="precomputed")
    max_diff = float(np.max(np.abs(ours - theirs)))
    status = "OK" if max_diff <= TOLERANCE else "DIVERGENCE"
    print(
        f"[{status}] silhouette  {name:32s} "
        f"mean(ours)={ours.mean():+.6f}  mean(sklearn)={theirs.mean():+.6f}  "
        f"max|diff|={max_diff:.3e}"
    )
    return max_diff


def _compare_linear_probe(name: str, mat: np.ndarray, labels: np.ndarray) -> float:
    _dist, mat_unit = _cosine_distance_matrix(mat)
    ours = nearest_centroid_loo_accuracy(mat_unit, labels)

    # sklearn ground truth for the SAME algorithm (nearest unit-normalized
    # centroid by cosine similarity), built from trusted sklearn primitives
    # rather than NearestCentroid (which doesn't offer a cosine metric and
    # would silently test a different classifier -- see module docstring).
    n = len(labels)
    unique_labels = np.unique(labels)
    correct = 0
    for train_idx, test_idx in LeaveOneOut().split(mat_unit):
        query = mat_unit[test_idx]
        train_labels = labels[train_idx]
        centroids = np.array(
            [
                # A held-out singleton class has zero training members left;
                # mirror nearest_centroid_loo_accuracy's convention of a
                # zero-vector centroid in that case rather than NaN.
                (
                    mat_unit[train_idx][train_labels == lbl].mean(axis=0)
                    if (train_labels == lbl).any()
                    else np.zeros(mat_unit.shape[1])
                )
                for lbl in unique_labels
            ]
        )
        centroids_unit = normalize(centroids)
        sims = cosine_similarity(query, centroids_unit)[0]
        pred = unique_labels[int(np.argmax(sims))]
        correct += int(pred == labels[test_idx[0]])
    theirs = correct / n

    diff = abs(ours - theirs)
    status = "OK" if diff <= TOLERANCE else "DIVERGENCE"
    print(
        f"[{status}] linear probe {name:31s} ours={ours:.6f}  sklearn={theirs:.6f}  diff={diff:.3e}"
    )
    return diff


def main() -> int:
    rng = np.random.default_rng(seed=42)
    worst = 0.0

    fixtures: list[tuple[str, np.ndarray, np.ndarray]] = []

    # 1. Well-separated clusters -> silhouette should be high and positive.
    mat, labels = _make_fixture(rng, [8, 8, 8], n_features=16, separation=8.0)
    fixtures.append(("well_separated", mat, labels))

    # 2. Overlapping/near-random clusters -> silhouette should be near 0.
    mat, labels = _make_fixture(rng, [8, 8, 8], n_features=16, separation=0.0)
    fixtures.append(("overlapping", mat, labels))

    # 3. Imbalanced class sizes.
    mat, labels = _make_fixture(rng, [3, 10, 20], n_features=12, separation=4.0)
    fixtures.append(("imbalanced_classes", mat, labels))

    # 4. Singleton cluster -> exercises the a(i)=0 divide-by-zero convention.
    mat, labels = _make_fixture(rng, [1, 6, 6], n_features=10, separation=5.0)
    fixtures.append(("singleton_cluster", mat, labels))

    # 5. Two classes only (minimum n_classes the benchmark itself allows).
    mat, labels = _make_fixture(rng, [5, 5], n_features=8, separation=3.0)
    fixtures.append(("two_classes_minimum", mat, labels))

    # 6-10. Randomized trials across varied shapes/separations for
    # statistical coverage beyond hand-picked cases.
    for trial in range(5):
        n_classes = int(rng.integers(2, 6))
        sizes = [int(rng.integers(2, 12)) for _ in range(n_classes)]
        n_features = int(rng.integers(4, 32))
        separation = float(rng.uniform(0.0, 6.0))
        mat, labels = _make_fixture(rng, sizes, n_features, separation)
        fixtures.append((f"random_trial_{trial}", mat, labels))

    print("=" * 100)
    for name, mat, labels in fixtures:
        worst = max(worst, _compare_silhouette(name, mat, labels))
        worst = max(worst, _compare_linear_probe(name, mat, labels))
    print("=" * 100)

    if worst > TOLERANCE:
        print(
            f"\nDIVERGENCE FOUND: max difference {worst:.3e} exceeds tolerance "
            f"{TOLERANCE:.0e}. See this script's module docstring (issue #330) "
            "for the fix-or-document-deviation decision process."
        )
        return 1

    print(
        f"\nAll {len(fixtures)} fixtures match sklearn within {TOLERANCE:.0e} "
        "(float64 numerical-noise floor). No divergence found — "
        "neuromorphic/src/neuromorphic/benchmarks.py's hand-rolled "
        "silhouette_scores_from_distance_matrix() and "
        "nearest_centroid_loo_accuracy() match their textbook definitions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
