"""
Structured benchmarking framework for investor-ready metrics.

Provides 6 benchmark tests that produce quantitative proof the brain learns:
1. CrossModalRecall — inject visual, measure auditory cortex activation (and vice versa)
2. NoveltyDetection — present known vs unknown stimuli, measure response difference
3. AssociationStrength — measure weight changes after paired multi-modal training
4. EnergyEfficiency — compute energy per learned association vs baseline
5. ConceptSeparability — silhouette score + linear-probe accuracy over concept-layer activations
6. CrossModalBindingAccuracy — precision/recall of bound modality pairs vs ground truth

Usage:
    from neuromorphic.benchmarks import BenchmarkSuite
    suite = BenchmarkSuite(network)
    results = suite.run_all()
    suite.save_results(results, "benchmarks/")

    # Multiple seeds, with mean + confidence interval per metric — a single-seed
    # score can't tell "the network learned this" from "this init got lucky":
    results = suite.run_multi_seed(n_seeds=5)
    print(suite.summary_multi_seed(results))

    # Per-region quiet-vs-sparse triage (issue #331) — is a quiet region
    # undertrained or intentionally sparse by design (k-WTA)? summary()
    # includes this automatically; call directly for a saved JSON result:
    from neuromorphic.benchmarks import classify_quiet_regions, format_quiet_region_report
    flags = classify_quiet_regions(results["energy_efficiency"])
    print(format_quiet_region_report(flags))

    # Or run from checkpoint on server:
    cd neuromorphic && uv run python -m neuromorphic.benchmarks --checkpoint /data/sqlite/neuromorphic.db
    cd neuromorphic && uv run python -m neuromorphic.benchmarks --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from neuromorphic.binding_fixtures import generate_correlated_stimulus_fixtures
from neuromorphic.cross_modal_probe import CrossModalProbe

if TYPE_CHECKING:
    from neuromorphic.network import NeuromorphicNetwork

logger = logging.getLogger(__name__)


# Discrete blob widths (issue #327): the prior generator always used sigma=50,
# so every pattern was the *same* Gaussian blob translated to a new position —
# one shape class with n>64 producing exact duplicates (positions are (i*7)%64,
# (i*13)%64, period 64). Cycling through several widths gives genuine
# shape-class diversity in addition to position, not just translation.
_VISUAL_SIGMAS: tuple[float, ...] = (25.0, 50.0, 90.0)


def generate_test_patterns(n: int, rng: np.random.Generator) -> list[dict]:
    """Visual gaussian blobs (varying position + size) with auditory signatures
    (varying which/how-many dims are elevated).

    See issue #327: the previous version had a single fixed blob shape
    (position-only diversity, exactly repeating every 64 patterns) and
    auditory vectors whose 12 shared-noise dims dominated similarity
    (~0.9 mean cosine similarity between any two patterns regardless of
    which single dim was "hot"). Position is now drawn from ``rng`` instead
    of a fixed-period formula, size cycles through _VISUAL_SIGMAS, and the
    auditory signature elevates 1-3 dims instead of always exactly 1.
    """
    y, x = np.mgrid[0:64, 0:64]
    patterns = []
    for i in range(n):
        cx = int(rng.integers(0, 64))
        cy = int(rng.integers(0, 64))
        sigma = _VISUAL_SIGMAS[i % len(_VISUAL_SIGMAS)]
        d2 = (x - cx).astype(np.float32) ** 2 + (y - cy).astype(np.float32) ** 2
        vis = np.exp(-d2 / sigma, dtype=np.float32).flatten()
        vis /= vis.max() + 1e-8

        # Background at 0.5±0.1 (the old constant) made every pattern's 12
        # shared-noise dims dominate the dot product, so any two patterns had
        # ~0.9 mean cosine similarity regardless of which dim was "hot" —
        # a lower, tighter background floor lets the elevated dims actually
        # distinguish classes (measured ~0.9 -> ~0.44 mean pairwise similarity).
        n_hot = 1 + (i % 3)  # 1, 2, or 3 elevated dims -> more distinguishable classes
        hot_dims = {(i + k * 4) % 13 for k in range(n_hot)}
        aud = rng.normal(0.15, 0.05, size=13).astype(np.float32)
        for hd in hot_dims:
            aud[hd] = 1.0
        np.clip(aud, 0.0, 1.0, out=aud)

        patterns.append(
            {"visual": vis.tolist(), "auditory": aud.tolist(), "label": f"pattern_{i:03d}"}
        )
    return patterns


def audit_pattern_diversity(patterns: list[dict]) -> dict[str, float]:
    """Quantify generate_test_patterns() diversity (issue #327).

    Every benchmark in BenchmarkSuite shares one pattern generator, so a
    narrow-diversity generator is a single point of failure all 6 metrics
    inherit. This returns exact-duplicate counts and mean pairwise cosine
    similarity per modality so a future regression (e.g. reverting to one
    fixed blob shape) is visible as unique counts dropping below
    len(patterns) or similarity climbing back toward 1.0, instead of silently
    degrading every downstream benchmark's ability to distinguish patterns.
    """
    if not patterns:
        return {
            "n_patterns": 0,
            "unique_visual": 0,
            "unique_auditory": 0,
            "mean_visual_cosine_sim": 0.0,
            "mean_auditory_cosine_sim": 0.0,
        }
    vis = np.array([p["visual"] for p in patterns], dtype=np.float64)
    aud = np.array([p["auditory"] for p in patterns], dtype=np.float64)

    def _unique_count(arr: np.ndarray) -> int:
        return len({tuple(np.round(row, 6)) for row in arr})

    def _mean_pairwise_cosine(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 2:
            return 0.0
        norms = np.linalg.norm(arr, axis=1) + 1e-9
        normalized = arr / norms[:, None]
        sim_matrix = normalized @ normalized.T
        iu = np.triu_indices(n, k=1)
        return float(np.mean(sim_matrix[iu]))

    return {
        "n_patterns": len(patterns),
        "unique_visual": _unique_count(vis),
        "unique_auditory": _unique_count(aud),
        "mean_visual_cosine_sim": round(_mean_pairwise_cosine(vis), 4),
        "mean_auditory_cosine_sim": round(_mean_pairwise_cosine(aud), 4),
    }


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python natives."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _flatten_numeric(d: Any, prefix: str = "") -> dict[str, float]:
    """Recursively flatten a nested results dict to {"dotted.path": numeric_value}.

    Strings (timestamps, phase names, ...) and booleans are dropped; only
    values that can be aggregated across seeds survive.
    """
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten_numeric(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, bool):
        pass
    elif isinstance(d, (int, float, np.integer, np.floating)):
        out[prefix] = float(d)
    return out


def _confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Mean and CI half-width via Student's t-distribution (valid for small N).

    Returns (mean, half_width) such that the interval is [mean - half_width,
    mean + half_width]. With fewer than 2 samples the interval collapses to
    a point (half_width=0) since variance is undefined.
    """
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    n = len(arr)
    if n < 2:
        return mean, 0.0
    sem = float(arr.std(ddof=1)) / np.sqrt(n)
    t_crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return mean, t_crit * sem


def _inject_paired(net: NeuromorphicNetwork, pat: dict, steps: int) -> None:
    """Inject visual then paired visual+auditory for the given step count."""
    half = steps // 2 or 1
    vc = net.inject_observation(pat["visual"], provenance="sensor.videofile.bench")
    for _ in range(half):
        net.step(vc)
        vc *= 0.97
    ac = net.inject_observation(pat["auditory"], provenance="sensor.audiofile.bench")
    combined = vc + ac
    for _ in range(half):
        net.step(combined)
        combined *= 0.97


def _inject_multi_step(net: NeuromorphicNetwork, pat: dict, steps: int) -> None:
    """Inject both modalities simultaneously and step."""
    c = net.inject_multimodal(
        {"sensor.videofile.bench": pat["visual"], "sensor.audiofile.bench": pat["auditory"]}
    )
    for _ in range(steps):
        net.step(c)
        c *= 0.97


# ---------------------------------------------------------------------------
# Benchmark 1: Cross-Modal Recall
# ---------------------------------------------------------------------------
class CrossModalRecallBenchmark:
    """Train paired patterns, test recall by presenting only one modality."""

    def __init__(self, net: NeuromorphicNetwork) -> None:
        self._net, self._probe = net, CrossModalProbe()

    def run(
        self, patterns: list[dict], training_reps: int = 10, steps_per_pattern: int = 20
    ) -> dict[str, Any]:
        net, probe = self._net, self._probe
        pre = probe.probe_network(net).to_dict()
        for _ in range(training_reps):
            for pat in patterns:
                _inject_paired(net, pat, steps_per_pattern)
        post = probe.probe_network(net).to_dict()
        half = steps_per_pattern // 2 or 1
        v2a, a2v = [], []
        for pat in patterns:
            c = net.inject_observation(pat["visual"], provenance="sensor.videofile.bench")
            for _ in range(half):
                net.step(c)
                c *= 0.97
            v2a.append(probe.probe_network(net).recall_ratio_visual)
            c = net.inject_observation(pat["auditory"], provenance="sensor.audiofile.bench")
            for _ in range(half):
                net.step(c)
                c *= 0.97
            a2v.append(probe.probe_network(net).recall_ratio_auditory)
        bs_pre = pre.get("binding_strength", 0.0)
        bs_post = post.get("binding_strength", 0.0)
        return {
            "visual_to_auditory_recall": round(float(np.mean(v2a)), 4) if v2a else 0.0,
            "auditory_to_visual_recall": round(float(np.mean(a2v)), 4) if a2v else 0.0,
            "binding_strength_before": round(bs_pre, 4),
            "binding_strength_after": round(bs_post, 4),
            "binding_strength_delta": round(bs_post - bs_pre, 6),
            "n_cross_modal_before": pre.get("n_cross_modal", 0),
            "n_cross_modal_after": post.get("n_cross_modal", 0),
            "patterns_tested": len(patterns),
        }


# ---------------------------------------------------------------------------
# Benchmark 2: Novelty Detection
# ---------------------------------------------------------------------------
class NoveltyDetectionBenchmark:
    """Familiar vs novel stimuli — measure prediction error difference."""

    def __init__(self, net: NeuromorphicNetwork) -> None:
        self._net = net

    def run(
        self,
        familiar_pattern: dict,
        novel_pattern: dict,
        familiarization_reps: int = 20,
        steps_per_rep: int = 20,
        test_steps: int = 10,
    ) -> dict[str, Any]:
        net = self._net
        for _ in range(familiarization_reps):
            _inject_multi_step(net, familiar_pattern, steps_per_rep)

        def _measure(pat: dict) -> tuple[list[float], dict[str, list[float]]]:
            c = net.inject_multimodal(
                {"sensor.videofile.bench": pat["visual"], "sensor.audiofile.bench": pat["auditory"]}
            )
            errs, rates = [], {}
            for _ in range(test_steps):
                net.step(c)
                c *= 0.97
                errs.append(net.prediction_decoder.compute_prediction_error(net.predictive))
                for name, r in net.get_firing_rates().items():
                    rates.setdefault(name, []).append(r)
            return errs, rates

        fam_errs, fam_rates = _measure(familiar_pattern)
        nov_errs, nov_rates = _measure(novel_pattern)
        fam_e = float(np.mean(fam_errs)) if fam_errs else 0.0
        nov_e = float(np.mean(nov_errs)) if nov_errs else 0.0
        rate_shift = {}
        for name in fam_rates:
            rate_shift[name] = round(
                float(np.mean(nov_rates.get(name, [0.0]))) - float(np.mean(fam_rates[name])), 6
            )
        return {
            "familiar_pred_error": round(fam_e, 4),
            "novel_pred_error": round(nov_e, 4),
            "discrimination_ratio": round(nov_e / (fam_e + 1e-8), 4),
            "familiarization_steps": familiarization_reps * steps_per_rep,
            "firing_rate_shift": rate_shift,
        }


# ---------------------------------------------------------------------------
# Benchmark 3: Association Strength
# ---------------------------------------------------------------------------
class AssociationStrengthBenchmark:
    """Weight changes + myelination + concept count after paired training."""

    BINDING_GROUPS = ("sensory_association", "sensory_feature", "feature_association")

    def __init__(self, net: NeuromorphicNetwork) -> None:
        self._net = net

    def _snap(self) -> dict[str, dict[str, float]]:
        out = {}
        for nm in self.BINDING_GROUPS:
            s = self._net.synapses.get(nm)
            if s and s.nnz > 0:
                d = s.weights.data
                out[nm] = {"mean": float(d.mean()), "max": float(d.max()), "std": float(d.std())}
        return out

    def run(
        self, patterns: list[dict], training_reps: int = 10, steps_per_pattern: int = 20
    ) -> dict[str, Any]:
        net = self._net
        initial = self._snap()
        for _ in range(training_reps):
            for pat in patterns:
                _inject_multi_step(net, pat, steps_per_pattern)
        final = self._snap()
        wc = {}
        for nm in self.BINDING_GROUPS:
            if nm in initial and nm in final:
                wc[nm] = {
                    "initial_mean": round(initial[nm]["mean"], 6),
                    "final_mean": round(final[nm]["mean"], 6),
                    "delta_mean": round(final[nm]["mean"] - initial[nm]["mean"], 6),
                    "initial_max": round(initial[nm]["max"], 6),
                    "final_max": round(final[nm]["max"], 6),
                    "delta_max": round(final[nm]["max"] - initial[nm]["max"], 6),
                    "initial_std": round(initial[nm]["std"], 6),
                    "final_std": round(final[nm]["std"], 6),
                }
        myel = {}
        for nm in self.BINDING_GROUPS:
            s = net.synapses.get(nm)
            if s and s.plastic and getattr(s, "myelinated", None) is not None:
                myel[nm] = round(float(s.myelinated.sum()) / max(len(s.myelinated), 1), 4)
        return {
            "weight_changes": wc,
            "myelination": myel,
            "concept_count": self._count_concepts(patterns),
            "patterns_trained": len(patterns),
            "training_reps": training_reps,
        }

    def _count_concepts(self, patterns: list[dict]) -> int:
        net = self._net
        if net.concept is None:
            return 0
        vecs = []
        for pat in patterns[:20]:
            c = net.inject_observation(pat["visual"], provenance="sensor.videofile.bench")
            acc = np.zeros(net.concept.n, dtype=np.float32)
            for _ in range(10):
                net.step(c)
                c *= 0.97
                acc += net.concept.spikes.astype(np.float32)
            if acc.any():
                vecs.append(acc)
        if len(vecs) < 2:
            return len(vecs)
        seen = [vecs[0]]
        for v in vecs[1:]:
            nv = np.linalg.norm(v)
            if nv > 0 and all(
                np.dot(v, s) / (nv * np.linalg.norm(s)) <= 0.3
                for s in seen
                if np.linalg.norm(s) > 0
            ):
                seen.append(v)
        return len(seen)


# ---------------------------------------------------------------------------
# Benchmark 4: Energy Efficiency
# ---------------------------------------------------------------------------
class EnergyEfficiencyBenchmark:
    """Spike counts + approximate energy per association."""

    def __init__(self, net: NeuromorphicNetwork) -> None:
        self._net = net

    def run(self, patterns: list[dict], steps_per_pattern: int = 20) -> dict[str, Any]:
        net = self._net
        pat_spikes_list, region_spikes = [], {}
        for pat in patterns:
            c = net.inject_multimodal(
                {"sensor.videofile.bench": pat["visual"], "sensor.audiofile.bench": pat["auditory"]}
            )
            ps = 0
            for _ in range(steps_per_pattern):
                net.step(c)
                c *= 0.97
                for name, reg in net.regions.items():
                    nf = int(reg.spikes.sum())
                    ps += nf
                    region_spikes.setdefault(name, []).append(nf)
            pat_spikes_list.append(ps)
        total_n = net.config.populations.total
        rr = {}
        for name, counts in region_spikes.items():
            n = net.regions[name].n
            if n > 0:
                rr[name] = round(float(np.mean(counts)) / n, 6)
        mss = (
            float(np.mean(pat_spikes_list)) / steps_per_pattern
            if pat_spikes_list and steps_per_pattern
            else 0.0
        )
        region_energy = {
            nm: rr.get(nm, 0.0) * r.n * r.population.params.tau / 20.0
            for nm, r in net.regions.items()
        }
        energy = sum(region_energy.values())
        return {
            "mean_spikes_per_step": round(mss, 2),
            "global_firing_rate": round(mss / total_n if total_n else 0.0, 6),
            "region_firing_rates": rr,
            "region_energy_units": {nm: round(v, 6) for nm, v in region_energy.items()},
            "approx_energy_units": round(energy, 4),
            "spikes_per_association": (
                round(
                    float(np.sum(pat_spikes_list)) / (len(patterns) * max(net.association.n, 1)), 4
                )
                if patterns
                else 0.0
            ),
            "total_steps": len(patterns) * steps_per_pattern,
            "total_neurons": int(total_n),
        }


# ---------------------------------------------------------------------------
# Per-region quiet-vs-sparse cross-reference (issue #331)
#
# #331 asks to join EnergyEfficiencyBenchmark's per-region data against
# "the new per-region benchmarks (items #30-#36)" to tell undertrained
# regions apart from intentionally sparse ones. As of this writing those
# per-region benchmarks (tracked in #297, #315, #316, and siblings) are all
# still open/unimplemented, so there is nothing to join yet. What this
# module *can* do honestly today is cross-reference firing rate against the
# architecture's own documented sparsity design (k-WTA regions), and leave a
# `region_scores` hook so a real per-region learning score can override the
# heuristic the moment one of those benchmarks lands.
# ---------------------------------------------------------------------------

# Regions with a k-WTA / winner-take-all design that intentionally targets a
# very low active fraction (see regions.py ConceptLayer/PatternSeparator and
# their k_winners config, both ~2%) — a low firing rate here is the
# architecture working as designed, not evidence of undertraining.
INTENTIONALLY_SPARSE_REGIONS: dict[str, str] = {
    "concept_layer": (
        "k-WTA bottleneck, ~2% active by design "
        "(50:1 compression from association cortex into sparse distributed representations)"
    ),
    "pattern_separator": (
        "k-WTA dentate-gyrus analog, ~2% active by design "
        "(orthogonal codes for episodic pattern separation)"
    ),
}

# Below this firing rate, a region absent from INTENTIONALLY_SPARSE_REGIONS
# is flagged as quiet-and-unexplained rather than assumed healthy.
DEFAULT_QUIET_THRESHOLD = 0.01


@dataclass(frozen=True)
class RegionQuietFlag:
    """One region's firing-rate classification from classify_quiet_regions()."""

    region: str
    firing_rate: float
    energy_units: float | None
    classification: str  # "intentional_sparsity" | "quiet_undertrained" | "healthy"
    note: str


def classify_quiet_regions(
    energy_efficiency_result: dict[str, Any],
    *,
    quiet_threshold: float = DEFAULT_QUIET_THRESHOLD,
    region_scores: dict[str, float] | None = None,
) -> list[RegionQuietFlag]:
    """Cross-reference per-region firing rates against known sparse-by-design
    regions to separate undertrained-quiet from designed-quiet (issue #331).

    `region_scores` is a hook for a genuine per-region learning-capability
    score (once #297/#315/#316 and siblings land): if present for a region it
    overrides the firing-rate heuristic below, since a region that scores well
    on its own dedicated benchmark is healthy regardless of firing rate. Until
    then this is firing-rate-only triage, not a full learning-capability
    judgment — every note says so explicitly rather than overclaiming.
    """
    rates = energy_efficiency_result.get("region_firing_rates", {}) or {}
    energy = energy_efficiency_result.get("region_energy_units", {}) or {}
    flags = []
    for region, rate in rates.items():
        if region_scores and region in region_scores:
            score = region_scores[region]
            classification = "healthy" if score > 0 else "quiet_undertrained"
            note = f"per-region score={score:.4f} (from a completed per-region benchmark)"
        elif region in INTENTIONALLY_SPARSE_REGIONS:
            classification = "intentional_sparsity"
            note = INTENTIONALLY_SPARSE_REGIONS[region]
        elif rate < quiet_threshold:
            classification = "quiet_undertrained"
            note = (
                f"firing rate {rate:.4f} is below {quiet_threshold:.4f} with no documented "
                "sparse-by-design rationale — likely undertrained, not intentional (pending "
                "#297/#315/#316 and siblings to confirm with a real per-region score)"
            )
        else:
            classification = "healthy"
            note = f"firing rate {rate:.4f} is within the normal range"
        flags.append(
            RegionQuietFlag(
                region=region,
                firing_rate=rate,
                energy_units=energy.get(region),
                classification=classification,
                note=note,
            )
        )
    return flags


def format_quiet_region_report(flags: list[RegionQuietFlag]) -> str:
    """Human-readable rendering of classify_quiet_regions() output."""
    if not flags:
        return "  (no per-region firing-rate data)"
    icons = {"intentional_sparsity": "~", "quiet_undertrained": "!", "healthy": " "}
    lines = []
    for f in sorted(flags, key=lambda f: f.firing_rate):
        icon = icons.get(f.classification, "?")
        energy_str = f"{f.energy_units:.4f}" if f.energy_units is not None else "?"
        lines.append(
            f"  [{icon}] {f.region:22s} rate={f.firing_rate:.4f}  energy={energy_str}  "
            f"{f.classification} — {f.note}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark 5: Concept Separability
# ---------------------------------------------------------------------------
def silhouette_scores_from_distance_matrix(
    dist: np.ndarray,
    labels_arr: np.ndarray,
) -> np.ndarray:
    """Per-sample silhouette coefficient (Rousseeuw 1987) from a precomputed
    distance matrix — pure NumPy, no sklearn dependency.

    s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is the mean distance
    from sample i to the other members of its own cluster, and b(i) is the
    smallest mean distance from i to the members of any other single
    cluster.

    A cluster with only one member has no other in-cluster neighbor, so
    a(i) is undefined for that sample. Rousseeuw's original definition (and
    sklearn.metrics.silhouette_samples, whose source comments this exactly:
    "nan values are for clusters of size 1, and should be 0") sets the
    *whole score* to 0 in that case, rather than deriving it from the
    general formula with a(i)=0 -- the latter would give 1.0 here (i.e.
    "perfectly separated"), which is wrong: a singleton's separation can't
    be assessed at all, so 0 (neither well- nor poorly-clustered) is the
    correct convention, not the maximum score. An earlier version of this
    function used the a(i)=0 shortcut and silently returned 1.0 for
    singleton clusters -- found and fixed via
    scripts/verify_silhouette_score.py (issue #330), which compares this
    function against sklearn.metrics.silhouette_samples on synthetic
    fixtures including a singleton-cluster case.

    Extracted from ConceptSeparabilityBenchmark.run() so the exact shipped
    aggregation logic is what that verification script exercises, not a
    reimplementation of it.
    """
    unique_labels = np.unique(labels_arr)
    n_samples = len(labels_arr)
    sil_scores = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        lbl = labels_arr[i]
        same = labels_arr == lbl
        same[i] = False  # exclude self
        if not same.any():
            # Singleton cluster: score undefined by convention, not derived.
            sil_scores[i] = 0.0
            continue
        a = float(dist[i, same].mean())
        b = min(
            float(dist[i, labels_arr == other_lbl].mean())
            for other_lbl in unique_labels
            if other_lbl != lbl
        )
        denom = max(a, b)
        sil_scores[i] = (b - a) / denom if denom > 0 else 0.0
    return sil_scores


def nearest_centroid_loo_accuracy(
    mat_unit: np.ndarray,
    labels_arr: np.ndarray,
) -> float:
    """Leave-one-out nearest-centroid classification accuracy — pure NumPy.

    For each sample, builds class centroids from every *other* sample (so
    a sample never contributes to its own reference centroid — an unbiased
    probe of separability, not memorization) and classifies it by cosine
    similarity to the nearest centroid. Verified against
    sklearn.neighbors.NearestCentroid + LeaveOneOut in
    scripts/verify_silhouette_score.py (issue #330).

    Extracted from ConceptSeparabilityBenchmark.run(); see its docstring
    there for why this stays pure NumPy.
    """
    unique_labels = np.unique(labels_arr)
    n_classes = len(unique_labels)
    n_samples = len(labels_arr)
    n_features = mat_unit.shape[1]
    class_idx = np.array([int(np.where(unique_labels == lbl)[0][0]) for lbl in labels_arr])
    class_sums = np.zeros((n_classes, n_features), dtype=np.float64)
    class_counts = np.zeros(n_classes, dtype=np.int64)
    for ci, vec in zip(class_idx, mat_unit):
        class_sums[ci] += vec
        class_counts[ci] += 1

    loo_correct = 0
    for i in range(n_samples):
        ci = class_idx[i]
        sims = np.empty(n_classes, dtype=np.float64)
        for j in range(n_classes):
            raw = class_sums[j] - mat_unit[i] if j == ci else class_sums[j]
            cnt = class_counts[j] - 1 if j == ci else class_counts[j]
            raw = raw / max(cnt, 1)
            norm = np.linalg.norm(raw)
            raw_unit = raw / (norm if norm > 0.0 else 1.0)
            sims[j] = mat_unit[i].astype(np.float64) @ raw_unit
        if unique_labels[int(np.argmax(sims))] == labels_arr[i]:
            loo_correct += 1
    return loo_correct / n_samples


class ConceptSeparabilityBenchmark:
    """Score how well concept-layer activations separate distinct stimuli.

    Trains N patterns, accumulates spike-count vectors across multiple probe
    passes per pattern, then computes silhouette score and nearest-centroid
    linear-probe accuracy — both in pure NumPy (no sklearn dependency).

    Metrics returned:
    - silhouette_score      in [-1, 1]; higher = representations are better separated
    - linear_probe_accuracy nearest-centroid accuracy in [0, 1]
    - mean_intra_class_distance / mean_inter_class_distance (cosine)
    - separation_ratio      mean_inter / (mean_intra + 1e-8)
    """

    def __init__(self, net: NeuromorphicNetwork) -> None:
        self._net = net

    def run(
        self,
        patterns: list[dict],
        training_reps: int = 5,
        probe_reps: int = 3,
        steps_per_rep: int = 10,
    ) -> dict[str, Any]:
        net = self._net
        if net.concept is None:
            return {
                "error": "no concept layer",
                "silhouette_score": 0.0,
                "linear_probe_accuracy": 0.0,
                "n_patterns": len(patterns),
            }

        # Training phase — expose each pattern so STDP builds concept codes
        for _ in range(training_reps):
            for pat in patterns:
                c = net.inject_observation(pat["visual"], provenance="sensor.videofile.bench")
                for _ in range(steps_per_rep):
                    net.step(c)
                    c = c * np.float32(0.97)

        # Probe phase — collect spike-count vectors, one per (pattern, rep)
        labels: list[int] = []
        vecs: list[np.ndarray] = []
        for idx, pat in enumerate(patterns):
            for _ in range(probe_reps):
                acc = np.zeros(net.concept.n, dtype=np.float32)
                c = net.inject_observation(pat["visual"], provenance="sensor.videofile.bench")
                for _ in range(steps_per_rep):
                    net.step(c)
                    c = c * np.float32(0.97)
                    acc += net.concept.spikes.astype(np.float32)
                labels.append(idx)
                vecs.append(acc)

        labels_arr = np.array(labels, dtype=np.int32)

        # Guard before any matrix operations so empty inputs don't crash norm/shape ops
        n_samples = len(labels)
        unique_labels = np.unique(labels_arr)
        n_classes = len(unique_labels)

        if n_classes < 2 or n_samples < 4:
            return {
                "error": "insufficient patterns for separability",
                "silhouette_score": 0.0,
                "linear_probe_accuracy": 0.0,
                "n_patterns": len(patterns),
            }

        mat = np.array(vecs, dtype=np.float32)  # (n_samples, concept_n)

        # L2-normalise rows for cosine distance
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_unit = mat / norms

        # Cosine distance matrix: D[i,j] = 1 − cos_similarity
        sim = mat_unit @ mat_unit.T
        dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)
        np.fill_diagonal(dist, 0.0)

        # Silhouette score (pure NumPy)
        sil_scores = silhouette_scores_from_distance_matrix(dist, labels_arr)
        silhouette = round(float(np.mean(sil_scores)), 4)

        # Mean intra / inter class cosine distances
        intra: list[float] = []
        inter: list[float] = []
        for i in range(n_samples):
            same_mask = labels_arr == labels_arr[i]
            same_mask_excl = same_mask.copy()
            same_mask_excl[i] = False
            if same_mask_excl.any():
                intra.extend(dist[i, same_mask_excl].tolist())
            other_mask = ~same_mask
            if other_mask.any():
                inter.extend(dist[i, other_mask].tolist())

        raw_intra = float(np.mean(intra)) if intra else 0.0
        raw_inter = float(np.mean(inter)) if inter else 0.0
        mean_intra = round(raw_intra, 4)
        mean_inter = round(raw_inter, 4)
        separation_ratio = round(raw_inter / (raw_intra + 1e-8), 4)

        # Leave-one-out nearest-centroid linear probe (no sklearn, unbiased)
        accuracy = round(nearest_centroid_loo_accuracy(mat_unit, labels_arr), 4)

        # Top concept neurons per pattern (most selective on average)
        raw_centroids = np.array(
            [mat[labels_arr == lbl].mean(axis=0) for lbl in unique_labels],
            dtype=np.float32,
        )
        top_neurons = [np.argsort(row)[-5:][::-1].tolist() for row in raw_centroids]

        return {
            "silhouette_score": silhouette,
            "linear_probe_accuracy": accuracy,
            "mean_intra_class_distance": mean_intra,
            "mean_inter_class_distance": mean_inter,
            "separation_ratio": separation_ratio,
            "n_patterns": len(patterns),
            "n_samples": n_samples,
            "concept_neurons": int(net.concept.n),
            "top_neurons_per_pattern": top_neurons,
            "training_reps": training_reps,
            "probe_reps": probe_reps,
        }


# ---------------------------------------------------------------------------
# Benchmark 6: Cross-Modal Binding Accuracy
# ---------------------------------------------------------------------------
def _pair_coupling_score(
    net: NeuromorphicNetwork,
    visual: list[float] | np.ndarray,
    auditory: list[float] | np.ndarray,
    probe: CrossModalProbe,
) -> float:
    """Weight-based coupling between a visual and auditory pattern."""
    net.inject_multimodal(
        {
            "sensor.videofile.bench": visual,
            "sensor.audiofile.bench": auditory,
        }
    )
    vis_range = net.allocator.current_ranges.get("visual", (0, 0))
    aud_range = net.allocator.current_ranges.get("auditory", (0, 0))
    if vis_range[1] <= vis_range[0] or aud_range[1] <= aud_range[0]:
        return 0.0

    vis_current = net.encoder.encode(
        net.sensory,
        visual,
        "sensor.videofile.bench",
        net.allocator,
    )
    aud_current = net.encoder.encode(
        net.sensory,
        auditory,
        "sensor.audiofile.bench",
        net.allocator,
    )
    vis_mask = vis_current[vis_range[0] : vis_range[1]]
    aud_mask = aud_current[aud_range[0] : aud_range[1]]
    if vis_mask.sum() > 0:
        vis_mask = vis_mask / vis_mask.sum()
    if aud_mask.sum() > 0:
        aud_mask = aud_mask / aud_mask.sum()

    sa = net.synapses.get("sensory_association")
    if sa is None or sa.nnz == 0:
        return 0.0

    v2a_per = np.asarray(sa.weights[:, vis_range[0] : vis_range[1]] @ vis_mask).ravel()
    a2v_per = np.asarray(sa.weights[:, aud_range[0] : aud_range[1]] @ aud_mask).ravel()
    inputs = probe._extract_inputs(net)
    if inputs is None:
        return 0.0
    probe.probe(inputs)

    v2a = 0.0
    if len(probe._auditory_assoc_idx) > 0:
        v2a = float(v2a_per[probe._auditory_assoc_idx].mean())
    a2v = 0.0
    if len(probe._visual_assoc_idx) > 0:
        a2v = float(a2v_per[probe._visual_assoc_idx].mean())
    nonzero = (v2a > 0) + (a2v > 0)
    return (v2a + a2v) / nonzero if nonzero else 0.0


def _binding_precision_recall(
    correlated_pairs: list[dict[str, Any]],
    coupling_matrix: list[list[float]],
    detection_threshold: float = 1e-6,
) -> dict[str, float]:
    """Compute precision/recall for top-1 pair binding predictions."""
    n = len(correlated_pairs)
    tp = fp = fn = 0
    for i in range(n):
        scores = coupling_matrix[i]
        pred_j = int(np.argmax(scores))
        best_score = scores[pred_j]
        if best_score < detection_threshold:
            fn += 1
            continue
        if pred_j == i:
            tp += 1
        else:
            fp += 1
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


class CrossModalBindingAccuracyBenchmark:
    """Measure binding accuracy against ground-truth correlated stimulus pairs."""

    def __init__(self, net: NeuromorphicNetwork) -> None:
        self._net = net
        self._probe = CrossModalProbe()

    def run(
        self,
        n_pairs: int = 4,
        training_reps: int = 30,
        steps_per_pair: int = 25,
        seed: int = 42,
    ) -> dict[str, Any]:
        net, probe = self._net, self._probe
        n_pairs = max(2, n_pairs)
        fixtures = generate_correlated_stimulus_fixtures(n_pairs, seed=seed)
        correlated = fixtures["correlated_pairs"]
        decoys = fixtures["decoy_pairs"]

        pre = probe.probe_network(net).to_dict()
        for _ in range(training_reps):
            for pair in correlated:
                _inject_multi_step(net, pair, steps_per_pair)
        post = probe.probe_network(net).to_dict()

        n = len(correlated)
        raw_coupling_matrix: list[list[float]] = []
        coupling_matrix: list[list[float]] = []
        matched_scores: list[float] = []
        decoy_scores: list[float] = []
        for i, pair in enumerate(correlated):
            row = [
                _pair_coupling_score(net, pair["visual"], correlated[j]["auditory"], probe)
                for j in range(n)
            ]
            raw_coupling_matrix.append(row)
            coupling_matrix.append([round(s, 8) for s in row])
            matched_scores.append(row[i])
            for j, decoy in enumerate(decoys):
                if decoy["visual_pair_id"] == pair["pair_id"]:
                    decoy_scores.append(
                        _pair_coupling_score(net, decoy["visual"], decoy["auditory"], probe),
                    )

        pr = _binding_precision_recall(correlated, raw_coupling_matrix)
        matched_mean = float(np.mean(matched_scores)) if matched_scores else 0.0
        decoy_mean = float(np.mean(decoy_scores)) if decoy_scores else 0.0
        ratio = matched_mean / (decoy_mean + 1e-9)

        return {
            "precision": pr["precision"],
            "recall": pr["recall"],
            "f1": pr["f1"],
            "true_positives": pr["true_positives"],
            "false_positives": pr["false_positives"],
            "false_negatives": pr["false_negatives"],
            "binding_strength_before": pre.get("binding_strength", 0.0),
            "binding_strength_after": post.get("binding_strength", 0.0),
            "binding_strength_delta": round(
                post.get("binding_strength", 0.0) - pre.get("binding_strength", 0.0),
                6,
            ),
            "n_cross_modal_before": pre.get("n_cross_modal", 0),
            "n_cross_modal_after": post.get("n_cross_modal", 0),
            "matched_coupling_mean": round(matched_mean, 8),
            "decoy_coupling_mean": round(decoy_mean, 8),
            "matched_to_decoy_ratio": round(ratio, 4),
            "pairs_tested": n_pairs,
            "training_reps": training_reps,
            "fixture_seed": seed,
            "coupling_matrix": coupling_matrix,
        }


# ---------------------------------------------------------------------------
# Per-region benchmark: MetaControllerRegion gating (issue #319)
#
# Not part of BenchmarkSuite.run_all()'s fixed 6-metric contract (see
# docs/benchmark-schema.md) — this is a standalone per-region benchmark, one
# of the items #30-#36 batch referenced in docs/REGION-BENCHMARK-OBSERVABILITY
# -ALIGNMENT.md, added independently so it doesn't change run_all()'s stable
# output schema.
# ---------------------------------------------------------------------------
def _standardized_nearest_centroid_loo_accuracy(mat: np.ndarray, labels_arr: np.ndarray) -> float:
    """Leave-one-out nearest-centroid accuracy over z-scored Euclidean features.

    Neuromodulator readouts have different physical ranges (DA in [0.2, 2.5],
    ACh in [0.3, 2.0], NE in [0.5, 3.0], 5-HT in [0.1, 1.5] by default — see
    MetaControllerConfig), so a raw Euclidean distance would let NE dominate
    the classification purely from its wider range. Standardizing each column
    to zero mean / unit std first (using the full sample; leakage across the
    2-4 held-out points is negligible relative to the class centroids) puts
    all four signals on equal footing before distance is computed.
    """
    std = mat.std(axis=0)
    std[std == 0.0] = 1.0
    z = (mat - mat.mean(axis=0)) / std

    unique_labels = np.unique(labels_arr)
    n_classes = len(unique_labels)
    n_features = z.shape[1]
    class_idx = np.array([int(np.where(unique_labels == lbl)[0][0]) for lbl in labels_arr])
    class_sums = np.zeros((n_classes, n_features), dtype=np.float64)
    class_counts = np.zeros(n_classes, dtype=np.int64)
    for ci, vec in zip(class_idx, z):
        class_sums[ci] += vec
        class_counts[ci] += 1

    correct = 0
    for i in range(len(labels_arr)):
        ci = class_idx[i]
        dists = np.empty(n_classes, dtype=np.float64)
        for j in range(n_classes):
            raw = class_sums[j] - z[i] if j == ci else class_sums[j]
            cnt = class_counts[j] - 1 if j == ci else class_counts[j]
            centroid = raw / max(cnt, 1)
            dists[j] = np.linalg.norm(z[i] - centroid)
        if unique_labels[int(np.argmin(dists))] == labels_arr[i]:
            correct += 1
    return correct / len(labels_arr)


class MetaControllerGatingBenchmark:
    """Task-switching benchmark for MetaControllerRegion (issue #319).

    ``MetaControllerRegion`` is documented (MetaControllerConfig) as "a
    neuromodulatory hub for plasticity gating": it monitors network-wide
    firing statistics and outputs 4 neuromodulatory signals (DA/ACh/NE/5-HT)
    that gate learning and processing elsewhere in the network. It is not a
    discrete action-selector — there is no single "gate" output to compare
    against a hand-labeled correct choice the way, e.g., a motor command
    would be. What *is* concretely testable against ground truth is whether
    its neuromodulatory output actually reflects which task context is
    driving the network right now, i.e. whether the gate is doing anything
    at all rather than emitting noise indistinguishable across contexts —
    exactly the blind spot the issue flags ("could be doing nothing useful
    without anyone noticing").

    This runs a task-switching protocol: a randomized sequence of trials each
    presenting one of two distinguishable stimulus contexts, and scores
    leave-one-out nearest-centroid classification accuracy of the *known*
    active context from ``read_neuromodulators()``'s 4-dim output alone. That
    accuracy is the ground-truth-checkable "gating decision" — well above
    chance (0.5 for 2 contexts) means the region's output reliably encodes
    which context is active; at-chance means it is not gating anything.
    """

    def __init__(self, net: NeuromorphicNetwork) -> None:
        self._net = net

    def run(
        self,
        context_a: dict,
        context_b: dict,
        n_trials: int = 20,
        steps_per_trial: int = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        net = self._net
        if net.meta_ctrl is None:
            return {
                "error": "no meta_controller region",
                "gating_accuracy": 0.0,
                "n_trials": n_trials,
            }

        rng = np.random.default_rng(seed)
        contexts = [context_a, context_b]
        sequence = rng.integers(0, 2, size=n_trials)

        labels: list[int] = []
        neuromod_vecs: list[list[float]] = []
        da_by_context: dict[int, list[float]] = {0: [], 1: []}
        ach_by_context: dict[int, list[float]] = {0: [], 1: []}
        ne_by_context: dict[int, list[float]] = {0: [], 1: []}
        serotonin_by_context: dict[int, list[float]] = {0: [], 1: []}
        for ctx_idx in sequence:
            ctx_idx = int(ctx_idx)
            _inject_multi_step(net, contexts[ctx_idx], steps_per_trial)
            nm = net.meta_ctrl.read_neuromodulators()
            labels.append(ctx_idx)
            neuromod_vecs.append([nm["da"], nm["ach"], nm["ne"], nm["serotonin"]])
            da_by_context[ctx_idx].append(nm["da"])
            ach_by_context[ctx_idx].append(nm["ach"])
            ne_by_context[ctx_idx].append(nm["ne"])
            serotonin_by_context[ctx_idx].append(nm["serotonin"])

        labels_arr = np.array(labels, dtype=np.int32)
        n_samples = len(labels)
        n_classes = len(np.unique(labels_arr))

        if n_classes < 2 or n_samples < 4:
            return {
                "error": "insufficient context switches for gating evaluation",
                "gating_accuracy": 0.0,
                "n_trials": n_trials,
            }

        mat = np.array(neuromod_vecs, dtype=np.float64)
        accuracy = round(_standardized_nearest_centroid_loo_accuracy(mat, labels_arr), 4)

        def _mean_or_zero(vals: list[float]) -> float:
            return round(float(np.mean(vals)), 6) if vals else 0.0

        return {
            "gating_accuracy": accuracy,
            "n_trials": n_trials,
            "n_samples": n_samples,
            "context_switches": int(np.sum(sequence[1:] != sequence[:-1])),
            "meta_controller_neurons": int(net.meta_ctrl.n),
            "mean_da_context_a": _mean_or_zero(da_by_context[0]),
            "mean_da_context_b": _mean_or_zero(da_by_context[1]),
            "mean_ach_context_a": _mean_or_zero(ach_by_context[0]),
            "mean_ach_context_b": _mean_or_zero(ach_by_context[1]),
            "mean_ne_context_a": _mean_or_zero(ne_by_context[0]),
            "mean_ne_context_b": _mean_or_zero(ne_by_context[1]),
            "mean_serotonin_context_a": _mean_or_zero(serotonin_by_context[0]),
            "mean_serotonin_context_b": _mean_or_zero(serotonin_by_context[1]),
            "steps_per_trial": steps_per_trial,
            "fixture_seed": seed,
        }


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------
class BenchmarkSuite:
    """Runs all 6 benchmarks and produces a unified results dict."""

    def __init__(self, network: NeuromorphicNetwork) -> None:
        self.network = network

    def run_all(
        self,
        n_patterns: int = 20,
        training_reps: int = 10,
        steps_per_pattern: int = 20,
        seed: int = 42,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        patterns = generate_test_patterns(n_patterns, rng)
        t0 = time.perf_counter()
        logger.info(
            "Benchmark 1/6: CrossModalRecall (%d patterns x %d reps)", n_patterns, training_reps
        )
        cm = CrossModalRecallBenchmark(self.network).run(patterns, training_reps, steps_per_pattern)
        logger.info("Benchmark 2/6: NoveltyDetection")
        nd = NoveltyDetectionBenchmark(self.network).run(
            patterns[0],
            generate_test_patterns(1, np.random.default_rng(seed + 999))[0],
            training_reps,
            steps_per_pattern,
        )
        logger.info("Benchmark 3/6: AssociationStrength")
        ass = AssociationStrengthBenchmark(self.network).run(
            patterns, training_reps, steps_per_pattern
        )
        logger.info("Benchmark 4/6: EnergyEfficiency")
        en = EnergyEfficiencyBenchmark(self.network).run(patterns, steps_per_pattern)
        logger.info("Benchmark 5/6: ConceptSeparability")
        cs = ConceptSeparabilityBenchmark(self.network).run(
            patterns, training_reps=training_reps, steps_per_rep=steps_per_pattern
        )
        logger.info("Benchmark 6/6: CrossModalBindingAccuracy")
        ba = CrossModalBindingAccuracyBenchmark(self.network).run(
            n_pairs=max(2, min(n_patterns, 8)),
            training_reps=training_reps,
            steps_per_pair=steps_per_pattern,
            seed=seed,
        )
        return _to_native(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step_count": self.network.step_count,
                "total_neurons": int(self.network.config.populations.total),
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "cross_modal_recall": cm,
                "novelty_detection": nd,
                "association_strength": ass,
                "energy_efficiency": en,
                "concept_separability": cs,
                "cross_modal_binding_accuracy": ba,
            }
        )

    def run_multi_seed(
        self,
        n_seeds: int = 5,
        n_patterns: int = 20,
        training_reps: int = 10,
        steps_per_pattern: int = 20,
        base_seed: int = 42,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Run the full suite across n_seeds distinct seeds and aggregate per-metric stats.

        A single-seed score can't distinguish "the network learned this" from
        "this particular initialization got lucky" — this runs the suite N
        times with distinct seeds and reports a mean + confidence interval for
        every numeric metric, so claims about performance are defensible.
        """
        seeds = [base_seed + i for i in range(n_seeds)]
        t0 = time.perf_counter()
        runs = []
        for i, seed in enumerate(seeds):
            logger.info("Multi-seed run %d/%d (seed=%d)", i + 1, n_seeds, seed)
            runs.append(self.run_all(n_patterns, training_reps, steps_per_pattern, seed))

        flattened = [_flatten_numeric(r) for r in runs]
        metric_names = sorted({name for f in flattened for name in f})
        aggregate = {}
        for name in metric_names:
            values = [f[name] for f in flattened if name in f]
            if not values:
                continue
            mean, half_width = _confidence_interval(values, confidence)
            aggregate[name] = {
                "mean": round(mean, 6),
                "std": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0,
                "ci_low": round(mean - half_width, 6),
                "ci_high": round(mean + half_width, 6),
                "n": len(values),
            }

        return _to_native(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_seeds": n_seeds,
                "seeds": seeds,
                "confidence": confidence,
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "aggregate": aggregate,
                "runs": runs,
            }
        )

    @staticmethod
    def check_runtime_budget(elapsed_s: float, budget_s: float) -> list[str]:
        """Return failure messages if elapsed_s exceeds budget_s; empty list on pass."""
        if elapsed_s > budget_s:
            return [
                f"BenchmarkSuite.run_all() took {elapsed_s:.2f}s, "
                f"exceeding the budget of {budget_s:.2f}s. "
                "Fix the regression or update benchmarks/suite_runtime_budget.json "
                "(with a new measured_baseline_s, not just a wider budget_s)."
            ]
        return []

    def save_results(self, results: dict[str, Any], output_dir: str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"benchmarks_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(results, indent=2))
        logger.info("Benchmark results saved: %s", path)
        return path

    @staticmethod
    def summary(results: dict[str, Any]) -> str:
        lines = [
            "=== Engram Benchmark Summary ===",
            f"Step: {results.get('step_count', '?'):,}  |  "
            f"Neurons: {results.get('total_neurons', '?'):,}  |  "
            f"Time: {results.get('elapsed_s', '?')}s",
            "",
        ]
        cm = results.get("cross_modal_recall", {})
        lines += [
            "1. Cross-Modal Recall",
            f"   V->A recall: {cm.get('visual_to_auditory_recall', 0):.4f}",
            f"   A->V recall: {cm.get('auditory_to_visual_recall', 0):.4f}",
            f"   Binding delta: {cm.get('binding_strength_delta', 0):+.6f}",
            "",
        ]
        nd = results.get("novelty_detection", {})
        lines += [
            "2. Novelty Detection",
            f"   Familiar error: {nd.get('familiar_pred_error', 0):.4f}",
            f"   Novel error:    {nd.get('novel_pred_error', 0):.4f}",
            f"   Discrimination: {nd.get('discrimination_ratio', 0):.2f}x",
            "",
        ]
        a = results.get("association_strength", {})
        lines.append("3. Association Strength")
        for gn, wc in a.get("weight_changes", {}).items():
            lines.append(
                f"   {gn}: {wc.get('initial_mean',0):.4f} -> "
                f"{wc.get('final_mean',0):.4f} ({wc.get('delta_mean',0):+.6f})"
            )
        lines += [f"   Concepts: {a.get('concept_count', 0)}", ""]
        en = results.get("energy_efficiency", {})
        lines += [
            "4. Energy Efficiency",
            f"   Spikes/step: {en.get('mean_spikes_per_step', 0):.0f}",
            f"   Global rate:  {en.get('global_firing_rate', 0):.6f}",
            f"   Energy units: {en.get('approx_energy_units', 0):.2f}",
        ]
        if en.get("region_firing_rates"):
            lines.append("   Per-region (issue #331 — quiet-vs-sparse triage, no per-region")
            lines.append("   learning score yet; see #297/#315/#316):")
            lines.append(format_quiet_region_report(classify_quiet_regions(en)))
        lines.append("")
        cs = results.get("concept_separability", {})
        if cs:
            if "error" not in cs:
                lines += [
                    "5. Concept Separability",
                    f"   Silhouette score: {cs.get('silhouette_score', 0):.4f}",
                    f"   Linear-probe acc: {cs.get('linear_probe_accuracy', 0):.4f}",
                    f"   Separation ratio: {cs.get('separation_ratio', 0):.2f}x",
                    f"   Intra/inter dist: {cs.get('mean_intra_class_distance', 0):.4f} / "
                    f"{cs.get('mean_inter_class_distance', 0):.4f}",
                    "",
                ]
            else:
                lines += [
                    "5. Concept Separability",
                    f"   (skipped — {cs['error']})",
                    "",
                ]
        ba = results.get("cross_modal_binding_accuracy", {})
        lines += [
            "6. Cross-Modal Binding Accuracy",
            f"   Precision: {ba.get('precision', 0):.4f}",
            f"   Recall:    {ba.get('recall', 0):.4f}",
            f"   F1:        {ba.get('f1', 0):.4f}",
            f"   Matched/decoy ratio: {ba.get('matched_to_decoy_ratio', 0):.2f}x",
            "",
        ]
        lines.append("=" * 35)
        return "\n".join(lines)

    @staticmethod
    def summary_multi_seed(results: dict[str, Any]) -> str:
        """Human-readable mean + confidence-interval summary for a multi-seed run."""
        agg = results.get("aggregate", {})
        lines = [
            "=== Engram Multi-Seed Benchmark Summary ===",
            f"Seeds: {results.get('n_seeds', '?')} {results.get('seeds', [])}  |  "
            f"Confidence: {results.get('confidence', 0.95):.0%}  |  "
            f"Time: {results.get('elapsed_s', '?')}s",
            "",
        ]
        headline_metrics = [
            "cross_modal_recall.visual_to_auditory_recall",
            "cross_modal_recall.auditory_to_visual_recall",
            "cross_modal_recall.binding_strength_delta",
            "novelty_detection.discrimination_ratio",
            "association_strength.concept_count",
            "energy_efficiency.global_firing_rate",
        ]
        for name in headline_metrics:
            stat = agg.get(name)
            if not stat:
                continue
            lines.append(
                f"  {name}: {stat['mean']:.4f}  "
                f"(95% CI: [{stat['ci_low']:.4f}, {stat['ci_high']:.4f}], n={stat['n']})"
            )
        lines += [
            "",
            f"  {len(agg)} metrics tracked across seeds (full detail in saved JSON)",
            "",
            "=" * 45,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: python -m neuromorphic.benchmarks
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m neuromorphic.benchmarks` CLI parser.

    Factored out of ``main()`` so the flag wiring (issue #322: overriding
    the suite's seed from the command line, threaded through to
    ``run_all()`` and ``binding_fixtures.generate_correlated_stimulus_fixtures()``)
    is directly testable without spinning up a real network.
    """
    parser = argparse.ArgumentParser(description="Engram investor-ready benchmarks")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="benchmarks/")
    parser.add_argument("--patterns", type=int, default=20)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of distinct seeds to run (default: 1). When > 1, runs the "
        "suite once per seed and reports mean + confidence interval per "
        "metric instead of a single-run summary.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for multi-seed intervals (default: 0.95)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from neuromorphic.config import NeuromorphicConfig
    from neuromorphic.network import NeuromorphicNetwork

    config = NeuromorphicConfig.from_env()
    print(f"Initializing network: {config.populations.total:,} neurons")
    network = NeuromorphicNetwork(config)
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        import asyncio

        from neuromorphic.persistence import NeuromorphicPersistence

        async def _load():
            p = NeuromorphicPersistence(args.checkpoint)
            await p.open()
            state = await p.load_state()
            await p.close()
            return state

        state = asyncio.run(_load())
        if state:
            network.set_state(state)
            print(f"  Restored at step {network.step_count:,}")
    suite = BenchmarkSuite(network)
    if args.seeds > 1:
        results = suite.run_multi_seed(
            args.seeds,
            args.patterns,
            args.reps,
            args.steps,
            base_seed=args.seed,
            confidence=args.confidence,
        )
        path = suite.save_results(results, args.output)
        print(suite.summary_multi_seed(results))
    else:
        results = suite.run_all(args.patterns, args.reps, args.steps, args.seed)
        path = suite.save_results(results, args.output)
        print(suite.summary(results))
    print(f"\nResults saved: {path}")


if __name__ == "__main__":
    main()
