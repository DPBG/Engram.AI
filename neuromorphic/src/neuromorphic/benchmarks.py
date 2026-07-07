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

    # Or run from checkpoint on server:
    cd neuromorphic && uv run python -m neuromorphic.benchmarks --checkpoint /data/sqlite/neuromorphic.db
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from neuromorphic.binding_fixtures import generate_correlated_stimulus_fixtures
from neuromorphic.cross_modal_probe import CrossModalProbe

if TYPE_CHECKING:
    from neuromorphic.network import NeuromorphicNetwork

logger = logging.getLogger(__name__)


def generate_test_patterns(n: int, rng: np.random.Generator) -> list[dict]:
    """Visual gaussian blobs + MFCC-like auditory signatures."""
    y, x = np.mgrid[0:64, 0:64]
    patterns = []
    for i in range(n):
        d2 = (x - (i * 7) % 64).astype(np.float32) ** 2 + (y - (i * 13) % 64).astype(
            np.float32
        ) ** 2
        vis = np.exp(-d2 / 50.0, dtype=np.float32).flatten()
        vis /= vis.max() + 1e-8
        aud = rng.normal(0.5, 0.1, size=13).astype(np.float32)
        aud[i % 13] = 1.0
        np.clip(aud, 0.0, 1.0, out=aud)
        patterns.append(
            {"visual": vis.tolist(), "auditory": aud.tolist(), "label": f"pattern_{i:03d}"}
        )
    return patterns


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
        energy = sum(
            rr.get(nm, 0.0) * r.n * r.population.params.tau / 20.0 for nm, r in net.regions.items()
        )
        return {
            "mean_spikes_per_step": round(mss, 2),
            "global_firing_rate": round(mss / total_n if total_n else 0.0, 6),
            "region_firing_rates": rr,
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
# Benchmark 5: Concept Separability
# ---------------------------------------------------------------------------
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
        sil_scores: list[float] = []
        for i in range(n_samples):
            lbl = labels_arr[i]
            same = labels_arr == lbl
            same[i] = False  # exclude self
            a = float(dist[i, same].mean()) if same.any() else 0.0
            b = min(
                float(dist[i, labels_arr == other_lbl].mean())
                for other_lbl in unique_labels
                if other_lbl != lbl
            )
            denom = max(a, b)
            sil_scores.append((b - a) / denom if denom > 0 else 0.0)

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
        # Pre-compute class sums so each LOO centroid is O(1) to derive
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
        accuracy = round(loo_correct / n_samples, 4)

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
            "",
        ]
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


# ---------------------------------------------------------------------------
# CLI: python -m neuromorphic.benchmarks
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Engram investor-ready benchmarks")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="benchmarks/")
    parser.add_argument("--patterns", type=int, default=20)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
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
    results = suite.run_all(args.patterns, args.reps, args.steps)
    path = suite.save_results(results, args.output)
    print(suite.summary(results))
    print(f"\nResults saved: {path}")


if __name__ == "__main__":
    main()
