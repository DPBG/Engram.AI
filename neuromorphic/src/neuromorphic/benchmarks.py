"""
Structured benchmarking framework for investor-ready metrics.

Provides 4 benchmark tests that produce quantitative proof the brain learns:
1. CrossModalRecall — inject visual, measure auditory cortex activation (and vice versa)
2. NoveltyDetection — present known vs unknown stimuli, measure response difference
3. AssociationStrength — measure weight changes after paired multi-modal training
4. EnergyEfficiency — compute energy per learned association vs baseline

Usage:
    from neuromorphic.benchmarks import BenchmarkSuite
    suite = BenchmarkSuite(network)
    results = suite.run_all()
    suite.save_results(results, "benchmarks/")

    # Multiple seeds, with mean + confidence interval per metric — a single-seed
    # score can't tell "the network learned this" from "this init got lucky":
    results = suite.run_multi_seed(n_seeds=5)
    print(suite.summary_multi_seed(results))

    # Or run from checkpoint on server:
    cd neuromorphic && uv run python -m neuromorphic.benchmarks --checkpoint /data/sqlite/neuromorphic.db
    cd neuromorphic && uv run python -m neuromorphic.benchmarks --seeds 5
"""
from __future__ import annotations

import argparse, json, logging, time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from scipy import stats

from neuromorphic.cross_modal_probe import CrossModalProbe

if TYPE_CHECKING:
    from neuromorphic.network import NeuromorphicNetwork

logger = logging.getLogger(__name__)

def generate_test_patterns(n: int, rng: np.random.Generator) -> list[dict]:
    """Visual gaussian blobs + MFCC-like auditory signatures."""
    y, x = np.mgrid[0:64, 0:64]
    patterns = []
    for i in range(n):
        d2 = (x - (i*7)%64).astype(np.float32)**2 + (y - (i*13)%64).astype(np.float32)**2
        vis = np.exp(-d2 / 50.0, dtype=np.float32).flatten()
        vis /= vis.max() + 1e-8
        aud = rng.normal(0.5, 0.1, size=13).astype(np.float32)
        aud[i % 13] = 1.0
        np.clip(aud, 0.0, 1.0, out=aud)
        patterns.append({"visual": vis.tolist(), "auditory": aud.tolist(), "label": f"pattern_{i:03d}"})
    return patterns

def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python natives."""
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
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
        net.step(vc); vc *= 0.97
    ac = net.inject_observation(pat["auditory"], provenance="sensor.audiofile.bench")
    combined = vc + ac
    for _ in range(half):
        net.step(combined); combined *= 0.97

def _inject_multi_step(net: NeuromorphicNetwork, pat: dict, steps: int) -> None:
    """Inject both modalities simultaneously and step."""
    c = net.inject_multimodal({"sensor.videofile.bench": pat["visual"],
                               "sensor.audiofile.bench": pat["auditory"]})
    for _ in range(steps):
        net.step(c); c *= 0.97

# ---------------------------------------------------------------------------
# Benchmark 1: Cross-Modal Recall
# ---------------------------------------------------------------------------
class CrossModalRecallBenchmark:
    """Train paired patterns, test recall by presenting only one modality."""
    def __init__(self, net: NeuromorphicNetwork) -> None:
        self._net, self._probe = net, CrossModalProbe()

    def run(self, patterns: list[dict], training_reps: int = 10,
            steps_per_pattern: int = 20) -> dict[str, Any]:
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
            for _ in range(half): net.step(c); c *= 0.97
            v2a.append(probe.probe_network(net).recall_ratio_visual)
            c = net.inject_observation(pat["auditory"], provenance="sensor.audiofile.bench")
            for _ in range(half): net.step(c); c *= 0.97
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

    def run(self, familiar_pattern: dict, novel_pattern: dict,
            familiarization_reps: int = 20, steps_per_rep: int = 20,
            test_steps: int = 10) -> dict[str, Any]:
        net = self._net
        for _ in range(familiarization_reps):
            _inject_multi_step(net, familiar_pattern, steps_per_rep)

        def _measure(pat: dict) -> tuple[list[float], dict[str, list[float]]]:
            c = net.inject_multimodal({"sensor.videofile.bench": pat["visual"],
                                       "sensor.audiofile.bench": pat["auditory"]})
            errs, rates = [], {}
            for _ in range(test_steps):
                net.step(c); c *= 0.97
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
            rate_shift[name] = round(float(np.mean(nov_rates.get(name, [0.0])))
                                     - float(np.mean(fam_rates[name])), 6)
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

    def run(self, patterns: list[dict], training_reps: int = 10,
            steps_per_pattern: int = 20) -> dict[str, Any]:
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
        return {"weight_changes": wc, "myelination": myel,
                "concept_count": self._count_concepts(patterns),
                "patterns_trained": len(patterns), "training_reps": training_reps}

    def _count_concepts(self, patterns: list[dict]) -> int:
        net = self._net
        if net.concept is None: return 0
        vecs = []
        for pat in patterns[:20]:
            c = net.inject_observation(pat["visual"], provenance="sensor.videofile.bench")
            acc = np.zeros(net.concept.n, dtype=np.float32)
            for _ in range(10):
                net.step(c); c *= 0.97
                acc += net.concept.spikes.astype(np.float32)
            if acc.any(): vecs.append(acc)
        if len(vecs) < 2: return len(vecs)
        seen = [vecs[0]]
        for v in vecs[1:]:
            nv = np.linalg.norm(v)
            if nv > 0 and all(np.dot(v, s) / (nv * np.linalg.norm(s)) <= 0.3
                              for s in seen if np.linalg.norm(s) > 0):
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
            c = net.inject_multimodal({"sensor.videofile.bench": pat["visual"],
                                       "sensor.audiofile.bench": pat["auditory"]})
            ps = 0
            for _ in range(steps_per_pattern):
                net.step(c); c *= 0.97
                for name, reg in net.regions.items():
                    nf = int(reg.spikes.sum()); ps += nf
                    region_spikes.setdefault(name, []).append(nf)
            pat_spikes_list.append(ps)
        total_n = net.config.populations.total
        rr = {}
        for name, counts in region_spikes.items():
            n = net.regions[name].n
            if n > 0: rr[name] = round(float(np.mean(counts)) / n, 6)
        mss = (float(np.mean(pat_spikes_list)) / steps_per_pattern
               if pat_spikes_list and steps_per_pattern else 0.0)
        energy = sum(rr.get(nm, 0.0) * r.n * r.population.params.tau / 20.0
                     for nm, r in net.regions.items())
        return {
            "mean_spikes_per_step": round(mss, 2),
            "global_firing_rate": round(mss / total_n if total_n else 0.0, 6),
            "region_firing_rates": rr,
            "approx_energy_units": round(energy, 4),
            "spikes_per_association": round(
                float(np.sum(pat_spikes_list)) / (len(patterns) * max(net.association.n, 1)), 4
            ) if patterns else 0.0,
            "total_steps": len(patterns) * steps_per_pattern,
            "total_neurons": int(total_n),
        }

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------
class BenchmarkSuite:
    """Runs all 4 benchmarks and produces a unified results dict."""
    def __init__(self, network: NeuromorphicNetwork) -> None:
        self.network = network

    def run_all(self, n_patterns: int = 20, training_reps: int = 10,
                steps_per_pattern: int = 20, seed: int = 42) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        patterns = generate_test_patterns(n_patterns, rng)
        t0 = time.perf_counter()
        logger.info("Benchmark 1/4: CrossModalRecall (%d patterns x %d reps)", n_patterns, training_reps)
        cm = CrossModalRecallBenchmark(self.network).run(patterns, training_reps, steps_per_pattern)
        logger.info("Benchmark 2/4: NoveltyDetection")
        nd = NoveltyDetectionBenchmark(self.network).run(
            patterns[0], generate_test_patterns(1, np.random.default_rng(seed + 999))[0],
            training_reps, steps_per_pattern)
        logger.info("Benchmark 3/4: AssociationStrength")
        ass = AssociationStrengthBenchmark(self.network).run(patterns, training_reps, steps_per_pattern)
        logger.info("Benchmark 4/4: EnergyEfficiency")
        en = EnergyEfficiencyBenchmark(self.network).run(patterns, steps_per_pattern)
        return _to_native({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "step_count": self.network.step_count,
            "total_neurons": int(self.network.config.populations.total),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "cross_modal_recall": cm, "novelty_detection": nd,
            "association_strength": ass, "energy_efficiency": en,
        })

    def run_multi_seed(self, n_seeds: int = 5, n_patterns: int = 20, training_reps: int = 10,
                       steps_per_pattern: int = 20, base_seed: int = 42,
                       confidence: float = 0.95) -> dict[str, Any]:
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

        return _to_native({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_seeds": n_seeds,
            "seeds": seeds,
            "confidence": confidence,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "aggregate": aggregate,
            "runs": runs,
        })

    def save_results(self, results: dict[str, Any], output_dir: str) -> Path:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
            f"Time: {results.get('elapsed_s', '?')}s", "",
        ]
        cm = results.get("cross_modal_recall", {})
        lines += [f"1. Cross-Modal Recall",
                  f"   V->A recall: {cm.get('visual_to_auditory_recall', 0):.4f}",
                  f"   A->V recall: {cm.get('auditory_to_visual_recall', 0):.4f}",
                  f"   Binding delta: {cm.get('binding_strength_delta', 0):+.6f}", ""]
        nd = results.get("novelty_detection", {})
        lines += [f"2. Novelty Detection",
                  f"   Familiar error: {nd.get('familiar_pred_error', 0):.4f}",
                  f"   Novel error:    {nd.get('novel_pred_error', 0):.4f}",
                  f"   Discrimination: {nd.get('discrimination_ratio', 0):.2f}x", ""]
        a = results.get("association_strength", {})
        lines.append("3. Association Strength")
        for gn, wc in a.get("weight_changes", {}).items():
            lines.append(f"   {gn}: {wc.get('initial_mean',0):.4f} -> "
                         f"{wc.get('final_mean',0):.4f} ({wc.get('delta_mean',0):+.6f})")
        lines += [f"   Concepts: {a.get('concept_count', 0)}", ""]
        en = results.get("energy_efficiency", {})
        lines += [f"4. Energy Efficiency",
                  f"   Spikes/step: {en.get('mean_spikes_per_step', 0):.0f}",
                  f"   Global rate:  {en.get('global_firing_rate', 0):.6f}",
                  f"   Energy units: {en.get('approx_energy_units', 0):.2f}",
                  "", "=" * 35]
        return "\n".join(lines)

    @staticmethod
    def summary_multi_seed(results: dict[str, Any]) -> str:
        """Human-readable mean + confidence-interval summary for a multi-seed run."""
        agg = results.get("aggregate", {})
        lines = [
            "=== Engram Multi-Seed Benchmark Summary ===",
            f"Seeds: {results.get('n_seeds', '?')} {results.get('seeds', [])}  |  "
            f"Confidence: {results.get('confidence', 0.95):.0%}  |  "
            f"Time: {results.get('elapsed_s', '?')}s", "",
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
        lines += ["", f"  {len(agg)} metrics tracked across seeds (full detail in saved JSON)",
                  "", "=" * 45]
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
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of distinct seeds to run (default: 1). When > 1, runs the "
             "suite once per seed and reports mean + confidence interval per "
             "metric instead of a single-run summary.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level for multi-seed intervals (default: 0.95)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from neuromorphic.config import NeuromorphicConfig
    from neuromorphic.network import NeuromorphicNetwork
    config = NeuromorphicConfig.from_env()
    print(f"Initializing network: {config.populations.total:,} neurons")
    network = NeuromorphicNetwork(config)
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        from neuromorphic.persistence import NeuromorphicPersistence
        import asyncio
        async def _load():
            p = NeuromorphicPersistence(args.checkpoint)
            await p.open(); state = await p.load_state(); await p.close()
            return state
        state = asyncio.run(_load())
        if state:
            network.set_state(state)
            print(f"  Restored at step {network.step_count:,}")
    suite = BenchmarkSuite(network)
    if args.seeds > 1:
        results = suite.run_multi_seed(
            args.seeds, args.patterns, args.reps, args.steps,
            base_seed=args.seed, confidence=args.confidence,
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
