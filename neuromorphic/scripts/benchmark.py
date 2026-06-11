"""
Benchmark runner — standardized performance and learning metrics.

Runs a series of tests against the neuromorphic network and produces
comparable JSON results. Use before/after training or before/after
code changes to measure impact.

Usage:
    cd neuromorphic && uv run python scripts/benchmark.py
    cd neuromorphic && uv run python scripts/benchmark.py --checkpoint /data/sqlite/neuromorphic.db
    cd neuromorphic && uv run python scripts/benchmark.py --steps 5000 --output benchmarks/

Metrics captured:
    Speed: steps/sec, STDP time per group
    Learning: convergence steps, concept count, cross-modal binding
    Memory: peak RSS, synapse count
    Phase: developmental phase, myelination fraction
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.network import NeuromorphicNetwork


def get_system_info() -> dict:
    """Capture system information for reproducibility."""
    import multiprocessing
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": multiprocessing.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def get_peak_rss_mb() -> float:
    """Get peak RSS in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)  # macOS: bytes -> MB
    return usage.ru_maxrss / 1024  # Linux: KB -> MB


def generate_test_patterns(n_patterns: int, rng: np.random.Generator) -> list[dict]:
    """Generate standardized visual+auditory test pattern pairs.

    Each pattern is a distinct visual gaussian blob + auditory signature
    so we can measure cross-modal binding strength.
    """
    # Precompute coordinate grids (vectorized, not nested loops)
    y_grid, x_grid = np.mgrid[0:64, 0:64]  # (64, 64) each

    patterns = []
    for i in range(n_patterns):
        # Visual: gaussian blob at distinct position
        cx = (i * 7) % 64
        cy = (i * 13) % 64
        dist2 = (x_grid - cx).astype(np.float32) ** 2 + (y_grid - cy).astype(np.float32) ** 2
        visual = np.exp(-dist2 / 50.0, dtype=np.float32).flatten()
        visual /= visual.max() + 1e-8

        # Auditory: 13 MFCC-like features with distinct signature
        auditory = rng.normal(0.5, 0.1, size=13).astype(np.float32)
        auditory[i % 13] = 1.0
        np.clip(auditory, 0.0, 1.0, out=auditory)

        patterns.append({
            "visual": visual.tolist(),
            "auditory": auditory.tolist(),
            "label": f"pattern_{i:03d}",
        })
    return patterns


def run_speed_benchmark(network: NeuromorphicNetwork, steps: int) -> dict:
    """Measure simulation speed with per-phase timing breakdown."""
    sensory_n = network.sensory.n
    rng = np.random.default_rng(99)

    # Warm up (with sensory input to exercise full pathway)
    for _ in range(10):
        current = rng.uniform(0, 0.3, size=sensory_n).astype(np.float32)
        network.step(current)

    # Collect per-step timings
    step_timings = []
    t0 = time.perf_counter()
    for _ in range(steps):
        current = rng.uniform(0, 0.3, size=sensory_n).astype(np.float32)
        network.step(current)
        step_timings.append(network.get_step_timing(ema=False))
    elapsed = time.perf_counter() - t0

    steps_per_sec = steps / elapsed if elapsed > 0 else 0

    # Report synapse group sizes (int() for JSON serialization — syn.nnz is numpy int64)
    synapse_info = {}
    for name, syn in network.synapses.items():
        if syn.plastic and syn.nnz > 0:
            synapse_info[name] = {"nnz": int(syn.nnz)}

    # Aggregate per-phase stats
    phase_stats = {}
    if step_timings:
        for phase in step_timings[0]:
            values = [st[phase] for st in step_timings]
            phase_stats[phase] = {
                "mean_ms": round(float(np.mean(values)), 3),
                "std_ms": round(float(np.std(values)), 3),
                "p50_ms": round(float(np.median(values)), 3),
                "p95_ms": round(float(np.percentile(values, 95)), 3),
                "p99_ms": round(float(np.percentile(values, 99)), 3),
            }

    return {
        "steps": steps,
        "elapsed_s": round(elapsed, 3),
        "steps_per_sec": round(steps_per_sec, 2),
        "synapse_info": synapse_info,
        "phase_timing": phase_stats,
    }


def run_learning_benchmark(
    network: NeuromorphicNetwork,
    patterns: list[dict],
    steps_per_pattern: int,
    repetitions: int,
) -> dict:
    """Measure learning capacity by injecting test patterns."""
    initial_metrics = network.get_metrics()
    initial_step = network._step_count

    # Record initial synapse state for key groups
    binding_groups = ["sensory_association", "sensory_feature", "feature_association"]
    initial_weights = {}
    for name in binding_groups:
        syn = network.synapses.get(name)
        if syn and syn.nnz > 0:
            initial_weights[name] = float(syn.weights.data.mean())

    # Inject patterns with repetition
    t0 = time.perf_counter()
    for rep in range(repetitions):
        for pat in patterns:
            # Inject visual
            visual_current = network.inject_observation(
                pat["visual"], provenance="sensor.videofile.bench"
            )
            # Run a few steps with visual
            for _ in range(steps_per_pattern // 2):
                network.step(visual_current)
                visual_current *= 0.97  # decay

            # Inject auditory simultaneously
            auditory_current = network.inject_observation(
                pat["auditory"], provenance="sensor.audiofile.bench"
            )
            combined = visual_current + auditory_current
            for _ in range(steps_per_pattern // 2):
                network.step(combined)
                combined *= 0.97
    learn_elapsed = time.perf_counter() - t0

    final_metrics = network.get_metrics()
    final_step = network._step_count
    total_steps = final_step - initial_step

    # Measure weight changes in binding groups
    weight_changes = {}
    for name in binding_groups:
        syn = network.synapses.get(name)
        if syn and syn.nnz > 0 and name in initial_weights:
            weight_changes[name] = {
                "initial_mean": round(initial_weights[name], 6),
                "final_mean": round(float(syn.weights.data.mean()), 6),
                "delta": round(float(syn.weights.data.mean()) - initial_weights[name], 6),
            }

    # Count concept patterns if concept layer exists
    concept_count = 0
    if network.concept is not None:
        # Accumulate firing rates over multiple steps per pattern (not single spikes)
        activation_vectors = []
        for pat in patterns[:20]:
            current = network.inject_observation(pat["visual"], provenance="sensor.videofile.bench")
            accumulated = np.zeros(network.concept.n, dtype=np.float32)
            for _ in range(10):
                network.step(current)
                current *= 0.97
                accumulated += network.concept.spikes.astype(np.float32)
            if accumulated.any():
                activation_vectors.append(accumulated)

        # Count distinct patterns (cosine similarity < 0.3)
        if len(activation_vectors) >= 2:
            seen = [activation_vectors[0]]
            for v in activation_vectors[1:]:
                is_new = True
                for s in seen:
                    norm_v = np.linalg.norm(v)
                    norm_s = np.linalg.norm(s)
                    if norm_v > 0 and norm_s > 0:
                        sim = np.dot(v, s) / (norm_v * norm_s)
                        if sim > 0.3:
                            is_new = False
                            break
                if is_new:
                    seen.append(v)
            concept_count = len(seen)

    return {
        "patterns": len(patterns),
        "repetitions": repetitions,
        "steps_per_pattern": steps_per_pattern,
        "total_steps": total_steps,
        "elapsed_s": round(learn_elapsed, 3),
        "steps_per_sec": round(total_steps / learn_elapsed, 2) if learn_elapsed > 0 else 0,
        "weight_changes": weight_changes,
        "concept_count": concept_count,
        "convergence": final_metrics.get("convergence", {}),
        "neuromodulation": final_metrics.get("neuromodulation", {}),
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the full benchmark suite."""
    print("=" * 60)
    print("ActiveLearningAI Neuromorphic Benchmark")
    print("=" * 60)

    system_info = get_system_info()
    print(f"System: {system_info['platform']}")
    print(f"CPU: {system_info['processor']} ({system_info['cpu_count']} cores)")

    # Build config (from env or defaults)
    config = NeuromorphicConfig.from_env()
    print(f"\nNetwork: {config.populations.total:,} neurons, "
          f"{len([c for c in config.connections.__dataclass_fields__])} connection types")
    print(f"Memory estimate: {config.estimate_memory_bytes() / (1024**2):.0f} MB")

    # Build or load network
    print("\nInitializing network...")
    rss_before = get_peak_rss_mb()
    t0 = time.perf_counter()
    network = NeuromorphicNetwork(config)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        from neuromorphic.persistence import NeuromorphicPersistence
        import asyncio

        async def _load():
            p = NeuromorphicPersistence(args.checkpoint)
            await p.open()
            state = await p.load_state()
            await p.close()
            return state

        state = asyncio.run(_load())
        if state:
            network.set_state(state)
            print(f"  Loaded at step {network._step_count:,}")

    init_elapsed = time.perf_counter() - t0
    rss_after = get_peak_rss_mb()
    print(f"Init time: {init_elapsed:.2f}s")
    print(f"Memory: {rss_after:.0f} MB (delta: +{rss_after - rss_before:.0f} MB)")

    # Synapse summary
    total_nnz = int(sum(s.nnz for s in network.synapses.values()))
    print(f"Synapses: {total_nnz:,} (across {len(network.synapses)} groups)")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": system_info,
        "config": {
            "total_neurons": int(config.populations.total),
            "total_synapses": total_nnz,
            "synapse_groups": len(network.synapses),
            "dendrites_enabled": config.dendrites.enabled,
            "cognitive_enabled": config.cognitive_action.enabled,
            "initial_step": network._step_count,
        },
        "init": {
            "time_s": round(init_elapsed, 3),
            "memory_mb": round(rss_after, 1),
            "memory_delta_mb": round(rss_after - rss_before, 1),
        },
    }

    # 1. Speed benchmark (fresh network, before learning modifies weights)
    print(f"\n--- Speed Benchmark ({args.steps} steps) ---")
    speed = run_speed_benchmark(network, args.steps)
    print(f"  {speed['steps_per_sec']} steps/sec ({speed['elapsed_s']}s)")
    if speed.get("phase_timing"):
        total_ms = speed["phase_timing"].get("total", {}).get("mean_ms", 1)
        print(f"\n  Per-phase breakdown (mean ms, % of total):")
        for phase, stats in speed["phase_timing"].items():
            pct = (stats["mean_ms"] / total_ms * 100) if total_ms > 0 else 0
            label = phase if phase == "total" else phase.split("_", 1)[1] if "_" in phase else phase
            print(f"    {label:30s}  {stats['mean_ms']:8.3f} ms  ({pct:5.1f}%)  p95={stats['p95_ms']:.3f}")
    results["speed"] = speed

    # 2. Learning benchmark (modifies weights — runs after speed)
    rng = np.random.default_rng(42)
    patterns = generate_test_patterns(args.patterns, rng)
    print(f"\n--- Learning Benchmark ({args.patterns} patterns x {args.reps} reps) ---")
    learning = run_learning_benchmark(network, patterns, args.steps_per_pattern, args.reps)
    print(f"  {learning['total_steps']:,} steps in {learning['elapsed_s']}s "
          f"({learning['steps_per_sec']} steps/sec)")
    if learning["weight_changes"]:
        for name, wc in learning["weight_changes"].items():
            print(f"  {name}: {wc['initial_mean']:.4f} -> {wc['final_mean']:.4f} "
                  f"(delta: {wc['delta']:+.6f})")
    if learning["concept_count"] > 0:
        print(f"  Concept patterns: {learning['concept_count']}")
    print(f"  Phase: {learning['neuromodulation'].get('phase', 'unknown')}")
    results["learning"] = learning

    # 3. Memory summary
    rss_final = get_peak_rss_mb()
    total_nnz_final = int(sum(s.nnz for s in network.synapses.values()))
    myelination = {}
    for name, syn in network.synapses.items():
        if syn.plastic and syn.myelinated is not None:
            myelination[name] = round(float(syn.myelinated.sum()) / max(len(syn.myelinated), 1), 4)

    results["memory"] = {
        "peak_rss_mb": round(rss_final, 1),
        "total_synapses": total_nnz_final,
        "myelination_fractions": myelination,
    }
    print(f"\n--- Memory ---")
    print(f"  Peak RSS: {rss_final:.0f} MB")
    print(f"  Synapses: {total_nnz_final:,}")

    # Final state
    metrics = network.get_metrics()
    results["final_state"] = {
        "step_count": network._step_count,
        "firing_rates": {k: round(v, 6) for k, v in metrics.get("firing_rates", {}).items()},
        "convergence": metrics.get("convergence", {}),
        "neuromodulation": metrics.get("neuromodulation", {}),
        "phase": metrics.get("neuromodulation", {}).get("phase", "unknown"),
    }

    # Save results
    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Cannot create output directory {output_dir}: {e}")
        print("Results not saved to disk.")
        return results

    filename = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {output_path}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Neuromorphic network benchmark")
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Steps for speed benchmark (default: 1000)",
    )
    parser.add_argument(
        "--patterns", type=int, default=50,
        help="Number of test patterns for learning benchmark (default: 50)",
    )
    parser.add_argument(
        "--reps", type=int, default=5,
        help="Repetitions per pattern (default: 5)",
    )
    parser.add_argument(
        "--steps-per-pattern", type=int, default=20,
        help="Steps per pattern injection (default: 20)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to SQLite checkpoint to load before benchmarking",
    )
    parser.add_argument(
        "--output", type=str, default=str(Path(__file__).parent.parent / "benchmarks"),
        help="Output directory for results JSON",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
