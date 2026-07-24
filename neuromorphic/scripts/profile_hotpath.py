"""
Neuromorphic hot-path profiler (issue #133, docs/adr/0002).

Profiles NeuromorphicNetwork.step() with cProfile at a representative
scale (the same population sizes launcher/registry.py's _NEURO_SMALL uses
for `python run.py`'s default core profile — i.e. what a contributor's
machine actually runs, not an arbitrary size), and reports the top
bottlenecks by self (tottime) and cumulative time.

Usage:
    cd neuromorphic && uv run python scripts/profile_hotpath.py
    cd neuromorphic && uv run python scripts/profile_hotpath.py --steps 500 --top 20
    cd neuromorphic && uv run python scripts/profile_hotpath.py --output profiling/hotpath.pstats

    # A/B compare the STDP thread pool against serial execution at the
    # profiled scale (see docs/adr/0002-neuromorphic-hotpath-profiling.md
    # for why this matters — the default thread pool is *slower* than
    # serial at the default dev scale):
    cd neuromorphic && uv run python scripts/profile_hotpath.py --compare-threading
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - resource exists on Linux/macOS CI
    resource = None

# Population sizes launcher/registry.py's _NEURO_SMALL uses for the
# default `python run.py` core profile — the actual scale a contributor
# runs locally, not an arbitrary benchmark size.
_DEV_SCALE_ENV = {
    "NEURO_BRAINSTEM_N": "2000",
    "NEURO_REFLEX_N": "1500",
    "NEURO_SENSORY_N": "12000",
    "NEURO_MOTOR_N": "6000",
    "NEURO_CEREBELLUM_N": "6000",
    "NEURO_ASSOCIATION_N": "12000",
    "NEURO_PREDICTIVE_N": "6000",
    "NEURO_WORKING_MEM_N": "2000",
    "NEURO_FEATURE_N": "5000",
    "NEURO_CONCEPT_N": "1500",
    "NEURO_META_N": "1000",
    "NEURO_COGNITIVE_ENABLED": "1",
}


@dataclass(frozen=True)
class MemorySnapshots:
    before_build_mb: float | None
    after_build_mb: float | None
    after_warmup_mb: float | None
    after_profile_mb: float | None


def _get_peak_rss_mb() -> float | None:
    """Return this process's peak resident-set size in MiB."""
    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (OSError, ValueError):
        return None
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(usage.ru_maxrss) / divisor


def _format_memory_summary(snapshots: MemorySnapshots) -> str:
    """Format absolute peak RSS snapshots and non-negative phase deltas."""
    values = (
        snapshots.before_build_mb,
        snapshots.after_build_mb,
        snapshots.after_warmup_mb,
        snapshots.after_profile_mb,
    )
    header = "=== MEMORY HIGH-WATER MARKS ==="
    if any(value is None for value in values):
        return f"{header}\nPeak RSS unavailable on this platform."

    before, after_build, after_warmup, after_profile = values
    assert before is not None
    assert after_build is not None
    assert after_warmup is not None
    assert after_profile is not None
    build_delta = max(0.0, after_build - before)
    warmup_delta = max(0.0, after_warmup - after_build)
    profile_delta = max(0.0, after_profile - after_warmup)
    total_delta = max(0.0, after_profile - before)
    return "\n".join(
        [
            header,
            f"Before network: {before:10.1f} MiB",
            f"After network:  {after_build:10.1f} MiB  (+{build_delta:.1f} MiB)",
            f"After warmup:   {after_warmup:10.1f} MiB  (+{warmup_delta:.1f} MiB)",
            f"After profile:  {after_profile:10.1f} MiB  (+{profile_delta:.1f} MiB)",
            f"Total observed:                    +{total_delta:.1f} MiB",
        ]
    )


def _build_network(threads: str | None = None):
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    env = dict(_DEV_SCALE_ENV)
    if threads is not None:
        env["NEURO_STDP_THREADS"] = threads
    os.environ.update(env)

    from neuromorphic.config import NeuromorphicConfig
    from neuromorphic.network import NeuromorphicNetwork

    config = NeuromorphicConfig.from_env()
    return NeuromorphicNetwork(config, seed=42), config


def _run_steps(net, n_steps: int, warmup: int = 20) -> float:
    import numpy as np

    rng = np.random.default_rng(0)
    vis = rng.random(3000).astype(np.float32)
    aud = rng.random(1500).astype(np.float32)

    for _ in range(warmup):
        c = net.inject_multimodal(
            {
                "sensor.videofile.profile": vis,
                "sensor.audiofile.profile": aud,
            }
        )
        net.step(c)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        c = net.inject_multimodal(
            {
                "sensor.videofile.profile": vis,
                "sensor.audiofile.profile": aud,
            }
        )
        net.step(c)
    return time.perf_counter() - t0


def profile(steps: int, top: int, output: str | None) -> None:
    net, config = _build_network()
    print(f"Total neurons: {config.populations.total:,}")

    def _run():
        _run_steps(net, steps, warmup=20)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    _run()
    profiler.disable()
    elapsed = time.perf_counter() - t0
    print(
        f"\n{steps} steps in {elapsed:.2f}s = {elapsed / steps * 1000:.2f} ms/step, "
        f"{steps / elapsed:.2f} steps/sec\n"
    )

    stats = pstats.Stats(profiler)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        stats.dump_stats(output)
        print(f"Raw profile saved: {output}\n")

    print(f"=== TOP {top} BY SELF TIME (tottime) ===")
    stats.sort_stats("tottime").print_stats(top)
    print(f"\n=== TOP {top} BY CUMULATIVE TIME ===")
    stats.sort_stats("cumulative").print_stats(top)


def compare_threading(steps: int) -> None:
    """A/B compare serial (NEURO_STDP_THREADS=1) vs. the default parallel
    thread pool at the profiled dev scale. See docs/adr/0002 for why this
    matters: the default is measurably *slower* at this scale."""
    for threads in ("1", "8"):
        net, _ = _build_network(threads=threads)
        elapsed = _run_steps(net, steps, warmup=20)
        print(
            f"NEURO_STDP_THREADS={threads}: {steps} steps in {elapsed:.3f}s = "
            f"{elapsed / steps * 1000:.3f} ms/step, {steps / elapsed:.2f} steps/sec"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--top", type=int, default=30, help="Rows to print per sort order")
    parser.add_argument("--output", type=str, default=None, help="Path to save raw .pstats")
    parser.add_argument(
        "--compare-threading",
        action="store_true",
        help="A/B compare serial vs. parallel STDP dispatch instead of profiling",
    )
    args = parser.parse_args()

    if args.compare_threading:
        compare_threading(args.steps)
    else:
        profile(args.steps, args.top, args.output)


if __name__ == "__main__":
    main()
