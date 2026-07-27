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

    # Same, at a scaled-up population size (issue #440 / ADR 0002 follow-up
    # #2: locate the actual serial/parallel crossover point between the
    # 55K dev-scale default and the 220K point ADR 0002 already measured).
    # --scale is a multiplier on every _DEV_SCALE_ENV population count, e.g.
    # --scale 1.4545 ≈ 80K neurons, --scale 2.909 ≈ 160K neurons:
    cd neuromorphic && uv run python scripts/profile_hotpath.py --compare-threading --scale 1.4545
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

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


# Keys in _DEV_SCALE_ENV that are population counts (scaled by --scale).
# NEURO_COGNITIVE_ENABLED is a boolean flag, not a count -- must not be
# multiplied.
_POPULATION_KEYS = tuple(k for k in _DEV_SCALE_ENV if k != "NEURO_COGNITIVE_ENABLED")


def _build_network(threads: str | None = None, scale: float = 1.0):
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    env = dict(_DEV_SCALE_ENV)
    if scale != 1.0:
        for key in _POPULATION_KEYS:
            env[key] = str(max(1, round(int(env[key]) * scale)))
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


def profile(steps: int, top: int, output: str | None, scale: float = 1.0) -> None:
    net, config = _build_network(scale=scale)
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


def compare_threading(steps: int, scale: float = 1.0) -> None:
    """A/B compare serial (NEURO_STDP_THREADS=1) vs. the default parallel
    thread pool at the profiled dev scale (or a --scale multiple of it).
    See docs/adr/0002 for why this matters: the default is measurably
    *slower* than serial at the unscaled dev scale."""
    printed_total = False
    for threads in ("1", "8"):
        net, config = _build_network(threads=threads, scale=scale)
        if not printed_total:
            print(f"Total neurons: {config.populations.total:,} (scale={scale})")
            printed_total = True
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
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier on every _DEV_SCALE_ENV population count (issue #440: "
            "locate the serial/parallel crossover between the 55K default and "
            "the 220K point ADR 0002 already measured). Default 1.0 = unscaled "
            "55K dev-scale config."
        ),
    )
    args = parser.parse_args()

    if args.compare_threading:
        compare_threading(args.steps, scale=args.scale)
    else:
        profile(args.steps, args.top, args.output, scale=args.scale)


if __name__ == "__main__":
    main()
