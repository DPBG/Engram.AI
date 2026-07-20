"""
CI performance-regression gate for scripts/benchmark.py (issue #132).

Runs a small fixed-size network benchmark and fails if step timing regresses
beyond the committed baseline in benchmarks/ci_performance_baseline.json.

Schema note (issue #325, see docs/benchmark-schema.md): this gate reads
scripts/benchmark.py's own "speed"/"learning" keys, produced by that script
directly (not read from disk) -- a different schema than
BenchmarkSuite.run_all()'s 6-benchmark output documented there. The two
producers' saved JSON share the neuromorphic/benchmarks/ directory with
similarly-named files (benchmark_*.json vs benchmarks_*.json), but this gate
never reads BenchmarkSuite's schema at all.

Usage:
    cd neuromorphic && uv run python scripts/benchmark_ci_gate.py
    cd neuromorphic && uv run python scripts/benchmark_ci_gate.py --baseline benchmarks/ci_performance_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_NEURO_DIR = _SCRIPT_DIR.parent
_DEFAULT_BASELINE = _NEURO_DIR / "benchmarks" / "ci_performance_baseline.json"


def load_baseline(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError(f"Baseline missing 'metrics': {path}")
    return data


def apply_ci_env(env_overrides: dict[str, str]) -> None:
    """Apply small-network env vars from the baseline profile."""
    for key, value in env_overrides.items():
        os.environ[key] = str(value)


def run_benchmark(
    output_dir: Path,
    steps: int,
    patterns: int,
    reps: int,
    steps_per_pattern: int,
) -> dict[str, Any]:
    """Invoke scripts/benchmark.py and return the latest result JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "benchmark.py"),
        "--steps",
        str(steps),
        "--patterns",
        str(patterns),
        "--reps",
        str(reps),
        "--steps-per-pattern",
        str(steps_per_pattern),
        "--output",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, cwd=_NEURO_DIR)
    files = sorted(output_dir.glob("benchmark_*.json"))
    if not files:
        raise RuntimeError(f"No benchmark output in {output_dir}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def check_regression(
    result: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    """Return human-readable failure messages (empty if within threshold)."""
    metrics = baseline["metrics"]
    threshold = float(baseline.get("regression_threshold_pct", 25)) / 100.0
    failures: list[str] = []

    speed = result.get("speed", {})
    learning = result.get("learning", {})

    actual_sps = float(speed.get("steps_per_sec", 0))
    base_sps = float(metrics["speed_steps_per_sec"])
    min_sps = base_sps * (1.0 - threshold)
    if actual_sps < min_sps:
        failures.append(
            f"speed steps/sec regressed: {actual_sps:.1f} < minimum {min_sps:.1f} "
            f"(baseline {base_sps:.1f}, threshold -{threshold * 100:.0f}%)",
        )

    phase_timing = speed.get("phase_timing", {})
    total_timing = phase_timing.get("total", {})
    actual_ms = float(total_timing.get("mean_ms", 0))
    base_ms = float(metrics["speed_mean_step_ms"])
    max_ms = base_ms * (1.0 + threshold)
    if actual_ms > max_ms:
        failures.append(
            f"mean step time regressed: {actual_ms:.3f} ms > maximum {max_ms:.3f} ms "
            f"(baseline {base_ms:.3f} ms, threshold +{threshold * 100:.0f}%)",
        )

    actual_learn = float(learning.get("steps_per_sec", 0))
    base_learn = float(metrics["learning_steps_per_sec"])
    min_learn = base_learn * (1.0 - threshold)
    if actual_learn < min_learn:
        failures.append(
            f"learning steps/sec regressed: {actual_learn:.1f} < minimum {min_learn:.1f} "
            f"(baseline {base_learn:.1f}, threshold -{threshold * 100:.0f}%)",
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="CI benchmark performance regression gate")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_DEFAULT_BASELINE,
        help="Committed baseline JSON (default: benchmarks/ci_performance_baseline.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for benchmark output (default: temp dir)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Print measured metrics for baseline refresh (does not fail)",
    )
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    bench_args = baseline.get("benchmark_args", {})
    apply_ci_env(baseline.get("env", {}))

    out_dir = args.output
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="engram-bench-ci-")
        out_dir = Path(tmp.name)

    result = run_benchmark(
        out_dir,
        steps=int(bench_args.get("steps", 50)),
        patterns=int(bench_args.get("patterns", 2)),
        reps=int(bench_args.get("reps", 1)),
        steps_per_pattern=int(bench_args.get("steps_per_pattern", 8)),
    )

    if args.update_baseline:
        speed = result.get("speed", {})
        learning = result.get("learning", {})
        measured = {
            "speed_steps_per_sec": speed.get("steps_per_sec"),
            "speed_mean_step_ms": speed.get("phase_timing", {}).get("total", {}).get("mean_ms"),
            "learning_steps_per_sec": learning.get("steps_per_sec"),
        }
        print(json.dumps({"metrics": measured}, indent=2))
        return 0

    failures = check_regression(result, baseline)
    if failures:
        print("PERFORMANCE REGRESSION DETECTED", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(
            "\nIf this regression is expected, update "
            "neuromorphic/benchmarks/ci_performance_baseline.json after review.",
            file=sys.stderr,
        )
        return 1

    speed = result["speed"]
    print(
        f"OK: speed {speed['steps_per_sec']:.1f} steps/sec, "
        f"mean step {speed['phase_timing']['total']['mean_ms']:.3f} ms",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
