"""
CI longitudinal learning-evidence regression gate (issue #286).

benchmark_ci_gate.py already gates scripts/benchmark.py's speed/learning
schema against a single committed baseline. Nothing gated BenchmarkSuite's
6-metric investor schema -- the one the dashboard's Learning Evidence panel
(dashboard/src/dashboard/learning_evidence.py) actually reads -- so a change
that silently degraded learning quality would only surface if a maintainer
happened to eyeball the dashboard.

This gate:
  1. Runs BenchmarkSuite on a small CI-sized network (with a real, non-zero
     concept layer -- unlike ci_performance_baseline.json's profile, which
     zeroes it out and would make concept_separability a meaningless
     always-0.0 metric here).
  2. Saves the result via BenchmarkSuite.save_results(), which writes the
     exact benchmarks_<timestamp>.json shape resolve_benchmark_dirs() and
     is_benchmark_result_file() already recognize.
  3. Reads it back through dashboard.learning_evidence.load_benchmark_files()
     -- the *actual* function the dashboard calls, not a reimplementation --
     so a regression in extract_learning_metrics() itself is caught too.
  4. Compares the extracted metrics against a rolling window of prior runs
     recorded in the committed baseline (benchmarks/learning_evidence_baseline.json),
     not a single fixed point: at CI scale (tiny network, few patterns) a
     single run is noisy, so the floor is the window's mean minus a tolerance.

Refresh the window the same way benchmark_ci_gate.py refreshes its baseline:

Usage:
    cd neuromorphic && uv run python scripts/learning_evidence_ci_gate.py
    cd neuromorphic && uv run python scripts/learning_evidence_ci_gate.py --update-baseline > /tmp/updated.json
        # review /tmp/updated.json, then copy it over
        # benchmarks/learning_evidence_baseline.json and commit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_NEURO_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _NEURO_DIR.parent
_DEFAULT_BASELINE = _NEURO_DIR / "benchmarks" / "learning_evidence_baseline.json"

for _extra_path in (_NEURO_DIR / "src", _REPO_ROOT / "dashboard" / "src"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

# Metrics extract_learning_metrics() produces that are meaningful to gate on.
# step_count/total_neurons are run metadata, not learning-quality signals.
TRACKED_METRICS: tuple[str, ...] = (
    "concept_separability",
    "binding_accuracy",
    "binding_precision",
    "binding_recall",
    "binding_matched_decoy_ratio",
)


def load_baseline(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "history" not in data:
        raise ValueError(f"Baseline missing 'history': {path}")
    return data


def apply_ci_env(env_overrides: dict[str, str]) -> None:
    for key, value in env_overrides.items():
        os.environ[key] = str(value)


def run_benchmark_suite(output_dir: Path, bench_args: dict[str, Any]) -> dict[str, Any]:
    """Build the CI-sized network, run BenchmarkSuite, save via the real save_results()."""
    from neuromorphic.benchmarks import BenchmarkSuite
    from neuromorphic.config import NeuromorphicConfig
    from neuromorphic.network import NeuromorphicNetwork

    config = NeuromorphicConfig.from_env()
    config.concept_layer.k_winners = int(bench_args.get("concept_k_winners", 10))
    # NeuromorphicNetwork's own `seed` controls weight-init randomness, entirely
    # separate from run_all()'s `seed` (which only governs pattern/fixture
    # generation -- see issue #322). Leaving it unset defaults to OS entropy,
    # which made this gate flaky: a CI regression check must be deterministic
    # run-to-run so a red build always means a real regression, never bad luck.
    network = NeuromorphicNetwork(config, seed=int(bench_args.get("network_seed", 1234)))
    suite = BenchmarkSuite(network)
    results = suite.run_all(
        n_patterns=int(bench_args.get("n_patterns", 3)),
        training_reps=int(bench_args.get("training_reps", 2)),
        steps_per_pattern=int(bench_args.get("steps_per_pattern", 4)),
        seed=int(bench_args.get("seed", 42)),
    )
    suite.save_results(results, str(output_dir))
    return results


def extract_metrics_via_dashboard(output_dir: Path) -> dict[str, float | int | None]:
    """Read the saved run back through the real dashboard-reading code path."""
    from dashboard.learning_evidence import load_benchmark_files

    entries, error = load_benchmark_files(benchmark_dirs=[output_dir])
    if error or not entries:
        raise RuntimeError(f"learning_evidence.load_benchmark_files() found nothing: {error}")
    return entries[-1]["metrics"]


def rolling_means(history: list[dict[str, Any]], window_size: int) -> dict[str, float]:
    """Mean of each tracked metric over the last `window_size` history entries."""
    window = history[-window_size:] if window_size > 0 else history
    means: dict[str, float] = {}
    for name in TRACKED_METRICS:
        values = [
            float(h["metrics"][name]) for h in window if h.get("metrics", {}).get(name) is not None
        ]
        if values:
            means[name] = sum(values) / len(values)
    return means


def check_regression(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    """Return human-readable failure messages (empty if within threshold)."""
    threshold = float(baseline.get("regression_threshold_pct", 30)) / 100.0
    window_size = int(baseline.get("window_size", 10))
    means = rolling_means(baseline.get("history", []), window_size)
    failures: list[str] = []
    for name, mean in means.items():
        actual = metrics.get(name)
        if actual is None:
            failures.append(f"{name}: missing from fresh run (tracked in baseline window)")
            continue
        floor = mean * (1.0 - threshold)
        if float(actual) < floor:
            failures.append(
                f"{name} regressed: {float(actual):.4f} < minimum {floor:.4f} "
                f"(rolling mean {mean:.4f} over last {min(window_size, len(baseline.get('history', [])))} "
                f"runs, threshold -{threshold * 100:.0f}%)"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CI longitudinal learning-evidence regression gate"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_DEFAULT_BASELINE,
        help="Committed rolling-window baseline JSON "
        "(default: benchmarks/learning_evidence_baseline.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to write the benchmarks_*.json run into (default: temp dir)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Append this run's metrics to the rolling window and print the updated "
        "baseline JSON to stdout (does not fail, does not write to disk)",
    )
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    apply_ci_env(baseline.get("env", {}))

    out_dir = args.output
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="engram-learning-evidence-ci-")
        out_dir = Path(tmp.name)

    run_benchmark_suite(out_dir, baseline.get("benchmark_args", {}))
    metrics = extract_metrics_via_dashboard(out_dir)

    if args.update_baseline:
        window_size = int(baseline.get("window_size", 10))
        history = [*baseline.get("history", []), {"metrics": metrics}][-window_size:]
        print(json.dumps({**baseline, "history": history}, indent=2))
        return 0

    failures = check_regression(metrics, baseline)
    if failures:
        print("LEARNING EVIDENCE REGRESSION DETECTED", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(
            "\nIf this regression is expected (e.g. a deliberate architecture change), "
            "refresh the rolling window with:\n"
            "  cd neuromorphic && python scripts/learning_evidence_ci_gate.py "
            "--update-baseline > /tmp/updated.json\n"
            "then review and commit it over benchmarks/learning_evidence_baseline.json.",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v is not None),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
