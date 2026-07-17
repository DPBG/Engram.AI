"""Tests for scripts/benchmark_ci_gate.py (issue #132)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_NEURO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

import benchmark_ci_gate  # noqa: E402


@pytest.fixture
def baseline() -> dict:
    return {
        "regression_threshold_pct": 25,
        "metrics": {
            "speed_steps_per_sec": 400.0,
            "speed_mean_step_ms": 2.5,
            "learning_steps_per_sec": 350.0,
        },
    }


def _result(sps: float, mean_ms: float, learn_sps: float) -> dict:
    return {
        "speed": {
            "steps_per_sec": sps,
            "phase_timing": {"total": {"mean_ms": mean_ms}},
        },
        "learning": {"steps_per_sec": learn_sps},
    }


class TestCheckRegression:
    def test_passes_within_threshold(self, baseline):
        assert (
            benchmark_ci_gate.check_regression(
                _result(450.0, 2.0, 400.0),
                baseline,
            )
            == []
        )

    def test_fails_slow_steps_per_sec(self, baseline):
        failures = benchmark_ci_gate.check_regression(
            _result(200.0, 2.0, 400.0),
            baseline,
        )
        assert any("speed steps/sec" in f for f in failures)

    def test_fails_high_mean_step_ms(self, baseline):
        failures = benchmark_ci_gate.check_regression(
            _result(450.0, 5.0, 400.0),
            baseline,
        )
        assert any("mean step time" in f for f in failures)

    def test_fails_slow_learning(self, baseline):
        failures = benchmark_ci_gate.check_regression(
            _result(450.0, 2.0, 100.0),
            baseline,
        )
        assert any("learning steps/sec" in f for f in failures)


def test_committed_baseline_is_valid_json():
    path = _NEURO_DIR / "benchmarks" / "ci_performance_baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "metrics" in data
    assert "benchmark_args" in data
    assert data["metrics"]["speed_steps_per_sec"] > 0


def test_fresh_network_meets_committed_baseline():
    """Verify that a fresh (no checkpoint) network run reproduces the committed baseline.

    This is the ground-truth check for issue #334: the baseline must be achievable
    from a clean-network run, not just be a valid JSON document. If this test
    fails, the baseline value itself is stale and must be recaptured with
    `python scripts/benchmark_ci_gate.py --update-baseline`.

    The CI profile uses the small fixed network defined in env overrides inside
    ci_performance_baseline.json (700 neurons total, no dendrites, no concept layer)
    so this test completes in well under a minute even on slow hardware.
    """
    baseline_path = _NEURO_DIR / "benchmarks" / "ci_performance_baseline.json"
    baseline = benchmark_ci_gate.load_baseline(baseline_path)
    benchmark_ci_gate.apply_ci_env(baseline.get("env", {}))

    bench_args = baseline.get("benchmark_args", {})
    with tempfile.TemporaryDirectory(prefix="engram-bench-test-") as tmp:
        result = benchmark_ci_gate.run_benchmark(
            Path(tmp),
            steps=int(bench_args.get("steps", 50)),
            patterns=int(bench_args.get("patterns", 2)),
            reps=int(bench_args.get("reps", 1)),
            steps_per_pattern=int(bench_args.get("steps_per_pattern", 8)),
        )

    failures = benchmark_ci_gate.check_regression(result, baseline)
    assert failures == [], (
        "Fresh-network benchmark did not meet the committed baseline.\n"
        "If this is a genuine regression, fix the code.\n"
        "If the baseline itself has drifted (e.g. network architecture changed), "
        "recapture it with:\n"
        "  cd neuromorphic && python scripts/benchmark_ci_gate.py --update-baseline\n"
        "Failures:\n" + "\n".join(f"  - {f}" for f in failures)
    )
