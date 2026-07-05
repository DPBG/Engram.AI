"""Tests for scripts/benchmark_ci_gate.py (issue #132)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
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
        assert benchmark_ci_gate.check_regression(
            _result(450.0, 2.0, 400.0), baseline,
        ) == []

    def test_fails_slow_steps_per_sec(self, baseline):
        failures = benchmark_ci_gate.check_regression(
            _result(200.0, 2.0, 400.0), baseline,
        )
        assert any("speed steps/sec" in f for f in failures)

    def test_fails_high_mean_step_ms(self, baseline):
        failures = benchmark_ci_gate.check_regression(
            _result(450.0, 5.0, 400.0), baseline,
        )
        assert any("mean step time" in f for f in failures)

    def test_fails_slow_learning(self, baseline):
        failures = benchmark_ci_gate.check_regression(
            _result(450.0, 2.0, 100.0), baseline,
        )
        assert any("learning steps/sec" in f for f in failures)


def test_committed_baseline_is_valid_json():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "ci_performance_baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "metrics" in data
    assert "benchmark_args" in data
    assert data["metrics"]["speed_steps_per_sec"] > 0
