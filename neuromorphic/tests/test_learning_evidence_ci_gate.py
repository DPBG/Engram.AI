"""Tests for scripts/learning_evidence_ci_gate.py (issue #286)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_NEURO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

import learning_evidence_ci_gate  # noqa: E402


@pytest.fixture
def baseline() -> dict:
    return {
        "window_size": 3,
        "regression_threshold_pct": 40,
        "history": [
            {"metrics": {"binding_accuracy": 0.2, "concept_separability": 0.0}},
            {"metrics": {"binding_accuracy": 0.3, "concept_separability": 0.0}},
            {"metrics": {"binding_accuracy": 0.4, "concept_separability": 0.0}},
        ],
    }


class TestRollingMeans:
    def test_mean_over_window(self, baseline):
        means = learning_evidence_ci_gate.rolling_means(baseline["history"], window_size=3)
        assert means["binding_accuracy"] == pytest.approx(0.3)
        assert means["concept_separability"] == pytest.approx(0.0)

    def test_window_smaller_than_history_uses_most_recent(self, baseline):
        means = learning_evidence_ci_gate.rolling_means(baseline["history"], window_size=2)
        # last 2 entries: 0.3, 0.4
        assert means["binding_accuracy"] == pytest.approx(0.35)

    def test_missing_metric_excluded_from_means(self):
        history = [
            {"metrics": {"binding_accuracy": 0.5}},
            {"metrics": {}},
        ]
        means = learning_evidence_ci_gate.rolling_means(history, window_size=2)
        assert means["binding_accuracy"] == pytest.approx(0.5)

    def test_empty_history_produces_no_means(self):
        assert learning_evidence_ci_gate.rolling_means([], window_size=5) == {}


class TestCheckRegression:
    def test_passes_within_threshold(self, baseline):
        # rolling mean 0.3, threshold 40% -> floor 0.18; 0.2 clears it
        metrics = {"binding_accuracy": 0.2, "concept_separability": 0.0}
        assert learning_evidence_ci_gate.check_regression(metrics, baseline) == []

    def test_fails_below_floor(self, baseline):
        # rolling mean 0.3, floor 0.18; 0.1 is below
        metrics = {"binding_accuracy": 0.1, "concept_separability": 0.0}
        failures = learning_evidence_ci_gate.check_regression(metrics, baseline)
        assert any("binding_accuracy regressed" in f for f in failures)

    def test_stable_zero_metric_does_not_false_positive(self, baseline):
        """concept_separability sits at a reproducible 0.0 floor at CI scale
        (see benchmarks/learning_evidence_baseline.json) -- a fresh run also
        landing at 0.0 must not be flagged as a regression."""
        metrics = {"binding_accuracy": 0.3, "concept_separability": 0.0}
        assert learning_evidence_ci_gate.check_regression(metrics, baseline) == []

    def test_negative_value_on_stable_zero_metric_fails(self, baseline):
        metrics = {"binding_accuracy": 0.3, "concept_separability": -0.1}
        failures = learning_evidence_ci_gate.check_regression(metrics, baseline)
        assert any("concept_separability regressed" in f for f in failures)

    def test_missing_tracked_metric_fails(self, baseline):
        metrics = {"concept_separability": 0.0}  # binding_accuracy absent
        failures = learning_evidence_ci_gate.check_regression(metrics, baseline)
        assert any("binding_accuracy" in f and "missing" in f for f in failures)


def test_committed_baseline_is_valid_json():
    path = _NEURO_DIR / "benchmarks" / "learning_evidence_baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "history" in data
    assert len(data["history"]) > 0
    assert "benchmark_args" in data
    assert "window_size" in data
    for entry in data["history"]:
        assert "metrics" in entry


def test_fresh_run_meets_committed_baseline():
    """Ground-truth check: a fresh run through the real gate pipeline

    (BenchmarkSuite -> save_results -> dashboard.learning_evidence.load_benchmark_files)
    must clear the committed rolling-window baseline. If this fails, either
    there's a genuine regression, or the baseline has drifted and needs
    `python scripts/learning_evidence_ci_gate.py --update-baseline`.
    """
    baseline_path = _NEURO_DIR / "benchmarks" / "learning_evidence_baseline.json"
    baseline = learning_evidence_ci_gate.load_baseline(baseline_path)
    learning_evidence_ci_gate.apply_ci_env(baseline.get("env", {}))

    with tempfile.TemporaryDirectory(prefix="engram-learning-evidence-test-") as tmp:
        learning_evidence_ci_gate.run_benchmark_suite(Path(tmp), baseline.get("benchmark_args", {}))
        metrics = learning_evidence_ci_gate.extract_metrics_via_dashboard(Path(tmp))

    failures = learning_evidence_ci_gate.check_regression(metrics, baseline)
    assert failures == [], (
        "Fresh BenchmarkSuite run did not meet the committed learning-evidence baseline.\n"
        "If this is a genuine regression, fix the code.\n"
        "If the baseline has drifted, refresh it with:\n"
        "  cd neuromorphic && python scripts/learning_evidence_ci_gate.py --update-baseline\n"
        "Failures:\n" + "\n".join(f"  - {f}" for f in failures)
    )


def test_gate_writes_files_load_benchmark_files_recognizes():
    """The saved run must actually match resolve_benchmark_dirs()'s naming
    convention (benchmarks_<timestamp>.json) -- this is the "wire it to the
    real dashboard-reading code path" acceptance criterion, verified rather
    than assumed."""
    from dashboard.learning_evidence import is_benchmark_result_file

    baseline_path = _NEURO_DIR / "benchmarks" / "learning_evidence_baseline.json"
    baseline = learning_evidence_ci_gate.load_baseline(baseline_path)
    learning_evidence_ci_gate.apply_ci_env(baseline.get("env", {}))

    with tempfile.TemporaryDirectory(prefix="engram-learning-evidence-test-") as tmp:
        learning_evidence_ci_gate.run_benchmark_suite(Path(tmp), baseline.get("benchmark_args", {}))
        written = list(Path(tmp).glob("*.json"))
        assert len(written) == 1
        assert is_benchmark_result_file(written[0].name)
