"""Tests for learning evidence benchmark parsing (issue #131)."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_LE_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "dashboard", "learning_evidence.py")
_spec = importlib.util.spec_from_file_location("learning_evidence", _LE_PATH)
learning_evidence = importlib.util.module_from_spec(_spec)
sys.modules["learning_evidence"] = learning_evidence
_spec.loader.exec_module(learning_evidence)


def _write_benchmark(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_metrics_from_benchmark_suite_format():
    raw = {
        "timestamp": "2026-07-01T12:00:00",
        "association_strength": {"concept_count": 7},
        "concept_separability": {"silhouette_score": 0.61, "linear_probe_accuracy": 0.75},
        "cross_modal_binding_accuracy": {"f1": 0.42, "precision": 0.5, "recall": 0.36},
        "step_count": 1200,
    }
    m = learning_evidence.extract_learning_metrics(raw)
    assert m["concept_separability"] == 0.61
    assert m["linear_probe_accuracy"] == 0.75
    assert m["binding_accuracy"] == 0.42
    assert m["binding_precision"] == 0.5


def test_extract_metrics_surfaces_linear_probe_independently():
    """Issue #287: linear_probe_accuracy is a reported metric, not only a fallback."""
    raw = {
        "concept_separability": {"silhouette_score": 0.42, "linear_probe_accuracy": 0.88},
    }
    m = learning_evidence.extract_learning_metrics(raw)
    assert m["concept_separability"] == 0.42
    assert m["linear_probe_accuracy"] == 0.88


def test_extract_metrics_prefers_silhouette_over_concept_count():
    raw = {
        "association_strength": {"concept_count": 7},
        "concept_separability": {"silhouette_score": 0.42},
    }
    m = learning_evidence.extract_learning_metrics(raw)
    assert m["concept_separability"] == 0.42
    assert m["linear_probe_accuracy"] is None


def test_extract_metrics_binding_matched_decoy_ratio():
    raw = {
        "cross_modal_binding_accuracy": {
            "f1": 0.5,
            "matched_to_decoy_ratio": 2.25,
        },
    }
    m = learning_evidence.extract_learning_metrics(raw)
    assert m["binding_matched_decoy_ratio"] == pytest.approx(2.25)


def test_extract_metrics_fallback_to_script_benchmark_format():
    raw = {
        "timestamp": "2026-03-04T04:13:26",
        "learning": {"concept_count": 3},
        "cross_modal_recall": {
            "visual_to_auditory_recall": 0.2,
            "auditory_to_visual_recall": 0.4,
        },
    }
    m = learning_evidence.extract_learning_metrics(raw)
    assert m["concept_separability"] == 3.0
    assert m["binding_accuracy"] == pytest.approx(0.3)


def test_load_benchmark_files_builds_time_series(tmp_path):
    _write_benchmark(
        tmp_path / "benchmark_20260101_100000.json",
        {
            "timestamp": "2026-01-01T10:00:00",
            "association_strength": {"concept_count": 1},
            "cross_modal_binding_accuracy": {"f1": 0.1},
        },
    )
    _write_benchmark(
        tmp_path / "benchmarks_20260102_100000.json",
        {
            "timestamp": "2026-01-02T10:00:00",
            "association_strength": {"concept_count": 4},
            "cross_modal_binding_accuracy": {"f1": 0.25},
        },
    )
    _write_benchmark(
        tmp_path / "ci_performance_baseline.json",
        {
            "metrics": {"speed_steps_per_sec": 400.0},
        },
    )

    entries, err = learning_evidence.load_benchmark_files(
        benchmark_dirs=[tmp_path],
    )
    assert err is None
    assert len(entries) == 2
    assert entries[0]["metrics"]["concept_separability"] == 1.0
    assert entries[1]["metrics"]["binding_accuracy"] == 0.25


def test_is_benchmark_result_file():
    assert learning_evidence.is_benchmark_result_file("benchmark_20260304_041527.json")
    assert learning_evidence.is_benchmark_result_file("benchmarks_20260102_100000.json")
    assert not learning_evidence.is_benchmark_result_file("ci_performance_baseline.json")
    assert not learning_evidence.is_benchmark_result_file("gpu_synapse_spike.json")


def test_build_learning_evidence_series(tmp_path):
    _write_benchmark(
        tmp_path / "benchmarks_20260101_120000.json",
        {
            "association_strength": {"concept_count": 2},
            "concept_separability": {"silhouette_score": 0.55, "linear_probe_accuracy": 0.8},
            "cross_modal_binding_accuracy": {"f1": 0.5},
        },
    )
    payload = learning_evidence.build_learning_evidence_series(
        benchmark_dirs=[tmp_path],
    )
    assert payload["count"] == 1
    assert len(payload["series"]["concept_separability"]) == 1
    assert len(payload["series"]["linear_probe_accuracy"]) == 1
    assert payload["series"]["linear_probe_accuracy"][0]["value"] == 0.8
    assert len(payload["series"]["binding_accuracy"]) == 1
    assert payload["latest"]["metrics"]["binding_accuracy"] == 0.5
    assert payload["latest"]["metrics"]["linear_probe_accuracy"] == 0.8


def test_repo_benchmark_sample_readable():
    """Smoke test: sample file under neuromorphic/benchmarks/ is parseable."""
    repo_root = Path(__file__).resolve().parents[2]
    sample = repo_root / "neuromorphic" / "benchmarks" / "benchmark_20260304_041527.json"
    if not sample.exists():
        return
    raw = json.loads(sample.read_text(encoding="utf-8"))
    m = learning_evidence.extract_learning_metrics(raw)
    assert m["concept_separability"] is not None or m["binding_accuracy"] is not None
