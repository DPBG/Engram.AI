"""Tests for the benchmarking framework (benchmarks.py).

Verifies all 4 benchmarks produce valid results on a small network.
"""

import numpy as np
import pytest

from neuromorphic.benchmarks import (
    AssociationStrengthBenchmark,
    BenchmarkSuite,
    CrossModalRecallBenchmark,
    EnergyEfficiencyBenchmark,
    NoveltyDetectionBenchmark,
    _to_native,
    generate_test_patterns,
)
from neuromorphic.config import NeuromorphicConfig
from neuromorphic.network import NeuromorphicNetwork


@pytest.fixture
def small_network():
    """Minimal network for fast benchmarking tests."""
    cfg = NeuromorphicConfig.from_env()
    # Override to small populations
    cfg.populations.brainstem = 50
    cfg.populations.reflex_arc = 30
    cfg.populations.sensory_cortex = 200
    cfg.populations.motor_cortex = 100
    cfg.populations.cerebellum = 50
    cfg.populations.association_cortex = 150
    cfg.populations.predictive_layer = 80
    cfg.populations.working_memory = 40
    cfg.populations.feature_layer = 0
    cfg.populations.concept_layer = 0
    cfg.populations.meta_controller = 0
    return NeuromorphicNetwork(cfg)


@pytest.fixture
def patterns():
    rng = np.random.default_rng(42)
    return generate_test_patterns(3, rng)


class TestGeneratePatterns:
    def test_shape(self):
        rng = np.random.default_rng(0)
        pats = generate_test_patterns(5, rng)
        assert len(pats) == 5
        for p in pats:
            assert len(p["visual"]) == 64 * 64
            assert len(p["auditory"]) == 13
            assert "label" in p

    def test_deterministic(self):
        a = generate_test_patterns(3, np.random.default_rng(42))
        b = generate_test_patterns(3, np.random.default_rng(42))
        assert a[0]["visual"] == b[0]["visual"]


class TestToNative:
    def test_numpy_types(self):
        d = {"a": np.int64(5), "b": np.float32(3.14), "c": np.array([1, 2])}
        r = _to_native(d)
        assert isinstance(r["a"], int)
        assert isinstance(r["b"], float)
        assert isinstance(r["c"], list)


class TestCrossModalRecall:
    def test_produces_metrics(self, small_network, patterns):
        bench = CrossModalRecallBenchmark(small_network)
        result = bench.run(patterns, training_reps=2, steps_per_pattern=4)
        assert "visual_to_auditory_recall" in result
        assert "auditory_to_visual_recall" in result
        assert "binding_strength_delta" in result
        assert "patterns_tested" in result
        assert result["patterns_tested"] == 3

    def test_recall_values_in_range(self, small_network, patterns):
        bench = CrossModalRecallBenchmark(small_network)
        result = bench.run(patterns, training_reps=2, steps_per_pattern=4)
        assert 0.0 <= result["visual_to_auditory_recall"] <= 1.0
        assert 0.0 <= result["auditory_to_visual_recall"] <= 1.0


class TestNoveltyDetection:
    def test_produces_metrics(self, small_network, patterns):
        rng = np.random.default_rng(999)
        novel = generate_test_patterns(1, rng)[0]
        bench = NoveltyDetectionBenchmark(small_network)
        result = bench.run(
            patterns[0], novel, familiarization_reps=2, steps_per_rep=4, test_steps=3
        )
        assert "familiar_pred_error" in result
        assert "novel_pred_error" in result
        assert "discrimination_ratio" in result
        assert "firing_rate_shift" in result

    def test_pred_error_non_negative(self, small_network, patterns):
        novel = generate_test_patterns(1, np.random.default_rng(999))[0]
        bench = NoveltyDetectionBenchmark(small_network)
        result = bench.run(
            patterns[0], novel, familiarization_reps=2, steps_per_rep=4, test_steps=3
        )
        assert result["familiar_pred_error"] >= 0.0
        assert result["novel_pred_error"] >= 0.0


class TestAssociationStrength:
    def test_produces_metrics(self, small_network, patterns):
        bench = AssociationStrengthBenchmark(small_network)
        result = bench.run(patterns, training_reps=2, steps_per_pattern=4)
        assert "weight_changes" in result
        assert "myelination" in result
        assert "concept_count" in result
        assert result["patterns_trained"] == 3

    def test_weight_changes_have_delta(self, small_network, patterns):
        bench = AssociationStrengthBenchmark(small_network)
        result = bench.run(patterns, training_reps=2, steps_per_pattern=4)
        for name, wc in result["weight_changes"].items():
            assert "delta_mean" in wc
            assert "initial_mean" in wc
            assert "final_mean" in wc


class TestEnergyEfficiency:
    def test_produces_metrics(self, small_network, patterns):
        bench = EnergyEfficiencyBenchmark(small_network)
        result = bench.run(patterns, steps_per_pattern=4)
        assert "mean_spikes_per_step" in result
        assert "global_firing_rate" in result
        assert "region_firing_rates" in result
        assert "approx_energy_units" in result
        assert result["total_neurons"] > 0

    def test_firing_rates_non_negative(self, small_network, patterns):
        bench = EnergyEfficiencyBenchmark(small_network)
        result = bench.run(patterns, steps_per_pattern=4)
        for name, rate in result["region_firing_rates"].items():
            assert rate >= 0.0


class TestBenchmarkSuite:
    def test_run_all(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4)
        assert "cross_modal_recall" in results
        assert "novelty_detection" in results
        assert "association_strength" in results
        assert "energy_efficiency" in results
        assert "timestamp" in results
        assert results["total_neurons"] > 0

    def test_summary(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4)
        text = suite.summary(results)
        assert "Cross-Modal Recall" in text
        assert "Novelty Detection" in text
        assert "Association Strength" in text
        assert "Energy Efficiency" in text

    def test_save_results(self, small_network, tmp_path):
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4)
        path = suite.save_results(results, str(tmp_path))
        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert "cross_modal_recall" in data
