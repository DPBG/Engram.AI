"""Cross-modal binding accuracy benchmark and CI regression tests (issue #130).

Verifies:
1. Ground-truth correlated-stimulus fixtures are generated
2. Precision/recall metrics are computed and JSON-serializable
3. Regression thresholds are enforced so binding accuracy cannot silently degrade
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neuromorphic.benchmarks import (
    BenchmarkSuite,
    CrossModalBindingAccuracyBenchmark,
    _binding_precision_recall,
)
from neuromorphic.binding_fixtures import (
    BINDING_ACCURACY_REGRESSION_THRESHOLDS,
    generate_correlated_stimulus_fixtures,
)
from neuromorphic.config import NeuromorphicConfig
from neuromorphic.network import NeuromorphicNetwork


@pytest.fixture
def small_network():
    """Minimal network for fast binding-accuracy tests."""
    cfg = NeuromorphicConfig.from_env()
    cfg.populations.brainstem = 50
    cfg.populations.reflex_arc = 30
    cfg.populations.sensory_cortex = 200
    cfg.populations.motor_cortex = 100
    cfg.populations.cerebellum = 50
    cfg.populations.association_cortex = 200
    cfg.populations.predictive_layer = 50
    cfg.populations.working_memory = 30
    cfg.populations.feature_layer = 0
    cfg.populations.concept_layer = 0
    cfg.populations.meta_controller = 0
    return NeuromorphicNetwork(cfg, seed=42)


class TestCorrelatedStimulusFixtures:
    def test_generates_correlated_and_decoy_pairs(self):
        fixtures = generate_correlated_stimulus_fixtures(n_pairs=4, seed=42)
        assert len(fixtures["correlated_pairs"]) == 4
        assert len(fixtures["decoy_pairs"]) == 4
        for pair in fixtures["correlated_pairs"]:
            assert "pair_id" in pair
            assert len(pair["visual"]) == 64 * 64
            assert len(pair["auditory"]) == 13
        for decoy in fixtures["decoy_pairs"]:
            assert decoy["visual_pair_id"] != decoy["auditory_pair_id"]

    def test_fixtures_deterministic(self):
        a = generate_correlated_stimulus_fixtures(3, seed=7)
        b = generate_correlated_stimulus_fixtures(3, seed=7)
        assert a["correlated_pairs"][0]["visual"] == b["correlated_pairs"][0]["visual"]

    def test_rejects_invalid_n_pairs(self):
        with pytest.raises(ValueError, match="at least 2"):
            generate_correlated_stimulus_fixtures(n_pairs=1)
        with pytest.raises(ValueError, match="auditory_size"):
            generate_correlated_stimulus_fixtures(n_pairs=14, auditory_size=13)

    def test_rejects_non_positive_sizes(self):
        with pytest.raises(ValueError, match="must be positive"):
            generate_correlated_stimulus_fixtures(visual_size=0)


class TestBindingPrecisionRecall:
    def test_perfect_predictions(self):
        matrix = [[1.0, 0.1], [0.2, 1.0]]
        pr = _binding_precision_recall(
            [{"pair_id": "a"}, {"pair_id": "b"}],
            matrix,
        )
        assert pr["precision"] == 1.0
        assert pr["recall"] == 1.0
        assert pr["f1"] == 1.0

    def test_zero_detection(self):
        matrix = [[0.0, 0.0], [0.0, 0.0]]
        pr = _binding_precision_recall([{"pair_id": "a"}, {"pair_id": "b"}], matrix)
        assert pr["recall"] == 0.0
        assert pr["precision"] == 0.0


class TestCrossModalBindingAccuracyBenchmark:
    def test_produces_metrics(self, small_network):
        bench = CrossModalBindingAccuracyBenchmark(small_network)
        result = bench.run(n_pairs=3, training_reps=5, steps_per_pair=10, seed=42)
        for key in (
            "precision",
            "recall",
            "f1",
            "binding_strength_after",
            "matched_coupling_mean",
            "decoy_coupling_mean",
            "matched_to_decoy_ratio",
            "coupling_matrix",
        ):
            assert key in result
        assert result["pairs_tested"] == 3

    def test_clamps_n_pairs_to_minimum(self, small_network):
        result = CrossModalBindingAccuracyBenchmark(small_network).run(
            n_pairs=1,
            training_reps=2,
            steps_per_pair=6,
            seed=42,
        )
        assert result["pairs_tested"] == 2

    def test_results_json_serializable(self, small_network):
        result = CrossModalBindingAccuracyBenchmark(small_network).run(
            n_pairs=2,
            training_reps=2,
            steps_per_pair=6,
            seed=42,
        )
        serialized = json.dumps(result)
        loaded = json.loads(serialized)
        assert loaded["precision"] == result["precision"]

    def test_saves_to_benchmarks_dir(self, small_network, tmp_path):
        bench = CrossModalBindingAccuracyBenchmark(small_network)
        result = bench.run(n_pairs=2, training_reps=2, steps_per_pair=6, seed=42)
        suite = BenchmarkSuite(small_network)
        path = suite.save_results({"cross_modal_binding_accuracy": result}, str(tmp_path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert "cross_modal_binding_accuracy" in data


class TestBindingAccuracyRegression:
    """CI gate: fail if cross-modal binding accuracy drops below baseline."""

    @pytest.fixture
    def benchmark_result(self, small_network):
        return CrossModalBindingAccuracyBenchmark(small_network).run(
            n_pairs=4,
            training_reps=20,
            steps_per_pair=20,
            seed=42,
        )

    def test_binding_strength_after_training(self, benchmark_result):
        threshold = BINDING_ACCURACY_REGRESSION_THRESHOLDS["binding_strength_min"]
        assert benchmark_result["binding_strength_after"] >= threshold

    def test_cross_modal_neurons_emerge(self, benchmark_result):
        threshold = BINDING_ACCURACY_REGRESSION_THRESHOLDS["n_cross_modal_min"]
        assert benchmark_result["n_cross_modal_after"] >= threshold

    def test_matched_coupling_exceeds_decoy(self, benchmark_result):
        threshold = BINDING_ACCURACY_REGRESSION_THRESHOLDS["matched_coupling_ratio_min"]
        assert benchmark_result["matched_to_decoy_ratio"] >= threshold

    def test_precision_recall_reported(self, benchmark_result):
        assert (
            benchmark_result["precision"] >= BINDING_ACCURACY_REGRESSION_THRESHOLDS["precision_min"]
        )
        assert benchmark_result["recall"] >= BINDING_ACCURACY_REGRESSION_THRESHOLDS["recall_min"]
        assert benchmark_result["f1"] >= BINDING_ACCURACY_REGRESSION_THRESHOLDS["f1_min"]

    def test_binding_strength_increases(self, benchmark_result):
        assert (
            benchmark_result["binding_strength_after"]
            >= benchmark_result["binding_strength_before"]
        )


class TestDecoyRealism:
    """Issue #323: verify decoy difficulty and confirm the hard-decoy variant works."""

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

    def test_easy_decoys_have_low_auditory_similarity(self):
        """Easy decoys (default) must be clearly distinguishable — cos_sim < 0.5."""
        fixtures = generate_correlated_stimulus_fixtures(n_pairs=4, seed=42, hard_decoys=False)
        matched = {p["pair_id"]: p["auditory"] for p in fixtures["correlated_pairs"]}
        for decoy in fixtures["decoy_pairs"]:
            sim = self._cosine_sim(matched[decoy["visual_pair_id"]], decoy["auditory"])
            assert sim < 0.5, (
                f"{decoy['pair_id']}: easy decoy is too similar to match (cos_sim={sim:.3f}); "
                "decoy generation may have drifted toward the matched signature"
            )

    def test_hard_decoys_have_high_auditory_similarity(self):
        """Hard decoys (noise-perturbed) must stay close to matched — cos_sim > 0.7."""
        fixtures = generate_correlated_stimulus_fixtures(n_pairs=4, seed=42, hard_decoys=True)
        matched = {p["pair_id"]: p["auditory"] for p in fixtures["correlated_pairs"]}
        for decoy in fixtures["decoy_pairs"]:
            sim = self._cosine_sim(matched[decoy["visual_pair_id"]], decoy["auditory"])
            assert sim > 0.7, (
                f"{decoy['pair_id']}: hard decoy is not similar enough to match "
                f"(cos_sim={sim:.3f}); increase σ or reduce noise"
            )

    def test_hard_decoys_are_deterministic(self):
        """Hard decoy generation must be deterministic for the same seed."""
        a = generate_correlated_stimulus_fixtures(n_pairs=4, seed=42, hard_decoys=True)
        b = generate_correlated_stimulus_fixtures(n_pairs=4, seed=42, hard_decoys=True)
        for da, db in zip(a["decoy_pairs"], b["decoy_pairs"]):
            assert da["auditory"] == db["auditory"], (
                f"{da['pair_id']}: hard decoy is not reproducible across calls with same seed"
            )

    def test_hard_decoys_differ_across_seeds(self):
        """Different seeds must produce different hard decoys (RNG is actually used)."""
        a = generate_correlated_stimulus_fixtures(n_pairs=4, seed=0, hard_decoys=True)
        b = generate_correlated_stimulus_fixtures(n_pairs=4, seed=99, hard_decoys=True)
        assert a["decoy_pairs"][0]["auditory"] != b["decoy_pairs"][0]["auditory"]

    def test_fixture_output_records_hard_decoy_flag(self):
        """Return dict must carry hard_decoys so consumers know which variant they got."""
        assert generate_correlated_stimulus_fixtures(n_pairs=2, hard_decoys=True)["hard_decoys"] is True
        assert (
            generate_correlated_stimulus_fixtures(n_pairs=2, hard_decoys=False)["hard_decoys"]
            is False
        )

    def test_easy_and_hard_decoy_correlated_pairs_are_identical(self):
        """Switching hard_decoys must not alter the matched correlated pairs."""
        easy = generate_correlated_stimulus_fixtures(n_pairs=4, seed=42, hard_decoys=False)
        hard = generate_correlated_stimulus_fixtures(n_pairs=4, seed=42, hard_decoys=True)
        assert easy["correlated_pairs"] == hard["correlated_pairs"]
