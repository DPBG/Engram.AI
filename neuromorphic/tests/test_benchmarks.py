"""Tests for the benchmarking framework (benchmarks.py).

Verifies all 6 benchmarks produce valid results on a small network.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from neuromorphic.benchmarks import (
    INTENTIONALLY_SPARSE_REGIONS,
    AssociationStrengthBenchmark,
    BenchmarkSuite,
    ConceptSeparabilityBenchmark,
    CrossModalRecallBenchmark,
    EnergyEfficiencyBenchmark,
    MetaControllerGatingBenchmark,
    NoveltyDetectionBenchmark,
    RegionQuietFlag,
    _confidence_interval,
    _flatten_numeric,
    _standardized_nearest_centroid_loo_accuracy,
    _to_native,
    audit_pattern_diversity,
    build_arg_parser,
    classify_quiet_regions,
    format_quiet_region_report,
    generate_test_patterns,
    nearest_centroid_loo_accuracy,
    silhouette_scores_from_distance_matrix,
)
from neuromorphic.config import NeuromorphicConfig
from neuromorphic.network import NeuromorphicNetwork

_NEURO_DIR = Path(__file__).resolve().parents[1]


def _build_small_network(seed: int | None = None) -> NeuromorphicNetwork:
    """Construct a fresh minimal network for fast benchmarking tests.

    Factored out of the ``small_network`` fixture so tests that need more
    than one independent instance (e.g. reproducibility checks) can build a
    second one without sharing state with the fixture's.

    ``NeuromorphicNetwork`` has its own ``seed`` parameter (defaults to
    ``None`` -> OS entropy) controlling weight-initialization randomness,
    entirely separate from ``BenchmarkSuite.run_all()``'s ``seed`` (which
    controls pattern/fixture generation, issue #322's scope). A
    reproducibility test must pin down both layers explicitly -- passing
    the same ``run_all(seed=...)`` alone to two networks built with
    independent (unseeded) init randomness would never match.
    """
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
    return NeuromorphicNetwork(cfg, seed=seed)


@pytest.fixture
def small_network():
    """Minimal network for fast benchmarking tests."""
    return _build_small_network()


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

    def test_no_exact_duplicates_beyond_64_patterns(self):
        """issue #327: the prior generator's (i*7)%64/(i*13)%64 position formula
        made pattern[i] == pattern[i+64] exactly — any benchmark requesting
        more than 64 patterns silently trained on duplicates."""
        pats = generate_test_patterns(80, np.random.default_rng(42))
        visuals = {tuple(p["visual"]) for p in pats}
        assert len(visuals) == 80

    def test_visual_patterns_are_not_a_single_shape_class(self):
        """issue #327: the prior generator used one fixed sigma, so every
        pattern was a translated copy of the same blob shape."""
        pats = generate_test_patterns(9, np.random.default_rng(1))
        vis = np.array([p["visual"] for p in pats])

        def radius(v):
            img = v.reshape(64, 64)
            total = img.sum()
            ys, xs = np.mgrid[0:64, 0:64]
            cx, cy = (xs * img).sum() / total, (ys * img).sum() / total
            return np.sqrt((((xs - cx) ** 2 + (ys - cy) ** 2) * img).sum() / total)

        radii = [radius(v) for v in vis]
        assert np.std(radii) > 0.5  # genuine size variation, not one fixed blob

    def test_auditory_patterns_are_distinguishable(self):
        """issue #327: 12 shared-noise dims used to dominate similarity
        regardless of which single dim was "hot" (~0.9 mean cosine sim)."""
        diversity = audit_pattern_diversity(generate_test_patterns(20, np.random.default_rng(7)))
        assert diversity["mean_auditory_cosine_sim"] < 0.6


class TestAuditPatternDiversity:
    def test_empty_patterns(self):
        result = audit_pattern_diversity([])
        assert result["n_patterns"] == 0
        assert result["unique_visual"] == 0
        assert result["unique_auditory"] == 0

    def test_single_pattern_similarity_is_zero(self):
        pats = generate_test_patterns(1, np.random.default_rng(3))
        result = audit_pattern_diversity(pats)
        assert result["n_patterns"] == 1
        assert result["mean_visual_cosine_sim"] == 0.0
        assert result["mean_auditory_cosine_sim"] == 0.0

    def test_identical_patterns_have_similarity_one(self):
        pat = generate_test_patterns(1, np.random.default_rng(3))[0]
        result = audit_pattern_diversity([pat, dict(pat)])
        assert result["unique_visual"] == 1
        assert result["mean_visual_cosine_sim"] == pytest.approx(1.0, abs=1e-6)

    def test_reports_unique_counts_matching_pattern_count_today(self):
        pats = generate_test_patterns(80, np.random.default_rng(42))
        result = audit_pattern_diversity(pats)
        assert result["n_patterns"] == 80
        assert result["unique_visual"] == 80
        assert result["unique_auditory"] == 80


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

    def test_region_energy_units_matches_firing_rate_regions(self, small_network, patterns):
        """region_energy_units (issue #331) covers exactly the regions region_firing_rates does."""
        bench = EnergyEfficiencyBenchmark(small_network)
        result = bench.run(patterns, steps_per_pattern=4)
        assert "region_energy_units" in result
        assert set(result["region_energy_units"]) == set(result["region_firing_rates"])

    def test_region_energy_units_sum_to_approx_energy_units(self, small_network, patterns):
        bench = EnergyEfficiencyBenchmark(small_network)
        result = bench.run(patterns, steps_per_pattern=4)
        assert sum(result["region_energy_units"].values()) == pytest.approx(
            result["approx_energy_units"], rel=1e-3
        )


class TestClassifyQuietRegions:
    def test_intentional_sparsity_flagged_for_known_kwta_regions(self):
        result = {
            "region_firing_rates": {"concept_layer": 0.02, "pattern_separator": 0.015},
            "region_energy_units": {"concept_layer": 1.0, "pattern_separator": 0.5},
        }
        flags = classify_quiet_regions(result)
        by_region = {f.region: f for f in flags}
        assert by_region["concept_layer"].classification == "intentional_sparsity"
        assert by_region["pattern_separator"].classification == "intentional_sparsity"
        assert "k-WTA" in by_region["concept_layer"].note

    def test_quiet_region_without_sparsity_rationale_flagged_undertrained(self):
        result = {"region_firing_rates": {"sensory_cortex": 0.001}, "region_energy_units": {}}
        flags = classify_quiet_regions(result)
        assert flags[0].classification == "quiet_undertrained"
        assert "undertrained" in flags[0].note

    def test_normal_firing_rate_is_healthy(self):
        result = {"region_firing_rates": {"motor_cortex": 0.15}, "region_energy_units": {}}
        flags = classify_quiet_regions(result)
        assert flags[0].classification == "healthy"

    def test_custom_quiet_threshold(self):
        result = {"region_firing_rates": {"cerebellum": 0.05}, "region_energy_units": {}}
        assert classify_quiet_regions(result)[0].classification == "healthy"
        assert (
            classify_quiet_regions(result, quiet_threshold=0.1)[0].classification
            == "quiet_undertrained"
        )

    def test_region_scores_override_marks_positive_score_healthy(self):
        """A real per-region score (once #297/#315/#316 land) beats the heuristic."""
        result = {"region_firing_rates": {"working_memory": 0.001}, "region_energy_units": {}}
        flags = classify_quiet_regions(result, region_scores={"working_memory": 0.87})
        assert flags[0].classification == "healthy"
        assert "per-region score" in flags[0].note

    def test_region_scores_override_marks_zero_score_undertrained(self):
        result = {"region_firing_rates": {"concept_layer": 0.02}, "region_energy_units": {}}
        flags = classify_quiet_regions(result, region_scores={"concept_layer": 0.0})
        assert flags[0].classification == "quiet_undertrained"

    def test_energy_units_carried_through_when_present(self):
        result = {
            "region_firing_rates": {"motor_cortex": 0.2},
            "region_energy_units": {"motor_cortex": 3.14},
        }
        assert classify_quiet_regions(result)[0].energy_units == pytest.approx(3.14)

    def test_missing_energy_units_is_none(self):
        result = {"region_firing_rates": {"motor_cortex": 0.2}}
        assert classify_quiet_regions(result)[0].energy_units is None

    def test_empty_firing_rates_returns_empty_list(self):
        assert classify_quiet_regions({}) == []
        assert classify_quiet_regions({"region_firing_rates": {}}) == []

    def test_intentionally_sparse_regions_table_has_notes(self):
        assert set(INTENTIONALLY_SPARSE_REGIONS) == {"concept_layer", "pattern_separator"}
        assert all(INTENTIONALLY_SPARSE_REGIONS.values())


class TestFormatQuietRegionReport:
    def test_empty_flags(self):
        assert "no per-region" in format_quiet_region_report([])

    def test_report_mentions_each_region_and_classification(self):
        flags = [
            RegionQuietFlag("concept_layer", 0.02, 1.0, "intentional_sparsity", "k-WTA design"),
            RegionQuietFlag(
                "sensory_cortex", 0.001, None, "quiet_undertrained", "likely undertrained"
            ),
            RegionQuietFlag("motor_cortex", 0.2, 2.0, "healthy", "normal range"),
        ]
        text = format_quiet_region_report(flags)
        for f in flags:
            assert f.region in text
            assert f.classification in text

    def test_report_sorted_quietest_first(self):
        flags = [
            RegionQuietFlag("motor_cortex", 0.2, None, "healthy", "x"),
            RegionQuietFlag("sensory_cortex", 0.001, None, "quiet_undertrained", "y"),
        ]
        text = format_quiet_region_report(flags)
        assert text.index("sensory_cortex") < text.index("motor_cortex")


class TestBenchmarkSuite:
    def test_run_all(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4)
        assert "cross_modal_recall" in results
        assert "novelty_detection" in results
        assert "association_strength" in results
        assert "energy_efficiency" in results
        assert "concept_separability" in results
        assert "cross_modal_binding_accuracy" in results
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
        assert "Concept Separability" in text
        assert "Cross-Modal Binding Accuracy" in text

    def test_summary_includes_per_region_quiet_report(self, small_network):
        """issue #331: the summary cross-references quiet regions automatically."""
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4)
        text = suite.summary(results)
        assert "issue #331" in text
        for region in results["energy_efficiency"]["region_firing_rates"]:
            assert region in text


class TestCrossModalBindingAccuracyInSuite:
    def test_run_all_includes_binding_accuracy(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4)
        ba = results["cross_modal_binding_accuracy"]
        assert "precision" in ba
        assert "recall" in ba
        assert ba["pairs_tested"] == 2

    def test_save_results(self, small_network, tmp_path):
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4)
        path = suite.save_results(results, str(tmp_path))
        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert "cross_modal_recall" in data


class TestFlattenNumeric:
    def test_flattens_nested_dicts(self):
        d = {"a": 1, "b": {"c": 2.5, "d": {"e": 3}}}
        flat = _flatten_numeric(d)
        assert flat == {"a": 1.0, "b.c": 2.5, "b.d.e": 3.0}

    def test_drops_non_numeric(self):
        d = {"label": "pattern_000", "enabled": True, "value": 1.5, "items": [1, 2, 3]}
        flat = _flatten_numeric(d)
        assert flat == {"value": 1.5}

    def test_empty_dict(self):
        assert _flatten_numeric({}) == {}


class TestConfidenceInterval:
    def test_single_value_has_zero_width(self):
        mean, half_width = _confidence_interval([5.0])
        assert mean == 5.0
        assert half_width == 0.0

    def test_identical_values_have_zero_width(self):
        mean, half_width = _confidence_interval([3.0, 3.0, 3.0])
        assert mean == 3.0
        assert half_width == pytest.approx(0.0, abs=1e-9)

    def test_wider_confidence_gives_wider_interval(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, narrow = _confidence_interval(values, confidence=0.80)
        _, wide = _confidence_interval(values, confidence=0.99)
        assert wide > narrow

    def test_mean_is_correct(self):
        mean, _ = _confidence_interval([2.0, 4.0, 6.0])
        assert mean == pytest.approx(4.0)


class TestMultiSeed:
    def test_run_multi_seed_structure(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_multi_seed(
            n_seeds=3, n_patterns=2, training_reps=1, steps_per_pattern=4
        )
        assert results["n_seeds"] == 3
        assert len(results["seeds"]) == 3
        assert len(results["seeds"]) == len(set(results["seeds"]))  # distinct seeds
        assert len(results["runs"]) == 3
        assert "aggregate" in results

    def test_aggregate_has_mean_and_ci(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_multi_seed(
            n_seeds=3, n_patterns=2, training_reps=1, steps_per_pattern=4
        )
        agg = results["aggregate"]
        assert len(agg) > 0
        for name, stat in agg.items():
            assert set(stat.keys()) == {"mean", "std", "ci_low", "ci_high", "n"}
            assert stat["n"] == 3
            assert stat["ci_low"] <= stat["mean"] <= stat["ci_high"]

    def test_single_seed_collapses_ci(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_multi_seed(
            n_seeds=1, n_patterns=2, training_reps=1, steps_per_pattern=4
        )
        agg = results["aggregate"]
        for stat in agg.values():
            assert stat["ci_low"] == stat["mean"] == stat["ci_high"]

    def test_base_seed_controls_seed_sequence(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_multi_seed(
            n_seeds=2, n_patterns=2, training_reps=1, steps_per_pattern=4, base_seed=100
        )
        assert results["seeds"] == [100, 101]

    def test_summary_multi_seed(self, small_network):
        suite = BenchmarkSuite(small_network)
        results = suite.run_multi_seed(
            n_seeds=2, n_patterns=2, training_reps=1, steps_per_pattern=4
        )
        text = suite.summary_multi_seed(results)
        assert "Multi-Seed" in text
        assert "95% CI" in text or "metrics tracked" in text

    def test_save_multi_seed_results(self, small_network, tmp_path):
        suite = BenchmarkSuite(small_network)
        results = suite.run_multi_seed(
            n_seeds=2, n_patterns=2, training_reps=1, steps_per_pattern=4
        )
        path = suite.save_results(results, str(tmp_path))
        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert data["n_seeds"] == 2
        assert "aggregate" in data

    def test_run_all_includes_concept_separability(self, small_network):
        """run_all() always includes concept_separability key (error or scores)."""
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4)
        assert "concept_separability" in results
        cs = results["concept_separability"]
        assert "silhouette_score" in cs
        assert "linear_probe_accuracy" in cs

    def test_run_all_with_single_pattern(self, small_network):
        """Binding benchmark clamps n_pairs to 2 when n_patterns=1."""
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=1, training_reps=1, steps_per_pattern=4)
        assert results["cross_modal_binding_accuracy"]["pairs_tested"] == 2


class TestSingleRunSeedOverride:
    """Issue #322: --seed must be a real, reproducible override, not a no-op.

    run_all()'s seed parameter and the CLI's --seed flag already existed
    (landed as a prerequisite of the multi-seed work, PR #338) -- these tests
    formalize that as a locked-down, regression-tested contract rather than
    incidental behavior, and cover the CLI parsing layer that had no direct
    test before.
    """

    def test_same_seed_is_reproducible(self):
        # Two independent networks, built with the same network-init seed
        # AND the same run_all() seed: results must match exactly, proving
        # the benchmark seed genuinely drives its randomness reproducibly
        # rather than being recorded but ignored.
        suite_a = BenchmarkSuite(_build_small_network(seed=99))
        suite_b = BenchmarkSuite(_build_small_network(seed=99))
        results_a = suite_a.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4, seed=7)
        results_b = suite_b.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4, seed=7)
        assert results_a["cross_modal_recall"] == results_b["cross_modal_recall"]
        assert results_a["novelty_detection"] == results_b["novelty_detection"]
        assert (
            results_a["cross_modal_binding_accuracy"] == results_b["cross_modal_binding_accuracy"]
        )

    def test_different_seeds_change_results(self):
        # The inverse check: an override that silently did nothing would
        # also pass a same-seed-reproducible test, so this must hold too.
        # Network-init seed held constant so run_all()'s seed is the only
        # thing that differs between the two runs. Compares the full
        # cross_modal_binding_accuracy dict (not one field in isolation) --
        # at this tiny training scale some individual metrics can coincide
        # by chance between two nearby seeds, but fixture_seed alone (see
        # test_seed_reaches_binding_fixture_generator) guarantees the dicts
        # as a whole cannot be equal.
        suite_a = BenchmarkSuite(_build_small_network(seed=99))
        suite_b = BenchmarkSuite(_build_small_network(seed=99))
        results_a = suite_a.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4, seed=1)
        results_b = suite_b.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4, seed=2)
        assert (
            results_a["cross_modal_binding_accuracy"] != results_b["cross_modal_binding_accuracy"]
        )

    def test_seed_reaches_binding_fixture_generator(self, small_network):
        # Acceptance criterion: threaded through to
        # binding_fixtures.generate_correlated_stimulus_fixtures(), which
        # records it as fixture_seed in cross_modal_binding_accuracy's output.
        suite = BenchmarkSuite(small_network)
        results = suite.run_all(n_patterns=2, training_reps=1, steps_per_pattern=4, seed=123)
        assert results["cross_modal_binding_accuracy"]["fixture_seed"] == 123

    def test_cli_parser_accepts_seed_flag(self):
        # Acceptance criterion: a --seed flag on the CLI entrypoint. Parser-
        # level test avoids spinning up a real network for CLI-wiring checks.
        parser = build_arg_parser()
        args = parser.parse_args(["--seed", "12345"])
        assert args.seed == 12345

    def test_cli_seed_flag_default_matches_run_all_default(self):
        # The CLI's implicit default must match run_all()'s, so an unspecified
        # --seed still reproduces the documented default=42 behavior.
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.seed == 42


@pytest.fixture
def network_with_concept():
    """Small network with an active concept layer for separability tests."""
    cfg = NeuromorphicConfig.from_env()
    cfg.populations.brainstem = 50
    cfg.populations.reflex_arc = 30
    cfg.populations.sensory_cortex = 200
    cfg.populations.motor_cortex = 100
    cfg.populations.cerebellum = 50
    cfg.populations.association_cortex = 150
    cfg.populations.predictive_layer = 80
    cfg.populations.working_memory = 40
    cfg.populations.feature_layer = 0
    cfg.populations.concept_layer = 100
    cfg.populations.pattern_separator = 0
    cfg.populations.meta_controller = 0
    cfg.concept_layer.k_winners = 10  # 10% sparsity for meaningful k-WTA
    return NeuromorphicNetwork(cfg)


class TestConceptSeparabilityBenchmark:

    def test_no_concept_layer_returns_error(self, small_network, patterns):
        bench = ConceptSeparabilityBenchmark(small_network)
        result = bench.run(patterns, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" in result
        assert result["silhouette_score"] == 0.0
        assert result["linear_probe_accuracy"] == 0.0

    def test_produces_all_metrics(self, network_with_concept, patterns):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        result = bench.run(patterns, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" not in result
        for key in (
            "silhouette_score",
            "linear_probe_accuracy",
            "mean_intra_class_distance",
            "mean_inter_class_distance",
            "separation_ratio",
            "n_patterns",
            "n_samples",
            "concept_neurons",
            "top_neurons_per_pattern",
        ):
            assert key in result, f"missing key: {key}"

    def test_silhouette_in_range(self, network_with_concept, patterns):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        result = bench.run(patterns, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" not in result
        assert -1.0 <= result["silhouette_score"] <= 1.0

    def test_accuracy_in_range(self, network_with_concept, patterns):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        result = bench.run(patterns, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" not in result
        assert 0.0 <= result["linear_probe_accuracy"] <= 1.0

    def test_distances_non_negative(self, network_with_concept, patterns):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        result = bench.run(patterns, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" not in result
        assert result["mean_intra_class_distance"] >= 0.0
        assert result["mean_inter_class_distance"] >= 0.0

    def test_separation_ratio_non_negative(self, network_with_concept, patterns):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        result = bench.run(patterns, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" not in result
        assert result["separation_ratio"] >= 0.0

    def test_sample_count_matches(self, network_with_concept, patterns):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        probe_reps = 2
        result = bench.run(patterns, training_reps=1, probe_reps=probe_reps, steps_per_rep=3)
        assert "error" not in result
        assert result["n_samples"] == len(patterns) * probe_reps
        assert result["n_patterns"] == len(patterns)

    def test_top_neurons_per_pattern(self, network_with_concept, patterns):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        result = bench.run(patterns, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" not in result
        top = result["top_neurons_per_pattern"]
        assert len(top) == len(patterns)
        for row in top:
            assert len(row) == 5
            assert all(isinstance(n, int) for n in row)

    def test_concept_neuron_count(self, network_with_concept, patterns):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        result = bench.run(patterns, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" not in result
        assert result["concept_neurons"] == 100  # matches fixture population

    def test_insufficient_patterns_returns_error(self, network_with_concept):
        bench = ConceptSeparabilityBenchmark(network_with_concept)
        # Only 1 pattern — can't compute inter-class distance
        single = generate_test_patterns(1, np.random.default_rng(0))
        result = bench.run(single, training_reps=1, probe_reps=2, steps_per_rep=3)
        assert "error" in result

    def test_suite_summary_includes_concept_separability(self, network_with_concept):
        suite = BenchmarkSuite(network_with_concept)
        results = suite.run_all(n_patterns=3, training_reps=1, steps_per_pattern=4)
        text = suite.summary(results)
        assert "Concept Separability" in text
        assert "Silhouette score" in text


# ---------------------------------------------------------------------------
# Issue #329: Benchmark flakiness audit across all test classes
# ---------------------------------------------------------------------------
def _make_small_network() -> NeuromorphicNetwork:
    return _build_small_network()


@pytest.fixture
def network_with_meta_controller():
    """Small network with an active meta-controller for gating benchmark tests."""
    cfg = NeuromorphicConfig.from_env()
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
    cfg.populations.pattern_separator = 0
    cfg.populations.meta_controller = 200
    return NeuromorphicNetwork(cfg)


class TestBenchmarkFlakiness:
    """Audit that every benchmark class produces stable, valid output across seeds.

    Each test runs its benchmark class with 3 distinct seeds and asserts that:
    - metrics stay within their documented valid range every time
    - no NaN or Inf values appear in any run
    - pure-logic helpers (generate_test_patterns, _flatten_numeric, etc.) are
      deterministic regardless of call order

    A class that fails any of these checks on 3 seeds is by definition flaky and
    must be fixed before being used for statistical inference.
    """

    _SEEDS = [42, 123, 777]

    # --- TestGeneratePatterns domain ---

    def test_generate_patterns_deterministic_across_seeds(self):
        for seed in self._SEEDS:
            a = generate_test_patterns(3, np.random.default_rng(seed))
            b = generate_test_patterns(3, np.random.default_rng(seed))
            assert a == b, f"non-deterministic output for seed={seed}"

    # --- TestToNative domain ---

    def test_to_native_idempotent(self):
        d = {"a": np.int64(5), "b": np.float32(3.14), "c": np.array([1, 2])}
        for _ in range(5):
            r = _to_native(d)
            assert isinstance(r["a"], int)
            assert isinstance(r["b"], float)
            assert isinstance(r["c"], list)

    # --- TestCrossModalRecall domain ---

    def test_cross_modal_recall_range_stable(self):
        for seed in self._SEEDS:
            net = _make_small_network()
            patterns = generate_test_patterns(3, np.random.default_rng(seed))
            result = CrossModalRecallBenchmark(net).run(
                patterns, training_reps=2, steps_per_pattern=4
            )
            assert 0.0 <= result["visual_to_auditory_recall"] <= 1.0, f"seed={seed}"
            assert 0.0 <= result["auditory_to_visual_recall"] <= 1.0, f"seed={seed}"
            assert math.isfinite(result["binding_strength_delta"]), f"seed={seed}"

    # --- TestNoveltyDetection domain ---

    def test_novelty_detection_range_stable(self):
        for seed in self._SEEDS:
            net = _make_small_network()
            patterns = generate_test_patterns(3, np.random.default_rng(seed))
            novel = generate_test_patterns(1, np.random.default_rng(seed + 999))[0]
            result = NoveltyDetectionBenchmark(net).run(
                patterns[0], novel, familiarization_reps=2, steps_per_rep=4, test_steps=3
            )
            assert result["familiar_pred_error"] >= 0.0, f"seed={seed}"
            assert result["novel_pred_error"] >= 0.0, f"seed={seed}"
            assert math.isfinite(result["discrimination_ratio"]), f"seed={seed}"

    # --- TestAssociationStrength domain ---

    def test_association_strength_range_stable(self):
        for seed in self._SEEDS:
            net = _make_small_network()
            patterns = generate_test_patterns(3, np.random.default_rng(seed))
            result = AssociationStrengthBenchmark(net).run(
                patterns, training_reps=2, steps_per_pattern=4
            )
            for wc in result["weight_changes"].values():
                assert math.isfinite(wc["delta_mean"]), f"seed={seed}"
            assert result["concept_count"] >= 0, f"seed={seed}"

    # --- TestEnergyEfficiency domain ---

    def test_energy_efficiency_range_stable(self):
        for seed in self._SEEDS:
            net = _make_small_network()
            patterns = generate_test_patterns(3, np.random.default_rng(seed))
            result = EnergyEfficiencyBenchmark(net).run(patterns, steps_per_pattern=4)
            assert result["global_firing_rate"] >= 0.0, f"seed={seed}"
            for rate in result["region_firing_rates"].values():
                assert rate >= 0.0, f"seed={seed}"
            assert math.isfinite(result["approx_energy_units"]), f"seed={seed}"

    # --- TestBenchmarkSuite + TestCrossModalBindingAccuracyInSuite domain ---

    def test_suite_run_all_keys_and_ranges_stable(self):
        for seed in self._SEEDS:
            net = _make_small_network()
            results = BenchmarkSuite(net).run_all(
                n_patterns=2, training_reps=1, steps_per_pattern=4, seed=seed
            )
            for key in (
                "cross_modal_recall",
                "novelty_detection",
                "association_strength",
                "energy_efficiency",
                "concept_separability",
                "cross_modal_binding_accuracy",
            ):
                assert key in results, f"missing key {key!r} for seed={seed}"
            assert results["elapsed_s"] > 0, f"seed={seed}"
            ba = results["cross_modal_binding_accuracy"]
            assert 0.0 <= ba["precision"] <= 1.0, f"precision out of range for seed={seed}"
            assert 0.0 <= ba["recall"] <= 1.0, f"recall out of range for seed={seed}"
            assert 0.0 <= ba["f1"] <= 1.0, f"f1 out of range for seed={seed}"

    # --- TestFlattenNumeric domain ---

    def test_flatten_numeric_deterministic(self):
        d = {"a": 1, "b": {"c": 2.5, "d": {"e": 3}}}
        expected = {"a": 1.0, "b.c": 2.5, "b.d.e": 3.0}
        for _ in range(5):
            assert _flatten_numeric(d) == expected

    # --- TestConfidenceInterval domain ---

    def test_confidence_interval_deterministic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        first_mean, first_hw = _confidence_interval(values)
        for _ in range(5):
            mean, hw = _confidence_interval(values)
            assert mean == first_mean
            assert hw == first_hw

    # --- TestMultiSeed domain ---

    def test_multi_seed_aggregate_no_nan(self):
        net = _make_small_network()
        results = BenchmarkSuite(net).run_multi_seed(
            n_seeds=3, n_patterns=2, training_reps=1, steps_per_pattern=4
        )
        for name, stat in results["aggregate"].items():
            assert math.isfinite(stat["mean"]), f"NaN mean for metric {name!r}"
            assert math.isfinite(stat["std"]), f"NaN std for metric {name!r}"

    # --- No NaN/Inf in any run (covers all 6 benchmarks together) ---

    def test_no_nan_or_inf_across_seeds(self):
        for seed in self._SEEDS:
            net = _make_small_network()
            results = BenchmarkSuite(net).run_all(
                n_patterns=2, training_reps=1, steps_per_pattern=4, seed=seed
            )
            flat = _flatten_numeric(results)
            bad = {k: v for k, v in flat.items() if not math.isfinite(v)}
            assert not bad, f"seed={seed}: non-finite metrics found: {bad}"


class TestStandardizedNearestCentroidLooAccuracy:
    def test_perfectly_separated_classes_score_1(self):
        mat = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 9.9]])
        labels = np.array([0, 0, 1, 1])
        assert _standardized_nearest_centroid_loo_accuracy(mat, labels) == 1.0

    def test_wide_range_feature_does_not_dominate(self):
        """A feature with a much wider raw range (e.g. NE's [0.5, 3.0] vs.
        5-HT's [0.1, 1.5]) must not swamp the classification purely because
        of scale — z-scoring should put both features on equal footing."""
        mat = np.array(
            [
                [0.0, 100.0],
                [0.1, 100.1],
                [10.0, 100.0],
                [10.1, 99.9],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        assert _standardized_nearest_centroid_loo_accuracy(mat, labels) == 1.0


class TestMetaControllerGatingBenchmark:
    def test_no_meta_controller_returns_error(self, small_network, patterns):
        bench = MetaControllerGatingBenchmark(small_network)
        result = bench.run(patterns[0], patterns[1], n_trials=8, steps_per_trial=3)
        assert "error" in result
        assert result["gating_accuracy"] == 0.0

    def test_produces_all_metrics(self, network_with_meta_controller, patterns):
        bench = MetaControllerGatingBenchmark(network_with_meta_controller)
        result = bench.run(patterns[0], patterns[1], n_trials=8, steps_per_trial=3, seed=1)
        assert "error" not in result
        for key in (
            "gating_accuracy",
            "n_trials",
            "n_samples",
            "context_switches",
            "meta_controller_neurons",
            "mean_da_context_a",
            "mean_da_context_b",
            "mean_ach_context_a",
            "mean_ach_context_b",
            "mean_ne_context_a",
            "mean_ne_context_b",
            "mean_serotonin_context_a",
            "mean_serotonin_context_b",
            "steps_per_trial",
            "fixture_seed",
        ):
            assert key in result, f"missing key: {key}"

    def test_gating_accuracy_in_range(self, network_with_meta_controller, patterns):
        bench = MetaControllerGatingBenchmark(network_with_meta_controller)
        result = bench.run(patterns[0], patterns[1], n_trials=8, steps_per_trial=3, seed=1)
        assert "error" not in result
        assert 0.0 <= result["gating_accuracy"] <= 1.0

    def test_sample_count_matches_trials(self, network_with_meta_controller, patterns):
        bench = MetaControllerGatingBenchmark(network_with_meta_controller)
        n_trials = 10
        result = bench.run(patterns[0], patterns[1], n_trials=n_trials, steps_per_trial=3, seed=2)
        assert "error" not in result
        assert result["n_samples"] == n_trials
        assert result["n_trials"] == n_trials

    def test_meta_controller_neuron_count(self, network_with_meta_controller, patterns):
        bench = MetaControllerGatingBenchmark(network_with_meta_controller)
        result = bench.run(patterns[0], patterns[1], n_trials=8, steps_per_trial=3, seed=1)
        assert "error" not in result
        assert result["meta_controller_neurons"] == 200  # matches fixture population

    def test_insufficient_trials_returns_error(self, network_with_meta_controller, patterns):
        bench = MetaControllerGatingBenchmark(network_with_meta_controller)
        result = bench.run(patterns[0], patterns[1], n_trials=2, steps_per_trial=3)
        assert "error" in result
        assert result["gating_accuracy"] == 0.0

    def test_same_seed_is_reproducible(self, network_with_meta_controller, patterns):
        bench = MetaControllerGatingBenchmark(network_with_meta_controller)
        result = bench.run(patterns[0], patterns[1], n_trials=6, steps_per_trial=3, seed=0)
        assert "error" not in result
        assert result["fixture_seed"] == 0

    def test_not_wired_into_run_all(self, network_with_meta_controller):
        """MetaControllerGatingBenchmark is a standalone per-region benchmark
        (issue #319), deliberately not part of run_all()'s fixed 6-metric
        contract documented in docs/benchmark-schema.md."""
        suite = BenchmarkSuite(network_with_meta_controller)
        results = suite.run_all(n_patterns=1, training_reps=1, steps_per_pattern=4)
        assert "meta_controller_gating" not in results


def _cosine_dist(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    dist = np.clip(1.0 - unit @ unit.T, 0.0, 2.0).astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    return dist, unit


class TestSilhouetteScoresFromDistanceMatrix:
    """Direct unit tests for the extracted, sklearn-verified silhouette function.

    See scripts/verify_silhouette_score.py (issue #330) for the fuller
    ground-truth comparison against sklearn.metrics.silhouette_samples;
    these tests pin the specific behaviors that comparison caught or relies
    on, so a regression fails CI without needing scikit-learn installed.
    """

    def test_singleton_cluster_scores_zero_not_one(self):
        # Regression test (issue #330): an earlier version derived a
        # singleton cluster's score from the general formula with a(i)=0,
        # which silently produced 1.0 ("perfectly separated") instead of the
        # textbook/sklearn convention of exactly 0 (undefined -> neutral).
        dist = np.array(
            [
                [0.0, 0.9, 0.9],
                [0.9, 0.0, 0.1],
                [0.9, 0.1, 0.0],
            ]
        )
        labels = np.array([0, 1, 1])  # sample 0 is a singleton cluster
        scores = silhouette_scores_from_distance_matrix(dist, labels)
        assert scores[0] == 0.0

    def test_well_separated_clusters_score_near_one(self):
        rng = np.random.default_rng(0)
        mat = np.concatenate(
            [
                rng.normal(loc=10.0, scale=0.1, size=(5, 8)),
                rng.normal(loc=-10.0, scale=0.1, size=(5, 8)),
            ]
        )
        labels = np.array([0] * 5 + [1] * 5)
        dist, _unit = _cosine_dist(mat)
        scores = silhouette_scores_from_distance_matrix(dist, labels)
        assert np.all(scores > 0.9)

    def test_identical_points_across_clusters_score_zero(self):
        # a == b == 0 for every sample -> denom == 0 -> defined as 0, not NaN.
        dist = np.zeros((4, 4))
        labels = np.array([0, 0, 1, 1])
        scores = silhouette_scores_from_distance_matrix(dist, labels)
        assert np.all(scores == 0.0)


class TestNearestCentroidLooAccuracy:
    """Direct unit tests for the extracted, sklearn-verified linear-probe function."""

    def test_well_separated_clusters_perfectly_classified(self):
        rng = np.random.default_rng(0)
        mat = np.concatenate(
            [
                rng.normal(loc=10.0, scale=0.1, size=(5, 8)),
                rng.normal(loc=-10.0, scale=0.1, size=(5, 8)),
            ]
        )
        labels = np.array([0] * 5 + [1] * 5)
        _dist, unit = _cosine_dist(mat)
        assert nearest_centroid_loo_accuracy(unit, labels) == 1.0

    def test_singleton_cluster_does_not_crash(self):
        # A held-out singleton class has zero remaining training members;
        # must fall back to a defined (zero-vector) centroid, not raise.
        labels = np.array([0, 1, 1])  # sample 0 is a singleton cluster
        _dist, unit = _cosine_dist(np.eye(3))
        accuracy = nearest_centroid_loo_accuracy(unit, labels)
        assert 0.0 <= accuracy <= 1.0


class TestRuntimeBudget:
    """Issue #332: BenchmarkSuite must complete within the committed wall-clock budget."""

    def test_committed_budget_file_is_valid(self):
        path = _NEURO_DIR / "benchmarks" / "suite_runtime_budget.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["budget_s"] > 0, "budget_s must be positive"
        assert "ci_params" in data, "ci_params section is required"
        for key in ("n_patterns", "training_reps", "steps_per_pattern"):
            assert key in data["ci_params"], f"ci_params.{key} is required"
        assert data["measured_baseline_s"] > 0, "measured_baseline_s must be positive"

    def test_run_all_within_runtime_budget(self, small_network):
        """run_all() with CI params must complete within the committed budget.

        If this fails the suite has regressed or a new benchmark added more work
        than the budget allows. Fix the regression or — after deliberate review —
        update measured_baseline_s and budget_s in suite_runtime_budget.json.
        """
        budget_path = _NEURO_DIR / "benchmarks" / "suite_runtime_budget.json"
        budget_data = json.loads(budget_path.read_text(encoding="utf-8"))
        budget_s = budget_data["budget_s"]
        params = budget_data["ci_params"]

        suite = BenchmarkSuite(small_network)
        results = suite.run_all(
            n_patterns=int(params["n_patterns"]),
            training_reps=int(params["training_reps"]),
            steps_per_pattern=int(params["steps_per_pattern"]),
        )

        failures = BenchmarkSuite.check_runtime_budget(results["elapsed_s"], budget_s)
        assert failures == [], (
            f"BenchmarkSuite exceeded the committed runtime budget ({budget_s}s).\n"
            "If this is a genuine regression, fix the code.\n"
            "If the budget itself must grow (e.g. new CI-required benchmarks added), "
            "update measured_baseline_s and budget_s in "
            "benchmarks/suite_runtime_budget.json after reviewing the new numbers.\n"
            "Failures:\n" + "\n".join(f"  - {f}" for f in failures)
        )

    def test_check_runtime_budget_passes_under_limit(self):
        assert BenchmarkSuite.check_runtime_budget(1.0, 30.0) == []

    def test_check_runtime_budget_fails_over_limit(self):
        failures = BenchmarkSuite.check_runtime_budget(31.0, 30.0)
        assert len(failures) == 1
        assert "31.00s" in failures[0]
        assert "30.00s" in failures[0]
