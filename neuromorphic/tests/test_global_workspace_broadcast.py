"""Tests for the GlobalWorkspace broadcast benchmark (issue #318).

Verifies:
1. The benchmark gracefully skips when GlobalWorkspace is disabled (default config).
2. On a workspace-enabled network, all required keys are returned.
3. Workspace firing rate is higher when injecting into association_cortex (wired to
   workspace) than into cerebellum (no direct path to workspace), confirming that
   the workspace responds specifically to its connected afferents.
4. All returned values are JSON-serializable.
"""

from __future__ import annotations

import json

import pytest

from neuromorphic.benchmarks import GlobalWorkspaceBroadcastBenchmark
from neuromorphic.config import NeuromorphicConfig
from neuromorphic.network import NeuromorphicNetwork

_REQUIRED_KEYS = {
    "global_workspace_enabled",
    "skipped",
    "source_region",
    "control_region",
    "inject_steps",
    "signal_amplitude",
    "baseline_workspace_rate",
    "source_workspace_rate",
    "control_workspace_rate",
    "workspace_response_ratio",
    "ignition_events_source",
    "ignition_events_control",
    "broadcast_detected",
    "downstream_rate_during_ignition",
}


def _workspace_cfg(global_workspace_n: int) -> NeuromorphicConfig:
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
    cfg.populations.meta_controller = 0
    cfg.populations.global_workspace = global_workspace_n
    return cfg


@pytest.fixture
def small_workspace_network():
    """Minimal network with GlobalWorkspace enabled."""
    return NeuromorphicNetwork(_workspace_cfg(100), seed=42)


@pytest.fixture
def no_workspace_network():
    """Minimal network with GlobalWorkspace disabled (population = 0)."""
    return NeuromorphicNetwork(_workspace_cfg(0), seed=42)


class TestGlobalWorkspaceBroadcastSkipsWhenDisabled:
    def test_returns_skipped_flag(self, no_workspace_network):
        result = GlobalWorkspaceBroadcastBenchmark(no_workspace_network).run()
        assert result["skipped"] is True

    def test_reports_workspace_disabled(self, no_workspace_network):
        result = GlobalWorkspaceBroadcastBenchmark(no_workspace_network).run()
        assert result["global_workspace_enabled"] is False

    def test_includes_reason_string(self, no_workspace_network):
        result = GlobalWorkspaceBroadcastBenchmark(no_workspace_network).run()
        assert "reason" in result
        assert isinstance(result["reason"], str)

    def test_skipped_result_is_json_serializable(self, no_workspace_network):
        result = GlobalWorkspaceBroadcastBenchmark(no_workspace_network).run()
        json.dumps(result)


class TestGlobalWorkspaceBroadcastStructure:
    @pytest.fixture
    def result(self, small_workspace_network):
        return GlobalWorkspaceBroadcastBenchmark(small_workspace_network).run(
            inject_steps=15, warmup_steps=5, signal_amplitude=3.0, seed=42
        )

    def test_not_skipped(self, result):
        assert result["skipped"] is False
        assert result["global_workspace_enabled"] is True

    def test_returns_all_required_keys(self, result):
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_result_is_json_serializable(self, result):
        json.dumps(result)

    def test_source_and_control_region_labels(self, result):
        assert result["source_region"] == "association_cortex"
        assert result["control_region"] == "cerebellum"

    def test_inject_steps_echoed(self, result):
        assert result["inject_steps"] == 15

    def test_signal_amplitude_echoed(self, result):
        assert result["signal_amplitude"] == 3.0

    def test_downstream_ignition_has_both_regions(self, result):
        keys = set(result["downstream_rate_during_ignition"].keys())
        assert "predictive_layer" in keys
        assert "working_memory" in keys


class TestGlobalWorkspaceBroadcastRates:
    @pytest.fixture
    def result(self, small_workspace_network):
        return GlobalWorkspaceBroadcastBenchmark(small_workspace_network).run(
            inject_steps=20, warmup_steps=5, signal_amplitude=3.0, seed=42
        )

    def test_rates_are_non_negative(self, result):
        assert result["baseline_workspace_rate"] >= 0.0
        assert result["source_workspace_rate"] >= 0.0
        assert result["control_workspace_rate"] >= 0.0

    def test_rates_are_valid_fractions(self, result):
        assert result["baseline_workspace_rate"] <= 1.0
        assert result["source_workspace_rate"] <= 1.0
        assert result["control_workspace_rate"] <= 1.0

    def test_workspace_responds_more_to_connected_region(self, result):
        """Association cortex (wired to workspace) must drive more workspace
        activity than cerebellum (no direct path to workspace)."""
        assert result["source_workspace_rate"] >= result["control_workspace_rate"]

    def test_response_ratio_is_non_negative(self, result):
        assert result["workspace_response_ratio"] >= 0.0

    def test_ignition_counts_are_non_negative_integers(self, result):
        assert isinstance(result["ignition_events_source"], int)
        assert isinstance(result["ignition_events_control"], int)
        assert result["ignition_events_source"] >= 0
        assert result["ignition_events_control"] >= 0

    def test_broadcast_detected_is_bool(self, result):
        assert isinstance(result["broadcast_detected"], bool)

    def test_broadcast_detected_consistent_with_ignition_count(self, result):
        expected = result["ignition_events_source"] > 0
        assert result["broadcast_detected"] == expected

    def test_downstream_rates_are_non_negative(self, result):
        for rate in result["downstream_rate_during_ignition"].values():
            assert rate >= 0.0
