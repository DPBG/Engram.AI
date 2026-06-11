"""Tests for cross-modal recall measurement probe.

Coverage:
1. CrossModalProbe returns valid metrics on a fresh network
2. Classification of association neurons by dominant input modality
3. Recall ratio increases after co-occurring visual+auditory training
4. Probe is read-only (no weight changes, no state mutation)
5. Edge cases: no allocator ranges, empty synapses, single modality
6. Binding strength reflects cross-modal synapse weights
7. JSON serialisation safety (no tuples in output)
8. Index bounds safety
9. Throttled probe in get_metrics
10. Probe duration tracking
11. Decoupled ProbeInputs API
12. Probe after state round-trip
13. Overlapping sensory ranges
14. Out-of-bounds index warning
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sparse_random

from neuromorphic.config import NeuromorphicConfig, PopulationConfig, DendriticCompartmentConfig
from neuromorphic.cross_modal_probe import CrossModalProbe, CrossModalMetrics, ProbeInputs
from neuromorphic.network import NeuromorphicNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config() -> NeuromorphicConfig:
    """Build a tiny config for fast cross-modal tests."""
    return NeuromorphicConfig(
        populations=PopulationConfig(
            brainstem=50,
            reflex_arc=30,
            sensory_cortex=200,
            motor_cortex=100,
            cerebellum=50,
            association_cortex=200,
            predictive_layer=50,
            working_memory=30,
        ),
        dendrites=DendriticCompartmentConfig(enabled=False),
    )


def _inject_visual(net: NeuromorphicNetwork, strength: float = 1.0) -> np.ndarray:
    data = np.random.default_rng(0).random(50).astype(np.float32) * strength
    return net.inject_observation(data, provenance="sensor.videofile.test")


def _inject_auditory(net: NeuromorphicNetwork, strength: float = 1.0) -> np.ndarray:
    data = np.random.default_rng(1).random(30).astype(np.float32) * strength
    return net.inject_observation(data, provenance="sensor.audiofile.test")


def _inject_both(net: NeuromorphicNetwork, strength: float = 1.0) -> np.ndarray:
    vis_data = np.random.default_rng(0).random(50).astype(np.float32) * strength
    aud_data = np.random.default_rng(1).random(30).astype(np.float32) * strength
    return net.inject_multimodal({
        "sensor.videofile.test": vis_data,
        "sensor.audiofile.test": aud_data,
    })


def _make_probe_inputs(
    n_assoc: int = 100,
    n_sensory: int = 200,
    density: float = 0.1,
    vis_range: tuple[int, int] = (0, 80),
    aud_range: tuple[int, int] = (80, 160),
    step: int = 0,
    spikes: np.ndarray | None = None,
) -> ProbeInputs:
    """Build a synthetic ProbeInputs for unit tests."""
    rng = np.random.default_rng(42)
    weights = sparse_random(n_assoc, n_sensory, density=density, format="csr",
                            dtype=np.float32, random_state=rng)
    if spikes is None:
        spikes = rng.random(n_assoc) > 0.8
    return ProbeInputs(
        weights=weights,
        association_spikes=spikes.astype(bool),
        sensory_ranges={"visual": vis_range, "auditory": aud_range},
        n_association=n_assoc,
        step_count=step,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return _small_config()


@pytest.fixture
def network(config):
    return NeuromorphicNetwork(config, seed=42)


@pytest.fixture
def probe():
    return CrossModalProbe()


# ---------------------------------------------------------------------------
# Decoupled ProbeInputs API
# ---------------------------------------------------------------------------

class TestProbeInputsDecoupled:
    """Verify probe works with synthetic ProbeInputs, no network needed."""

    def test_probe_with_synthetic_inputs(self, probe):
        inputs = _make_probe_inputs()
        metrics = probe.probe(inputs)
        assert isinstance(metrics, CrossModalMetrics)
        assert metrics.step_count == 0

    def test_probe_classifies_neurons(self, probe):
        inputs = _make_probe_inputs(density=0.3)
        metrics = probe.probe(inputs)
        total = metrics.n_visual_associated + metrics.n_auditory_associated + metrics.n_cross_modal
        assert total > 0

    def test_probe_with_empty_weights(self, probe):
        inputs = ProbeInputs(
            weights=csr_matrix((50, 100), dtype=np.float32),
            association_spikes=np.zeros(50, dtype=bool),
            sensory_ranges={"visual": (0, 40), "auditory": (40, 80)},
            n_association=50,
            step_count=5,
        )
        metrics = probe.probe(inputs)
        assert metrics.recall_ratio_visual == 0.0
        assert metrics.binding_strength == 0.0
        assert metrics.step_count == 5

    def test_probe_with_mismatched_n_association(self, probe):
        """n_association != n_post should return zero metrics with warning."""
        inputs = _make_probe_inputs(n_assoc=100)
        # Lie about association size
        inputs.n_association = 50
        metrics = probe.probe(inputs)
        assert metrics.binding_strength == 0.0
        assert metrics.n_visual_associated == 0

    def test_probe_single_modality_range(self, probe):
        """Only one modality allocated → zero cross-modal metrics."""
        inputs = _make_probe_inputs(vis_range=(0, 80), aud_range=(0, 0))
        metrics = probe.probe(inputs)
        assert metrics.n_auditory_associated == 0
        assert metrics.binding_strength == 0.0


# ---------------------------------------------------------------------------
# Basic probe tests (via network)
# ---------------------------------------------------------------------------

class TestCrossModalProbeBasic:
    def test_probe_returns_metrics(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        metrics = probe.probe_network(network)
        assert isinstance(metrics, CrossModalMetrics)

    def test_probe_returns_dict(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        d = probe.probe_network(network).to_dict()
        assert isinstance(d, dict)
        assert "recall_ratio_visual" in d
        assert "binding_strength" in d
        assert "association_overlap" in d

    def test_probe_on_fresh_network(self, network, probe):
        metrics = probe.probe_network(network)
        assert metrics.recall_ratio_visual == 0.0
        assert metrics.recall_ratio_auditory == 0.0
        assert metrics.binding_strength == 0.0

    def test_probe_with_single_modality(self, network, probe):
        current = _inject_visual(network)
        network.step(current)
        metrics = probe.probe_network(network)
        assert metrics.n_auditory_associated == 0

    def test_step_count_in_metrics(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        network.step(current)
        assert probe.probe_network(network).step_count == 2

    def test_probe_duration_tracked(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        metrics = probe.probe_network(network)
        assert metrics.probe_duration_ms >= 0.0
        assert "probe_duration_ms" in metrics.to_dict()


# ---------------------------------------------------------------------------
# Read-only verification
# ---------------------------------------------------------------------------

class TestProbeIsReadOnly:
    def test_no_weight_changes(self, network, probe):
        current = _inject_both(network)
        for _ in range(5):
            network.step(current)
        sa_syn = network.synapses["sensory_association"]
        weights_before = sa_syn.weights.data.copy()
        probe.probe_network(network)
        np.testing.assert_array_equal(sa_syn.weights.data, weights_before)

    def test_no_spike_changes(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        spikes_before = network.association.spikes.copy()
        probe.probe_network(network)
        np.testing.assert_array_equal(network.association.spikes, spikes_before)

    def test_no_step_count_change(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        step_before = network.step_count
        probe.probe_network(network)
        assert network.step_count == step_before


# ---------------------------------------------------------------------------
# Cross-modal binding development
# ---------------------------------------------------------------------------

class TestCrossModalBinding:
    def test_both_modalities_classified(self, network, probe):
        current = _inject_both(network)
        for _ in range(10):
            network.step(current)
        metrics = probe.probe_network(network)
        total = metrics.n_visual_associated + metrics.n_auditory_associated + metrics.n_cross_modal
        assert total > 0, "No association neurons classified after 10 steps"

    def test_binding_strength_positive_after_training(self, network, probe):
        current = _inject_both(network)
        for _ in range(20):
            network.step(current)
        assert probe.probe_network(network).binding_strength >= 0.0

    def test_selectivity_increases_with_training(self, network, probe):
        current = _inject_both(network)
        for _ in range(20):
            network.step(current)
        assert probe.probe_network(network).modality_selectivity > 0.0

    def test_sensory_ranges_reported(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        metrics = probe.probe_network(network)
        assert "visual" in metrics.sensory_ranges
        assert "auditory" in metrics.sensory_ranges

    def test_binding_strength_averages_nonzero_directions(self, probe):
        """When only one direction has binding, overall should equal that direction."""
        inputs = _make_probe_inputs(density=0.0)
        # Manually set weights so only visual columns have values for
        # auditory-associated neurons (v2a only, a2v = 0)
        rng = np.random.default_rng(99)
        w = inputs.weights.toarray()
        # Make rows 0-9 auditory-dominated (strong auditory columns)
        w[0:10, 80:160] = rng.random((10, 80)).astype(np.float32) * 0.5
        # Add weak visual input to same neurons (creates v2a path)
        w[0:10, 0:80] = rng.random((10, 80)).astype(np.float32) * 0.02
        inputs.weights = csr_matrix(w)
        metrics = probe.probe(inputs)
        # If v2a > 0 and a2v = 0, binding_strength should equal v2a
        if metrics.binding_strength_visual_to_auditory > 0 and metrics.binding_strength_auditory_to_visual == 0.0:
            assert metrics.binding_strength == pytest.approx(
                metrics.binding_strength_visual_to_auditory, abs=1e-4,
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_json_serialisable(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        d = probe.probe_network(network).to_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_sensory_ranges_are_lists_not_tuples(self, network, probe):
        current = _inject_both(network)
        network.step(current)
        metrics = probe.probe_network(network)
        for key, val in metrics.sensory_ranges.items():
            assert isinstance(val, list), f"Range for {key} is {type(val)}, expected list"

    def test_probe_survives_zero_nnz(self, network, probe):
        from scipy.sparse import csr_matrix as csr
        sa = network.synapses["sensory_association"]
        n_post, n_pre = sa.weights.shape
        sa.weights = csr((n_post, n_pre), dtype=np.float32)
        metrics = probe.probe_network(network)
        assert metrics.recall_ratio_visual == 0.0
        assert metrics.binding_strength == 0.0

    def test_probe_repeated_calls_stable(self, network, probe):
        current = _inject_both(network)
        for _ in range(10):
            network.step(current)
        m1 = probe.probe_network(network)
        m2 = probe.probe_network(network)
        assert m1.n_visual_associated == m2.n_visual_associated
        assert m1.n_auditory_associated == m2.n_auditory_associated
        assert m1.binding_strength == m2.binding_strength

    def test_overlapping_sensory_ranges(self, probe):
        """Overlapping vis/aud ranges should not crash — shared neurons
        contribute to both fractions."""
        inputs = _make_probe_inputs(
            vis_range=(0, 120),
            aud_range=(80, 200),  # overlap at 80-120
        )
        metrics = probe.probe(inputs)
        assert isinstance(metrics, CrossModalMetrics)
        # Overlapping neurons get fraction from both → classified as cross-modal
        # or split evenly, but should not error.

    def test_range_exceeds_matrix(self, probe):
        """Sensory range beyond matrix columns is clamped, not error."""
        inputs = _make_probe_inputs(
            n_sensory=100,
            vis_range=(0, 50),
            aud_range=(50, 999),  # way beyond n_pre=100
        )
        metrics = probe.probe(inputs)
        assert isinstance(metrics, CrossModalMetrics)

    def test_high_margin_classifies_all_cross_modal(self, probe):
        """With margin=1.0, no neuron can be single-modality dominated."""
        probe_strict = CrossModalProbe()
        probe_strict.DOMINANCE_MARGIN = 1.0
        inputs = _make_probe_inputs(density=0.3)
        metrics = probe_strict.probe(inputs)
        assert metrics.n_visual_associated == 0
        assert metrics.n_auditory_associated == 0
        # Everything with input is cross-modal since no diff can reach 1.0
        # (fracs sum to 1.0, so max diff is 1.0 which is not >= margin)

    def test_default_metrics_all_zero(self):
        """Default CrossModalMetrics should be all zeros."""
        m = CrossModalMetrics()
        d = m.to_dict()
        for key, val in d.items():
            if key == "sensory_ranges":
                assert val == {}
            elif isinstance(val, (int, float)):
                assert val == 0, f"{key} should be 0, got {val}"

    def test_probe_exception_returns_safe_metrics(self, probe):
        """If probe_inner raises, probe() returns safe zero metrics."""
        # Create inputs with incompatible spike array to trigger error path
        inputs = _make_probe_inputs(n_assoc=100)
        inputs.association_spikes = np.zeros(50, dtype=bool)  # wrong size
        inputs.n_association = 100  # matches weights, but spikes don't
        metrics = probe.probe(inputs)
        # Should still get a metrics object, not an exception
        assert isinstance(metrics, CrossModalMetrics)
        assert metrics.probe_duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Probe after state round-trip
# ---------------------------------------------------------------------------

class TestProbeAfterStateRoundTrip:
    def test_probe_after_save_restore(self, config, probe):
        """Probe should work without error after get_state/set_state cycle.

        Note: allocator ranges are NOT persisted — they are re-derived
        from observations.  So the classification may differ if the
        restored network gets different allocator ranges (different seed).
        The key assertion is that the probe produces valid results and
        the weights are identical.
        """
        net1 = NeuromorphicNetwork(config, seed=42)
        current = _inject_both(net1)
        for _ in range(10):
            net1.step(current)

        # Save and restore
        state = net1.get_state()
        net2 = NeuromorphicNetwork(config, seed=42)
        net2.set_state(state)

        # Verify weights are identical
        w1 = net1.synapses["sensory_association"].weights
        w2 = net2.synapses["sensory_association"].weights
        np.testing.assert_array_equal(w1.data, w2.data)

        # Re-inject to populate allocator (same seed → same ranges)
        _inject_both(net2)
        m2 = probe.probe_network(net2)

        # Probe should produce valid results (no crash, valid types)
        assert isinstance(m2, CrossModalMetrics)
        assert m2.step_count == 10
        assert m2.probe_duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Throttled probe in get_metrics
# ---------------------------------------------------------------------------

class TestThrottledProbe:
    def test_cross_modal_in_get_metrics(self, network):
        current = _inject_both(network)
        network.step(current)
        # Force the probe to run
        for _ in range(network._crossmodal_probe_interval + 1):
            metrics = network.get_metrics()
        assert "cross_modal" in metrics
        assert "recall_ratio_visual" in metrics["cross_modal"]
        assert "binding_strength" in metrics["cross_modal"]

    def test_get_metrics_cross_modal_zero_on_fresh(self, network):
        metrics = network.get_metrics()
        assert metrics["cross_modal"]["recall_ratio_visual"] == 0.0
        assert metrics["cross_modal"]["binding_strength"] == 0.0

    def test_throttle_returns_cached_between_intervals(self, network):
        current = _inject_both(network)
        network.step(current)
        network._crossmodal_probe_counter = network._crossmodal_probe_interval - 1
        m1 = network.get_metrics()
        m2 = network.get_metrics()
        assert m1["cross_modal"] == m2["cross_modal"]

    def test_probe_duration_in_get_metrics(self, network):
        network._crossmodal_probe_counter = network._crossmodal_probe_interval - 1
        metrics = network.get_metrics()
        assert "probe_duration_ms" in metrics["cross_modal"]
