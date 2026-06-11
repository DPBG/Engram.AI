"""Tests for hierarchical brain architecture — concept layer, feature layer, meta-controller (Phases 3-4, 6)."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig, PopulationConfig, ConceptLayerConfig, FeatureLayerConfig, MetaControllerConfig
from neuromorphic.regions import (
    FeatureLayer, ConceptLayer, MetaControllerRegion, create_all_regions,
)
from neuromorphic.network import NeuromorphicNetwork


def _hierarchical_config():
    """Config with all new regions enabled."""
    return NeuromorphicConfig(
        populations=PopulationConfig(
            brainstem=20, reflex_arc=20, sensory_cortex=100,
            motor_cortex=50, cerebellum=50, association_cortex=100,
            predictive_layer=50, working_memory=20,
            feature_layer=80, concept_layer=100, meta_controller=60,
        ),
        concept_layer=ConceptLayerConfig(n_neurons=100, k_winners=5),
    )


class TestFeatureLayer:
    """Test V4/IT-like feature integration layer."""

    def test_creation(self):
        config = _hierarchical_config()
        region = FeatureLayer(config)
        assert region.n == 80
        assert region.name == "feature_layer"

    def test_step(self):
        config = _hierarchical_config()
        region = FeatureLayer(config, rng=np.random.default_rng(42))
        current = np.full(80, 30.0, dtype=np.float32)
        spikes = region.step(current)
        assert spikes.shape == (80,)
        assert spikes.dtype == bool

    def test_has_inhibitory(self):
        config = _hierarchical_config()
        region = FeatureLayer(config)
        n_inh = region.population.is_inhibitory.sum()
        assert n_inh == int(80 * 0.2)  # 20% inhibitory


class TestConceptLayer:
    """Test k-WTA concept bottleneck layer."""

    def test_creation(self):
        config = _hierarchical_config()
        region = ConceptLayer(config)
        assert region.n == 100
        assert region.name == "concept_layer"
        assert region._k == 5

    def test_kwta_enforces_sparsity(self):
        """k-WTA should limit firing to at most k neurons."""
        config = _hierarchical_config()
        region = ConceptLayer(config, rng=np.random.default_rng(42))
        # Drive many neurons above threshold
        current = np.full(100, 50.0, dtype=np.float32)
        spikes = region.step(current)
        assert spikes.sum() <= 5  # k=5

    def test_kwta_allows_fewer(self):
        """k-WTA should allow fewer than k if not enough neurons fire."""
        config = _hierarchical_config()
        region = ConceptLayer(config, rng=np.random.default_rng(42))
        # Only drive a small subset
        current = np.zeros(100, dtype=np.float32)
        current[:3] = 50.0
        spikes = region.step(current)
        assert spikes.sum() <= 5

    def test_concept_lif_params(self):
        """Concept layer should have custom LIF params (slower tau, harder threshold)."""
        config = _hierarchical_config()
        region = ConceptLayer(config)
        # Check that the LIF params match the ConceptLayerConfig
        cl = config.concept_layer
        assert region.population.params.tau == cl.tau
        assert region.population.params.threshold == cl.threshold


class TestMetaControllerRegion:
    """Test neuromodulatory meta-controller region."""

    def test_creation(self):
        config = _hierarchical_config()
        region = MetaControllerRegion(config)
        assert region.n == 60
        assert region.name == "meta_controller"

    def test_sub_ranges(self):
        config = _hierarchical_config()
        region = MetaControllerRegion(config)
        names = [sr.name for sr in region.sub_ranges]
        assert "monitor" in names
        assert "da" in names
        assert "ach" in names
        assert "ne" in names
        assert "serotonin" in names
        assert "integrator" in names

    def test_read_neuromodulators(self):
        config = _hierarchical_config()
        region = MetaControllerRegion(config, rng=np.random.default_rng(42))
        # Step with some input
        current = np.full(60, 30.0, dtype=np.float32)
        region.step(current)
        nm = region.read_neuromodulators()
        assert "da" in nm
        assert "ach" in nm
        assert "ne" in nm
        assert "serotonin" in nm
        # All should be within configured ranges
        mc = config.meta_controller
        assert mc.da_range[0] <= nm["da"] <= mc.da_range[1]
        assert mc.ach_range[0] <= nm["ach"] <= mc.ach_range[1]
        assert mc.ne_range[0] <= nm["ne"] <= mc.ne_range[1]
        assert mc.serotonin_range[0] <= nm["serotonin"] <= mc.serotonin_range[1]


class TestCreateAllRegionsHierarchical:
    """Test that create_all_regions creates the right number of regions."""

    def test_base_config_8_regions(self):
        config = NeuromorphicConfig(
            populations=PopulationConfig(
                brainstem=20, reflex_arc=20, sensory_cortex=100,
                motor_cortex=50, cerebellum=50, association_cortex=100,
                predictive_layer=50, working_memory=20,
            )
        )
        regions = create_all_regions(config)
        assert len(regions) == 8

    def test_hierarchical_config_11_regions(self):
        config = _hierarchical_config()
        regions = create_all_regions(config)
        assert len(regions) == 11
        assert "feature_layer" in regions
        assert "concept_layer" in regions
        assert "meta_controller" in regions

    def test_total_neuron_count(self):
        config = _hierarchical_config()
        regions = create_all_regions(config)
        total = sum(r.n for r in regions.values())
        assert total == config.populations.total


class TestHierarchicalNetwork:
    """Test full network with hierarchical regions enabled."""

    @pytest.fixture
    def network(self):
        config = _hierarchical_config()
        return NeuromorphicNetwork(config, seed=42)

    def test_creates_feature_synapse_groups(self, network):
        assert "sensory_feature" in network.synapses
        assert "feature_association" in network.synapses

    def test_creates_concept_synapse_groups(self, network):
        assert "association_concept" in network.synapses
        assert "concept_lateral" in network.synapses
        assert "concept_predictive" in network.synapses
        assert "concept_working" in network.synapses
        assert "predictive_concept" in network.synapses

    def test_creates_meta_synapse_groups(self, network):
        assert "association_meta" in network.synapses
        assert "meta_association" in network.synapses

    def test_step_with_hierarchy(self, network):
        """Full step should work with all hierarchical regions."""
        result = network.step()
        assert "motor_commands" in result
        assert "step_count" in result
        assert result["step_count"] == 1

    def test_multiple_steps(self, network):
        for _ in range(10):
            result = network.step()
        assert result["step_count"] == 10

    def test_inject_observation(self, network):
        """inject_observation should work with hierarchical network."""
        current = network.inject_observation([0.5, 0.3, 0.7], "sensor.camera.0")
        assert current.shape == (100,)  # sensory_cortex size

    def test_state_roundtrip(self, network):
        """State save/restore should include all new regions."""
        network.step()
        state = network.get_state()
        assert "feature_layer" in state["regions"]
        assert "concept_layer" in state["regions"]
        assert "meta_controller" in state["regions"]
        assert "sensory_feature" in state["synapses"]
        assert "neuromodulation" in state
