"""Tests for all 8 brain regions."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.regions import (
    Brainstem,
    ReflexArc,
    SensoryCortex,
    MotorCortex,
    Cerebellum,
    AssociationCortex,
    PredictiveLayer,
    WorkingMemory,
    create_all_regions,
)


@pytest.fixture
def config():
    """Small config for fast tests."""
    cfg = NeuromorphicConfig()
    cfg.populations.brainstem = 100
    cfg.populations.reflex_arc = 80
    cfg.populations.sensory_cortex = 400
    cfg.populations.motor_cortex = 200
    cfg.populations.cerebellum = 200
    cfg.populations.association_cortex = 400
    cfg.populations.predictive_layer = 200
    cfg.populations.working_memory = 50
    return cfg


class TestBrainstem:
    def test_population_size(self, config):
        region = Brainstem(config)
        assert region.n == 100

    def test_sub_ranges(self, config):
        region = Brainstem(config)
        names = [sr.name for sr in region.sub_ranges]
        assert "energy" in names
        assert "damage" in names
        assert "temperature" in names
        assert "fatigue" in names

    def test_slow_dynamics(self, config):
        """Brainstem should have moderately slow tau (pacemaker-excitable)."""
        assert config.brainstem_lif.tau == 40.0


class TestReflexArc:
    def test_population_size(self, config):
        region = ReflexArc(config)
        assert region.n == 80

    def test_sub_ranges(self, config):
        region = ReflexArc(config)
        names = [sr.name for sr in region.sub_ranges]
        assert "pain_withdrawal" in names
        assert "grip" in names
        assert "startle" in names
        assert "flinch" in names

    def test_fast_dynamics(self, config):
        """Reflex arc should have low tau (fast)."""
        assert config.reflex_lif.tau == 10.0


class TestSensoryCortex:
    def test_population_size(self, config):
        region = SensoryCortex(config)
        assert region.n == 400

    def test_sub_ranges_cover_full(self, config):
        region = SensoryCortex(config)
        total_covered = sum(sr.size for sr in region.sub_ranges)
        assert total_covered == region.n

    def test_modality_sub_ranges(self, config):
        region = SensoryCortex(config)
        names = [sr.name for sr in region.sub_ranges]
        assert "visual" in names
        assert "auditory" in names
        assert "tactile" in names
        assert "proprioceptive" in names


class TestMotorCortex:
    def test_population_size(self, config):
        region = MotorCortex(config)
        assert region.n == 200

    def test_sub_ranges(self, config):
        region = MotorCortex(config)
        names = [sr.name for sr in region.sub_ranges]
        assert "locomotion" in names
        assert "manipulation" in names
        assert "head" in names
        assert "expression" in names


class TestCerebellum:
    def test_population_size(self, config):
        region = Cerebellum(config)
        assert region.n == 200


class TestAssociationCortex:
    def test_population_size(self, config):
        region = AssociationCortex(config)
        assert region.n == 400


class TestPredictiveLayer:
    def test_population_size(self, config):
        region = PredictiveLayer(config)
        assert region.n == 200

    def test_prev_spikes_tracking(self, config):
        region = PredictiveLayer(config)
        current = np.zeros(200, dtype=np.float32)
        region.step(current)
        # After first step, prev_spikes should exist
        assert region.prev_spikes.shape == (200,)


class TestWorkingMemory:
    def test_population_size(self, config):
        region = WorkingMemory(config)
        assert region.n == 50


class TestCreateAllRegions:
    def test_creates_8_regions(self, config):
        regions = create_all_regions(config)
        assert len(regions) == 8

    def test_region_names(self, config):
        regions = create_all_regions(config)
        expected = {
            "brainstem", "reflex_arc", "sensory_cortex", "motor_cortex",
            "cerebellum", "association_cortex", "predictive_layer", "working_memory",
        }
        assert set(regions.keys()) == expected

    def test_total_neuron_count(self, config):
        regions = create_all_regions(config)
        total = sum(r.n for r in regions.values())
        assert total == config.populations.total


class TestBrainRegionBase:
    def test_inject_current(self, config):
        region = SensoryCortex(config)
        current = np.ones(region.n, dtype=np.float32)
        region.inject_current(current)
        # Step should use the injected current then clear it
        region.step(np.zeros(region.n, dtype=np.float32))
        # External current should be cleared
        np.testing.assert_allclose(region._external_current, 0.0)

    def test_inject_current_subrange(self, config):
        region = SensoryCortex(config)
        sr = region.get_subrange("visual")
        current = np.ones(sr.size, dtype=np.float32)
        region.inject_current_subrange("visual", current)
        assert region._external_current[sr.slice()].sum() > 0

    def test_state_roundtrip(self, config):
        region = SensoryCortex(config)
        region.step(np.random.randn(region.n).astype(np.float32) * 10)
        state = region.get_state()
        region2 = SensoryCortex(config)
        region2.set_state(state)
        np.testing.assert_allclose(
            region.population.v_membrane,
            region2.population.v_membrane,
        )
