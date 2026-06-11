"""Tests for homeostatic drive system."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.drives import HomeostaticDriveSystem
from neuromorphic.regions import Brainstem


@pytest.fixture
def config():
    cfg = NeuromorphicConfig()
    cfg.populations.brainstem = 100
    return cfg


@pytest.fixture
def drives(config):
    return HomeostaticDriveSystem(config)


class TestDriveState:
    def test_initial_state(self, drives):
        assert drives.energy == 1.0
        assert drives.damage == 0.0
        assert drives.temperature == 0.5
        assert drives.fatigue == 0.0

    def test_energy_decay(self, drives):
        drives.update(dt_seconds=100.0)
        assert drives.energy < 1.0

    def test_damage_recovery(self, drives):
        drives.damage = 0.5
        drives.update(dt_seconds=100.0)
        assert drives.damage < 0.5

    def test_fatigue_accumulation(self, drives):
        drives.update(dt_seconds=100.0)
        assert drives.fatigue > 0.0

    def test_temperature_drift(self, drives):
        # Temperature should change (random walk)
        # Use larger dt to amplify drift (drift std = 0.0002 * dt_seconds)
        initial = drives.temperature
        for _ in range(200):
            drives.update(dt_seconds=10.0)
        # After 200 steps × dt=10, cumulative drift should be measurable
        assert drives.temperature != initial, "Temperature should drift via random walk"


class TestDriveEvents:
    def test_feed_event(self, drives):
        drives.energy = 0.3
        drives.handle_event({"type": "feed", "amount": 0.5})
        assert drives.energy == pytest.approx(0.8)

    def test_feed_caps_at_1(self, drives):
        drives.handle_event({"type": "feed", "amount": 0.5})
        assert drives.energy == 1.0

    def test_damage_event(self, drives):
        drives.handle_event({"type": "damage", "amount": 0.3})
        assert drives.damage == pytest.approx(0.3)

    def test_damage_caps_at_1(self, drives):
        drives.handle_event({"type": "damage", "amount": 1.5})
        assert drives.damage == 1.0

    def test_rest_event(self, drives):
        drives.fatigue = 0.8
        drives.handle_event({"type": "rest", "amount": 0.3})
        assert drives.fatigue == pytest.approx(0.5)

    def test_temperature_event(self, drives):
        drives.handle_event({"type": "temperature", "value": 0.8})
        assert drives.temperature == 0.8


class TestDriveCritical:
    def test_not_critical_initial(self, drives):
        assert not drives.is_critical()

    def test_critical_low_energy(self, drives):
        drives.energy = 0.1
        assert drives.is_critical()

    def test_critical_high_damage(self, drives):
        drives.damage = 0.9
        assert drives.is_critical()

    def test_critical_high_fatigue(self, drives):
        drives.fatigue = 0.9
        assert drives.is_critical()


class TestBrainstemCurrent:
    def test_brainstem_current_shape(self, config, drives):
        brainstem = Brainstem(config)
        current = drives.compute_brainstem_current(brainstem)
        assert current.shape == (brainstem.n,)
        assert current.dtype == np.float32

    def test_low_energy_high_current(self, config, drives):
        brainstem = Brainstem(config)
        drives.energy = 0.1  # very low energy
        current = drives.compute_brainstem_current(brainstem)
        sr = brainstem.get_subrange("energy")
        assert current[sr.slice()].mean() > 0

    def test_no_urgency_tonic_baseline(self, config, drives):
        brainstem = Brainstem(config)
        # All drives at optimal
        drives.energy = 1.0
        drives.damage = 0.0
        drives.temperature = 0.5
        drives.fatigue = 0.0
        current = drives.compute_brainstem_current(brainstem)
        # Energy sub-range should have only tonic current (no urgency added)
        sr = brainstem.get_subrange("energy")
        assert abs(current[sr.slice()].mean() - config.drives.tonic_current) < 0.1


class TestDriveStateSerialization:
    def test_get_set_state(self, drives):
        drives.energy = 0.5
        drives.damage = 0.3
        state = drives.get_state()
        assert state["energy"] == 0.5
        assert state["damage"] == 0.3

        drives2 = HomeostaticDriveSystem(NeuromorphicConfig())
        drives2.set_state(state)
        assert drives2.energy == 0.5
        assert drives2.damage == 0.3

    def test_summary(self, drives):
        s = drives.summary()
        assert "energy" in s
        assert "damage" in s
