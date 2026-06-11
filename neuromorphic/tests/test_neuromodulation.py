"""Tests for neuromodulation system and critical periods (Phases 6-7)."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig, CriticalPeriodConfig
from neuromorphic.neuromodulation import NeuromodulationSystem


class TestNeuromodulationBasics:
    """Test basic neuromodulation functionality."""

    @pytest.fixture
    def nm(self):
        return NeuromodulationSystem(NeuromorphicConfig())

    def test_initial_state(self, nm):
        """Initial state should match infant critical period."""
        cp = CriticalPeriodConfig()
        nm.update(0)
        assert nm.da == pytest.approx(cp.infant_da)
        assert nm.ach == pytest.approx(cp.infant_ach)
        assert nm.ne == pytest.approx(cp.infant_ne)
        assert nm.serotonin == pytest.approx(cp.infant_serotonin)

    def test_plasticity_multiplier(self, nm):
        """Plasticity multiplier = DA * (1 - 5-HT)."""
        nm.da = 2.0
        nm.serotonin = 0.3
        assert nm.plasticity_multiplier == pytest.approx(2.0 * 0.7)

    def test_high_serotonin_suppresses(self, nm):
        """High serotonin should suppress plasticity."""
        nm.da = 1.0
        nm.serotonin = 0.9
        assert nm.plasticity_multiplier == pytest.approx(0.1)

    def test_meta_controller_modulation(self, nm):
        """Meta-controller outputs should scale baselines."""
        nm.update(0, {"da": 2.0, "ach": 0.5, "ne": 1.5, "serotonin": 1.0})
        cp = CriticalPeriodConfig()
        assert nm.da == pytest.approx(cp.infant_da * 2.0)
        assert nm.ach == pytest.approx(cp.infant_ach * 0.5)

    def test_no_meta_uses_baselines(self, nm):
        """Without meta-controller, should use raw baselines."""
        nm.update(0, None)
        cp = CriticalPeriodConfig()
        assert nm.da == pytest.approx(cp.infant_da)


class TestCriticalPeriods:
    """Test developmental schedule."""

    @pytest.fixture
    def nm(self):
        return NeuromodulationSystem(NeuromorphicConfig())

    def test_infant_phase(self, nm):
        nm.update(0)
        assert nm.phase == "infant"

    def test_toddler_phase(self, nm):
        cp = CriticalPeriodConfig()
        nm.update(cp.infant_end + 1)
        assert nm.phase == "toddler"

    def test_juvenile_phase(self, nm):
        cp = CriticalPeriodConfig()
        nm.update(cp.toddler_end + 1)
        assert nm.phase == "juvenile"

    def test_mature_phase(self, nm):
        cp = CriticalPeriodConfig()
        nm.update(cp.juvenile_end + 1)
        assert nm.phase == "mature"

    def test_infant_high_plasticity(self, nm):
        """Infant phase should have high DA and ACh for wide-open learning."""
        nm.update(0)
        assert nm.da >= 1.5
        assert nm.ach >= 2.0
        assert nm.serotonin <= 0.2

    def test_mature_stable(self, nm):
        """Mature phase should have baseline-1.0 neuromodulators."""
        cp = CriticalPeriodConfig()
        nm.update(cp.juvenile_end + 1000)
        assert nm.da == pytest.approx(1.0)
        assert nm.ach == pytest.approx(1.0)
        assert nm.ne == pytest.approx(1.0)
        assert nm.serotonin == pytest.approx(0.5)

    def test_smooth_transition(self, nm):
        """Transitions between phases should be smooth (interpolated)."""
        cp = CriticalPeriodConfig()
        # At midpoint of infant→toddler transition
        mid = cp.infant_end + (cp.toddler_end - cp.infant_end) // 2
        nm.update(mid)
        # DA should be between infant and toddler values
        assert cp.toddler_da < nm.da < cp.infant_da

    def test_plasticity_decreases_with_development(self, nm):
        """Plasticity multiplier should decrease from infant to mature."""
        cp = CriticalPeriodConfig()
        nm.update(0)
        infant_plasticity = nm.plasticity_multiplier

        nm.update(cp.juvenile_end + 1000)
        mature_plasticity = nm.plasticity_multiplier

        assert infant_plasticity > mature_plasticity


class TestNeuromodulationState:
    """Test state save/restore."""

    def test_state_roundtrip(self):
        nm = NeuromodulationSystem(NeuromorphicConfig())
        nm.update(1_000_000)
        state = nm.get_state()

        nm2 = NeuromodulationSystem(NeuromorphicConfig())
        nm2.set_state(state)

        assert nm2.da == pytest.approx(nm.da)
        assert nm2.ach == pytest.approx(nm.ach)
        assert nm2.ne == pytest.approx(nm.ne)
        assert nm2.serotonin == pytest.approx(nm.serotonin)
        assert nm2.phase == nm.phase
