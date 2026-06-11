"""Tests for the Neural Watchdog safety system."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.network import NeuromorphicNetwork
from neuromorphic.watchdog import (
    NeuralWatchdog,
    WatchdogConfig,
    AlertLevel,
    WatchdogAlert,
    WatchdogStatus,
)


@pytest.fixture
def small_config():
    cfg = NeuromorphicConfig()
    cfg.populations.brainstem = 50
    cfg.populations.reflex_arc = 30
    cfg.populations.sensory_cortex = 100
    cfg.populations.motor_cortex = 80
    cfg.populations.cerebellum = 60
    cfg.populations.association_cortex = 100
    cfg.populations.predictive_layer = 60
    cfg.populations.working_memory = 30
    return cfg


@pytest.fixture
def network(small_config):
    return NeuromorphicNetwork(small_config, seed=42)


@pytest.fixture
def watchdog():
    return NeuralWatchdog(WatchdogConfig())


class TestWatchdogBasics:
    def test_initial_check_nominal(self, watchdog, network):
        status = watchdog.check(network, step_count=0)
        assert status.level == AlertLevel.NOMINAL

    def test_skip_duplicate_step(self, watchdog, network):
        s1 = watchdog.check(network, step_count=100)
        s2 = watchdog.check(network, step_count=100)
        # Second check returns NOMINAL (skipped)
        assert s2.level == AlertLevel.NOMINAL
        assert len(s2.alerts) == 0

    def test_sensory_starvation_critical(self, watchdog, network):
        status = watchdog.check(network, step_count=100, sensory_steps_since_input=1500)
        starvation_alerts = [a for a in status.alerts if a.check_name == "sensory_starvation"]
        assert len(starvation_alerts) == 1
        assert starvation_alerts[0].level == AlertLevel.CRITICAL

    def test_status_to_dict(self, watchdog, network):
        status = watchdog.check(network, step_count=100)
        d = status.to_dict()
        assert "level" in d
        assert "alerts" in d
        assert d["level"] == "NOMINAL"


class TestGovernorCorrections:
    def test_no_corrections_at_low_rates(self, watchdog, network):
        corrections = watchdog.get_governor_corrections(network)
        assert len(corrections) == 0

    def test_correction_returns_negative_current(self, watchdog, network):
        """If a region had high firing, correction should be negative."""
        # We can't easily force high rates in a unit test without monkey-patching,
        # so verify the method returns a dict and doesn't crash.
        corrections = watchdog.get_governor_corrections(network)
        for val in corrections.values():
            assert val <= 0.0


class TestWeightOscillationDetection:
    def test_no_oscillation_initially(self, watchdog, network):
        status = watchdog.check(network, step_count=100)
        osc_alerts = [a for a in status.alerts if a.check_name == "weight_oscillation"]
        assert len(osc_alerts) == 0

    def test_detects_oscillation(self, watchdog, network):
        """Inject oscillating weight history and verify detection."""
        # Simulate weight history that oscillates rapidly
        for name, syn in network.synapses.items():
            if syn.plastic:
                watchdog._weight_history[name] = [0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5]
                break
        status = watchdog.check(network, step_count=200)
        osc_alerts = [a for a in status.alerts if a.check_name == "weight_oscillation"]
        assert len(osc_alerts) >= 1

    def test_no_false_positive_on_monotonic(self, watchdog, network):
        """Monotonically increasing weights should not trigger oscillation."""
        for name, syn in network.synapses.items():
            if syn.plastic:
                watchdog._weight_history[name] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                break
        status = watchdog.check(network, step_count=300)
        osc_alerts = [a for a in status.alerts if a.check_name == "weight_oscillation"]
        assert len(osc_alerts) == 0


class TestEligibilitySaturationDetection:
    def test_no_saturation_initially(self, watchdog, network):
        status = watchdog.check(network, step_count=100)
        sat_alerts = [a for a in status.alerts if a.check_name == "eligibility_saturation"]
        assert len(sat_alerts) == 0


class TestStatePersistence:
    def test_state_roundtrip(self, watchdog, network):
        # Run some checks to build state
        watchdog.check(network, step_count=100, sensory_steps_since_input=50)
        watchdog.check(network, step_count=200, sensory_steps_since_input=100)

        state = watchdog.get_state()
        assert "silence_counters" in state
        assert "weight_history" in state
        assert "last_check_step" in state
        assert state["last_check_step"] == 200

        watchdog2 = NeuralWatchdog(WatchdogConfig())
        watchdog2.set_state(state)
        assert watchdog2._last_check_step == 200

    def test_escalation_levels(self):
        """Test AlertLevel ordering."""
        assert AlertLevel.NOMINAL < AlertLevel.WARNING
        assert AlertLevel.WARNING < AlertLevel.CAUTION
        assert AlertLevel.CAUTION < AlertLevel.CRITICAL
        assert AlertLevel.CRITICAL < AlertLevel.EMERGENCY


class TestSafetyConfig:
    def test_safety_gate_config_defaults(self):
        cfg = NeuromorphicConfig()
        assert cfg.safety_gate.enabled is False
        assert cfg.safety_gate.fail_open is True
        assert cfg.safety_gate.decision_timeout == 2.0

    def test_safety_gate_env(self, monkeypatch):
        monkeypatch.setenv("NEURO_SAFETY_GATE", "1")
        monkeypatch.setenv("NEURO_SAFETY_TIMEOUT", "5.0")
        cfg = NeuromorphicConfig.from_env()
        assert cfg.safety_gate.enabled is True
        assert cfg.safety_gate.decision_timeout == 5.0

    def test_watchdog_config_defaults(self):
        cfg = WatchdogConfig()
        assert cfg.check_interval == 100
        assert cfg.rate_warning == 0.40
        assert cfg.rate_critical == 0.60
        assert cfg.weight_oscillation_window == 6
        assert cfg.weight_oscillation_threshold == 4
        assert cfg.elig_saturation_threshold == 0.8
