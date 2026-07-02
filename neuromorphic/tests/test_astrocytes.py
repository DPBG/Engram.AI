"""Tests for astrocyte-gated plasticity (CIP-22)."""

import numpy as np

from neuromorphic.astrocytes import AstrocyteNetwork
from neuromorphic.config import AstrocyteConfig


class TestAstrocyteNetwork:
    """Core astrocyte dynamics tests."""

    def _make_astrocytes(self, **overrides):
        cfg = AstrocyteConfig(enabled=True, **overrides)
        regions = ["sensory_cortex", "association_cortex", "motor_cortex"]
        return AstrocyteNetwork(cfg, regions)

    def test_init_zero_calcium(self):
        """Calcium and gliotransmitter should start at zero."""
        astro = self._make_astrocytes()
        assert astro._calcium.sum() == 0.0
        assert astro._gliotransmitter.sum() == 0.0

    def test_calcium_rises_with_activity(self):
        """High firing rates should increase calcium."""
        astro = self._make_astrocytes(tau_calcium=100.0)
        rates = {"sensory_cortex": 0.5, "association_cortex": 0.0, "motor_cortex": 0.0}
        for _ in range(100):
            astro.step(rates)
        # Sensory should have high calcium, others low
        assert astro._calcium[0] > 0.1
        assert astro._calcium[1] < 0.01
        assert astro._calcium[2] < 0.01

    def test_calcium_decays_without_activity(self):
        """Calcium should decay when firing rate drops to zero."""
        astro = self._make_astrocytes(tau_calcium=100.0)
        # Build up calcium
        high_rates = {"sensory_cortex": 0.8, "association_cortex": 0.0, "motor_cortex": 0.0}
        for _ in range(200):
            astro.step(high_rates)
        peak = float(astro._calcium[0])
        assert peak > 0.1
        # Let it decay
        zero_rates = {"sensory_cortex": 0.0, "association_cortex": 0.0, "motor_cortex": 0.0}
        for _ in range(500):
            astro.step(zero_rates)
        assert astro._calcium[0] < peak * 0.5, "Calcium should decay significantly"

    def test_plasticity_gate_1_when_inactive(self):
        """Plasticity gate should be ~1.0 when region is inactive."""
        astro = self._make_astrocytes()
        gate = astro.get_plasticity_gate("sensory_cortex")
        assert gate > 0.95, f"Gate should be ~1.0 when inactive, got {gate}"

    def test_plasticity_gate_drops_with_high_activity(self):
        """Plasticity gate should drop when region is hyperactive."""
        astro = self._make_astrocytes(metabolic_threshold=0.2, tau_calcium=100.0)
        rates = {"sensory_cortex": 0.8, "association_cortex": 0.0, "motor_cortex": 0.0}
        for _ in range(500):
            astro.step(rates)
        gate = astro.get_plasticity_gate("sensory_cortex")
        assert gate < 0.5, f"Gate should drop with high activity, got {gate}"
        # Other regions should be unaffected
        gate_assoc = astro.get_plasticity_gate("association_cortex")
        assert gate_assoc > 0.85, f"Inactive region should have high gate, got {gate_assoc}"

    def test_plasticity_gate_never_below_minimum(self):
        """Plasticity gate should never go below plasticity_gate_min."""
        astro = self._make_astrocytes(
            plasticity_gate_min=0.1,
            metabolic_threshold=0.1,
        )
        # Extreme activity
        rates = {"sensory_cortex": 1.0, "association_cortex": 1.0, "motor_cortex": 1.0}
        for _ in range(2000):
            astro.step(rates)
        for name in ["sensory_cortex", "association_cortex", "motor_cortex"]:
            gate = astro.get_plasticity_gate(name)
            assert gate >= 0.1, f"Gate for {name} below minimum: {gate}"

    def test_excitability_gate(self):
        """Excitability gate should also drop with high activity."""
        astro = self._make_astrocytes(metabolic_threshold=0.2, tau_calcium=100.0)
        rates = {"sensory_cortex": 0.8, "association_cortex": 0.0, "motor_cortex": 0.0}
        for _ in range(500):
            astro.step(rates)
        exc_gate = astro.get_excitability_gate("sensory_cortex")
        assert exc_gate < 0.9, f"Excitability gate should drop, got {exc_gate}"

    def test_unknown_region_returns_1(self):
        """Unknown region names should return gate=1.0 (no gating)."""
        astro = self._make_astrocytes()
        assert astro.get_plasticity_gate("nonexistent") == 1.0
        assert astro.get_excitability_gate("nonexistent") == 1.0

    def test_state_persistence(self):
        """Astrocyte state should survive get_state/set_state."""
        astro = self._make_astrocytes()
        rates = {"sensory_cortex": 0.5, "association_cortex": 0.3, "motor_cortex": 0.1}
        for _ in range(100):
            astro.step(rates)
        state = astro.get_state()
        assert "calcium" in state
        assert "gliotransmitter" in state
        # Restore into fresh instance
        astro2 = self._make_astrocytes()
        astro2.set_state(state)
        np.testing.assert_allclose(astro._calcium, astro2._calcium)
        np.testing.assert_allclose(astro._gliotransmitter, astro2._gliotransmitter)

    def test_metrics_output(self):
        """get_metrics() should return calcium, gliotransmitter, and gates."""
        astro = self._make_astrocytes()
        metrics = astro.get_metrics()
        assert "calcium" in metrics
        assert "gliotransmitter" in metrics
        assert "plasticity_gates" in metrics
        assert "sensory_cortex" in metrics["calcium"]

    def test_sigmoid_slope_controls_sharpness(self):
        """Higher sigmoid slope should create sharper transition."""
        astro_sharp = self._make_astrocytes(
            sigmoid_slope=20.0, metabolic_threshold=0.3, tau_calcium=100.0
        )
        astro_gentle = self._make_astrocytes(
            sigmoid_slope=2.0, metabolic_threshold=0.3, tau_calcium=100.0
        )
        rates = {"sensory_cortex": 0.5, "association_cortex": 0.0, "motor_cortex": 0.0}
        for _ in range(300):
            astro_sharp.step(rates)
            astro_gentle.step(rates)
        # Sharp should be more extreme (closer to 0 or 1)
        g_sharp = astro_sharp.get_plasticity_gate("sensory_cortex")
        g_gentle = astro_gentle.get_plasticity_gate("sensory_cortex")
        # Both should gate, but sharp should gate MORE
        assert (
            g_sharp < g_gentle
        ), f"Sharp slope should gate more: sharp={g_sharp}, gentle={g_gentle}"

    def test_slow_timescale(self):
        """Astrocyte dynamics should operate on seconds timescale, not milliseconds."""
        astro = self._make_astrocytes(tau_calcium=5000.0)
        rates = {"sensory_cortex": 0.5, "association_cortex": 0.0, "motor_cortex": 0.0}
        # After 100 steps (~100ms), calcium should still be rising (not saturated)
        for _ in range(100):
            astro.step(rates)
        ca_100 = float(astro._calcium[0])
        # After 1000 more steps (~1s total), calcium should be higher
        for _ in range(1000):
            astro.step(rates)
        ca_1100 = float(astro._calcium[0])
        assert ca_1100 > ca_100, "Calcium should still be rising at 1s (tau=5s)"


class TestAstrocyteNetworkIntegration:
    """Test astrocyte gating through the network path."""

    def test_astrocyte_gate_uses_region_name_string(self):
        """Network must pass region name (str), not BrainRegion object."""
        from neuromorphic.config import NeuromorphicConfig, PopulationConfig
        from neuromorphic.network import NeuromorphicNetwork

        cfg = NeuromorphicConfig(
            populations=PopulationConfig(
                sensory_cortex=50,
                motor_cortex=50,
                brainstem=20,
                association_cortex=50,
                predictive_layer=30,
                working_memory=30,
                cerebellum=20,
                reflex_arc=10,
            ),
            astrocyte=AstrocyteConfig(enabled=True, tau_calcium=50.0),
        )
        net = NeuromorphicNetwork(cfg, seed=42)
        assert net.astrocytes is not None
        # Run enough steps to raise calcium and reduce gate below 1.0
        for _ in range(200):
            net.step(np.random.default_rng(0).random(50, dtype=np.float32) * 30)
        # Verify the gate is actually being used (not silently bypassed)
        gate = net.astrocytes.get_plasticity_gate("association_cortex")
        # With high activity, gate should be < 1.0 (some suppression)
        assert gate < 1.0, f"Astrocyte gate should suppress: got {gate}"
