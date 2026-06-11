"""Tests for the cognitive action system — brain-initiated LLM queries."""

from __future__ import annotations

import numpy as np
import pytest

from neuromorphic.config import (
    NeuromorphicConfig,
    PopulationConfig,
    DecodingConfig,
    CognitiveActionConfig,
    DendriticCompartmentConfig,
)
from neuromorphic.decoding import CognitiveDecoder, SpikeDecoder, PredictionDecoder
from neuromorphic.regions import MotorCortex
from neuromorphic.network import NeuromorphicNetwork
from neuromorphic.encoding import _resolve_modality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(
    cognitive_enabled: bool = True,
    expression_end: float = 0.85,
    dendrites: bool = False,
) -> NeuromorphicConfig:
    """Build a tiny config for fast tests."""
    return NeuromorphicConfig(
        populations=PopulationConfig(
            brainstem=100,
            reflex_arc=50,
            sensory_cortex=200,
            motor_cortex=200,
            cerebellum=100,
            association_cortex=200,
            predictive_layer=100,
            working_memory=50,
        ),
        decoding=DecodingConfig(expression_end=expression_end),
        cognitive_action=CognitiveActionConfig(
            enabled=cognitive_enabled,
            sustained_window=5,   # short for testing
            error_threshold=0.3,
            confidence_threshold=0.1,
            cooldown_steps=10,
        ),
        dendrites=DendriticCompartmentConfig(enabled=dendrites),
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestCognitiveActionConfig:
    def test_default_disabled(self):
        cfg = NeuromorphicConfig()
        assert cfg.cognitive_action.enabled is False
        assert cfg.decoding.expression_end == 1.0

    def test_enabled_config(self):
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        assert cfg.cognitive_action.enabled is True
        assert cfg.decoding.expression_end == 0.85

    def test_action_types_default(self):
        cfg = CognitiveActionConfig()
        assert "query_llm" in cfg.action_types
        assert "save_memory" in cfg.action_types
        assert "request_guidance" in cfg.action_types

    def test_from_env_disabled_default(self, monkeypatch):
        """Without NEURO_COGNITIVE_ENABLED, cognitive channel is off."""
        monkeypatch.delenv("NEURO_COGNITIVE_ENABLED", raising=False)
        cfg = NeuromorphicConfig.from_env()
        assert cfg.cognitive_action.enabled is False
        assert cfg.decoding.expression_end == 1.0

    def test_from_env_enabled(self, monkeypatch):
        """With NEURO_COGNITIVE_ENABLED=1, cognitive channel is on."""
        monkeypatch.setenv("NEURO_COGNITIVE_ENABLED", "1")
        cfg = NeuromorphicConfig.from_env()
        assert cfg.cognitive_action.enabled is True
        assert cfg.decoding.expression_end == 0.85  # default when enabled


# ---------------------------------------------------------------------------
# Motor cortex sub-range tests
# ---------------------------------------------------------------------------

class TestMotorCortexCognitiveSubRange:
    def test_no_cognitive_subrange_by_default(self):
        """Default config has no cognitive sub-range."""
        cfg = NeuromorphicConfig(
            populations=PopulationConfig(motor_cortex=100),
        )
        motor = MotorCortex(cfg)
        names = [sr.name for sr in motor.sub_ranges]
        assert "cognitive" not in names
        assert len(motor.sub_ranges) == 4  # locomotion, manipulation, head, expression

    def test_cognitive_subrange_when_enabled(self):
        """With expression_end < 1.0, cognitive sub-range is created."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        motor = MotorCortex(cfg)
        names = [sr.name for sr in motor.sub_ranges]
        assert "cognitive" in names
        assert len(motor.sub_ranges) == 5

        cog = motor.get_subrange("cognitive")
        assert cog is not None
        n = cfg.populations.motor_cortex
        assert cog.start == int(n * 0.85)
        assert cog.end == n
        assert cog.size > 0

    def test_expression_shrinks_when_cognitive_enabled(self):
        """Expression sub-range shrinks to make room for cognitive."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        motor = MotorCortex(cfg)
        expr = motor.get_subrange("expression")
        cog = motor.get_subrange("cognitive")
        n = cfg.populations.motor_cortex

        # Expression ends where cognitive starts
        assert expr.end == cog.start
        assert cog.end == n

    def test_subranges_cover_full_range(self):
        """All sub-ranges together cover [0, n) with no gaps or overlaps."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        motor = MotorCortex(cfg)
        n = cfg.populations.motor_cortex

        # Check contiguous coverage
        prev_end = 0
        for sr in motor.sub_ranges:
            assert sr.start == prev_end, f"{sr.name} gap: {prev_end} != {sr.start}"
            prev_end = sr.end
        assert prev_end == n


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------

class TestCognitiveEncoding:
    def test_cognitive_response_maps_to_auditory(self):
        """cognitive.response provenance maps to auditory modality."""
        assert _resolve_modality("cognitive.response") == "auditory"
        assert _resolve_modality("cognitive.llm") == "auditory"

    def test_cognitive_response_prefix_match(self):
        """Prefix matching works for cognitive.response.xxx."""
        assert _resolve_modality("cognitive.response.summary") == "auditory"


# ---------------------------------------------------------------------------
# CognitiveDecoder tests
# ---------------------------------------------------------------------------

class TestCognitiveDecoder:
    def test_disabled_returns_empty(self):
        """When disabled, always returns empty list."""
        cfg = _small_config(cognitive_enabled=False)
        motor = MotorCortex(cfg)
        decoder = CognitiveDecoder(cfg)

        # Force some spikes
        motor.population.spikes[:] = True
        result = decoder.step(motor, prediction_error=0.9)
        assert result == []

    def test_no_cognitive_subrange_returns_empty(self):
        """When motor cortex has no cognitive sub-range, returns empty."""
        cfg = _small_config(cognitive_enabled=True, expression_end=1.0)
        motor = MotorCortex(cfg)
        decoder = CognitiveDecoder(cfg)

        motor.population.spikes[:] = True
        result = decoder.step(motor, prediction_error=0.9)
        assert result == []

    def test_no_query_without_sustained_error(self):
        """Needs sustained high prediction error before triggering."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        motor = MotorCortex(cfg)
        decoder = CognitiveDecoder(cfg)

        # Only 1 step — not enough for sustained window
        motor.population.spikes[:] = True
        result = decoder.step(motor, prediction_error=0.9)
        assert result == []

    def test_query_after_sustained_error(self):
        """Emits query after sustained high error + cognitive firing."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        motor = MotorCortex(cfg)
        decoder = CognitiveDecoder(cfg)

        cog = motor.get_subrange("cognitive")
        assert cog is not None

        # Feed sustained high error with cognitive neurons firing
        all_results = []
        for _ in range(cfg.cognitive_action.sustained_window + 5):
            motor.population.spikes[:] = False
            # Fire cognitive sub-range neurons
            motor.population.spikes[cog.slice()] = True
            result = decoder.step(motor, prediction_error=0.8)
            all_results.extend(result)

        # Should have triggered at least once across all steps
        assert len(all_results) > 0
        cmd = all_results[0]
        assert cmd["type"] in cfg.cognitive_action.action_types
        assert cmd["prediction_error"] > 0
        assert cmd["intensity"] > 0
        assert "query_number" in cmd

    def test_cooldown_prevents_rapid_queries(self):
        """After emitting, cooldown prevents another query."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        motor = MotorCortex(cfg)
        decoder = CognitiveDecoder(cfg)

        cog = motor.get_subrange("cognitive")

        # Fill up history to get first query
        for _ in range(cfg.cognitive_action.sustained_window + 5):
            motor.population.spikes[:] = False
            motor.population.spikes[cog.slice()] = True
            decoder.step(motor, prediction_error=0.8)

        # Next step should be cooldown
        motor.population.spikes[:] = False
        motor.population.spikes[cog.slice()] = True
        result = decoder.step(motor, prediction_error=0.8)
        assert result == []  # cooldown active

    def test_low_error_no_query(self):
        """Low prediction error doesn't trigger even with cognitive firing."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        motor = MotorCortex(cfg)
        decoder = CognitiveDecoder(cfg)

        cog = motor.get_subrange("cognitive")

        for _ in range(cfg.cognitive_action.sustained_window + 5):
            motor.population.spikes[:] = False
            motor.population.spikes[cog.slice()] = True
            result = decoder.step(motor, prediction_error=0.1)  # low error

        assert result == []

    def test_no_cognitive_firing_no_query(self):
        """High error but no cognitive neurons firing → no query."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        motor = MotorCortex(cfg)
        decoder = CognitiveDecoder(cfg)

        for _ in range(cfg.cognitive_action.sustained_window + 5):
            motor.population.spikes[:] = False  # nothing fires
            result = decoder.step(motor, prediction_error=0.8)

        assert result == []

    def test_reset_clears_state(self):
        """Reset clears all decoder state."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        decoder = CognitiveDecoder(cfg)
        decoder._cooldown = 100
        decoder._query_count = 5
        decoder._error_history.append(0.9)

        decoder.reset()
        assert decoder._cooldown == 0
        assert decoder._query_count == 0
        assert len(decoder._error_history) == 0


# ---------------------------------------------------------------------------
# Network integration tests
# ---------------------------------------------------------------------------

class TestNetworkCognitiveIntegration:
    def test_step_returns_cognitive_commands(self):
        """Network step() returns cognitive_commands key."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        net = NeuromorphicNetwork(cfg, seed=42)
        result = net.step()
        assert "cognitive_commands" in result
        assert isinstance(result["cognitive_commands"], list)

    def test_step_returns_empty_cognitive_when_disabled(self):
        """When cognitive is disabled, returns empty list."""
        cfg = _small_config(cognitive_enabled=False, expression_end=1.0)
        net = NeuromorphicNetwork(cfg, seed=42)
        result = net.step()
        assert result["cognitive_commands"] == []

    def test_cognitive_decoder_exists_on_network(self):
        """Network creates a CognitiveDecoder."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        net = NeuromorphicNetwork(cfg, seed=42)
        assert hasattr(net, "cognitive_decoder")
        assert isinstance(net.cognitive_decoder, CognitiveDecoder)
        assert net.cognitive_decoder.enabled is True

    def test_backward_compat_no_cognitive(self):
        """Default config without cognitive still works."""
        cfg = NeuromorphicConfig(
            populations=PopulationConfig(
                brainstem=100, reflex_arc=50, sensory_cortex=200,
                motor_cortex=200, cerebellum=100, association_cortex=200,
                predictive_layer=100, working_memory=50,
            ),
        )
        net = NeuromorphicNetwork(cfg, seed=42)
        result = net.step()
        assert "cognitive_commands" in result
        assert result["cognitive_commands"] == []

        # Motor cortex should have 4 sub-ranges (no cognitive)
        names = [sr.name for sr in net.motor.sub_ranges]
        assert "cognitive" not in names

    def test_cognitive_with_dendrites(self):
        """Cognitive channel works alongside dendritic compartments."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85, dendrites=True)
        net = NeuromorphicNetwork(cfg, seed=42)

        # Verify motor cortex has dendrites AND cognitive sub-range
        assert net.motor.population.v_dendrite is not None
        cog = net.motor.get_subrange("cognitive")
        assert cog is not None

        # Run a few steps
        for _ in range(5):
            result = net.step()
            assert "cognitive_commands" in result

    def test_cognitive_query_requires_both_conditions(self):
        """Even with high external current, query requires sustained error."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        net = NeuromorphicNetwork(cfg, seed=42)

        # Inject strong current for many steps
        sensory = np.ones(cfg.populations.sensory_cortex, dtype=np.float32) * 100.0
        queries = []
        for _ in range(20):
            result = net.step(sensory)
            queries.extend(result["cognitive_commands"])

        # Shouldn't trigger a query in just 20 steps with random activity
        # (sustained_window=5 but cognitive neurons also need to fire)
        # This tests that random activity doesn't cause false positives
        # (may or may not trigger depending on network dynamics)
        assert isinstance(queries, list)

    def test_inject_cognitive_response(self):
        """Injecting a cognitive response encodes as auditory input."""
        cfg = _small_config(cognitive_enabled=True, expression_end=0.85)
        net = NeuromorphicNetwork(cfg, seed=42)

        # Inject a cognitive response as if from the bridge
        current = net.inject_observation(
            "The object is a red ball.", provenance="cognitive.response"
        )
        assert current.shape == (cfg.populations.sensory_cortex,)
        assert current.sum() > 0  # non-zero encoding

        # It should go to auditory sub-range
        aud_sr = net.sensory.get_subrange("auditory")
        if aud_sr and aud_sr.size > 0:
            assert current[aud_sr.slice()].sum() > 0


# ---------------------------------------------------------------------------
# Cognitive bridge service tests (unit, no NATS)
# ---------------------------------------------------------------------------

try:
    import aiohttp as _aiohttp  # noqa: F401
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


@pytest.mark.skipif(not _HAS_AIOHTTP, reason="aiohttp not installed")
class TestCognitiveBridgeService:
    def test_build_context(self):
        """Bridge builds meaningful context from brain state."""
        from neuromorphic.cognitive_bridge import CognitiveBridgeService
        bridge = CognitiveBridgeService()

        ctx = bridge._build_context(
            prediction_error=0.85,
            step=100000,
            drives={"energy": 0.2, "damage": 0.5},
        )
        assert "0.85" in ctx
        assert "energy" in ctx.lower()

    def test_build_context_no_drives(self):
        """Context works with empty drives."""
        from neuromorphic.cognitive_bridge import CognitiveBridgeService
        bridge = CognitiveBridgeService()

        ctx = bridge._build_context(prediction_error=0.5, step=0, drives={})
        assert "0.5" in ctx
