"""Tests for orienting instincts (novelty, change, cross-modal, habituation)."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.instincts import OrientingInstincts


@pytest.fixture
def config():
    return NeuromorphicConfig()


@pytest.fixture
def instincts(config):
    return OrientingInstincts(config)


class TestNovelty:
    def test_first_appearance_gets_boost(self, instincts):
        """A brand-new modality should get a novelty burst."""
        current = np.ones(100, dtype=np.float32) * 10.0
        active = {"visual": (0, 50)}
        gain = instincts.compute_gain(current, active, step=0)
        # Visual range should have boost > 1
        assert gain[:50].mean() > 1.5

    def test_repeated_modality_no_novelty(self, instincts):
        """After seeing a modality, it shouldn't get novelty boost next step."""
        current = np.ones(100, dtype=np.float32) * 10.0
        active = {"visual": (0, 50)}
        instincts.compute_gain(current, active, step=0)
        gain2 = instincts.compute_gain(current, active, step=1)
        # No novelty boost (but may still have other gains)
        assert gain2[:50].mean() < 2.0

    def test_long_silence_triggers_novelty(self, instincts):
        """After long silence, modality should be treated as novel again."""
        current = np.ones(100, dtype=np.float32) * 10.0
        active = {"visual": (0, 50)}
        instincts.compute_gain(current, active, step=0)
        # Long gap
        gain = instincts.compute_gain(current, active, step=1000)
        assert gain[:50].mean() > 1.5


class TestChangeDetection:
    def test_changed_input_gets_boost(self, instincts):
        """Significantly different input should trigger change detection."""
        active = {"visual": (0, 50)}

        # Build up history with similar inputs
        for step in range(5):
            base = np.ones(100, dtype=np.float32) * 10.0
            instincts.compute_gain(base, active, step=step)

        # Very different input
        changed = np.ones(100, dtype=np.float32) * 10.0
        changed[:50] = 100.0  # big change in visual range
        gain = instincts.compute_gain(changed, active, step=10)
        assert gain[:50].mean() > 1.0


class TestCrossModalBinding:
    def test_multimodal_gets_binding_boost(self, instincts):
        """When 2+ modalities fire together, all should get extra boost."""
        current = np.ones(100, dtype=np.float32) * 10.0
        # Single modality first (step 0 establishes non-novel)
        instincts.compute_gain(current, {"visual": (0, 50)}, step=0)
        instincts.compute_gain(current, {"auditory": (50, 100)}, step=1)

        # Both together
        active = {"visual": (0, 50), "auditory": (50, 100)}
        gain = instincts.compute_gain(current, active, step=2)
        # Cross-modal boost should be present
        assert gain[:50].mean() >= 1.0
        assert gain[50:100].mean() >= 1.0


class TestHabituation:
    def test_habituation_reduces_gain(self, instincts):
        """Repeated identical inputs should produce decreasing gain."""
        current = np.ones(100, dtype=np.float32) * 10.0
        active = {"visual": (0, 50)}

        gains = []
        for step in range(20):
            gain = instincts.compute_gain(current, active, step=step)
            gains.append(gain[:50].mean())

        # Later gains should be lower than earlier ones (habituation)
        assert gains[-1] <= gains[0]

    def test_habituation_recovers_when_silent(self, instincts):
        """Habituation should recover during periods of no input."""
        current = np.ones(100, dtype=np.float32) * 10.0
        active = {"visual": (0, 50)}

        # Habituate
        for step in range(20):
            instincts.compute_gain(current, active, step=step)

        # Silent period (just call with empty)
        for step in range(20, 120):
            instincts.compute_gain(current, {}, step=step)

        # Habituation should have partially recovered
        hab = instincts._habituation.get("visual", 0)
        assert hab < 0.4  # recovered from max


class TestGainFloor:
    """Patent Claim 4: instinctual gain is always multiplicative (>= 1.0).

    Instincts amplify, never suppress. This is a hard architectural constraint.
    """

    def test_gain_floor_single_modality(self, instincts):
        """Gain must be >= 1.0 for all neurons, single modality."""
        current = np.ones(100, dtype=np.float32) * 10.0
        active = {"visual": (0, 50)}
        for step in range(30):
            gain = instincts.compute_gain(current, active, step=step)
            assert (
                gain.min() >= 1.0 - 1e-6
            ), f"Gain dropped below 1.0 at step {step}: min={gain.min():.6f}"

    def test_gain_floor_multi_modality(self, instincts):
        """Gain must be >= 1.0 with multiple modalities active."""
        current = np.ones(100, dtype=np.float32) * 10.0
        active = {"visual": (0, 30), "auditory": (30, 60), "tactile": (60, 90)}
        for step in range(20):
            gain = instincts.compute_gain(current, active, step=step)
            assert gain.min() >= 1.0 - 1e-6, f"Multi-modal gain dropped below 1.0 at step {step}"

    def test_gain_floor_zero_current(self, instincts):
        """Gain must be >= 1.0 even with zero input current."""
        current = np.zeros(100, dtype=np.float32)
        active = {"visual": (0, 50)}
        gain = instincts.compute_gain(current, active, step=0)
        assert gain.min() >= 1.0 - 1e-6

    def test_gain_floor_max_habituation(self, instincts):
        """Even after maximal habituation, gain must not drop below 1.0."""
        current = np.ones(100, dtype=np.float32) * 10.0
        active = {"visual": (0, 50)}
        # Habituate heavily
        for step in range(200):
            gain = instincts.compute_gain(current, active, step=step)
        assert gain.min() >= 1.0 - 1e-6, f"Habituated gain dropped below 1.0: min={gain.min():.6f}"
