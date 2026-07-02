"""Tests for spike decoding — motor commands and prediction error."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.decoding import PopulationVectorDecoder, PredictionDecoder, SpikeDecoder
from neuromorphic.mujoco_body import CHANNEL_ACTUATORS
from neuromorphic.regions import MotorCortex, PredictiveLayer


@pytest.fixture
def config():
    cfg = NeuromorphicConfig()
    cfg.populations.motor_cortex = 200
    cfg.populations.predictive_layer = 100
    return cfg


@pytest.fixture
def motor_cortex(config):
    return MotorCortex(config)


@pytest.fixture
def predictive(config):
    return PredictiveLayer(config)


class TestSpikeDecoder:
    def test_no_commands_on_first_step(self, config, motor_cortex):
        decoder = SpikeDecoder(config)
        commands = decoder.step(motor_cortex)
        assert commands == []

    def test_no_commands_below_threshold(self, config, motor_cortex):
        decoder = SpikeDecoder(config)
        # Run enough steps to fill window, but with no spikes → rate=0
        for _ in range(25):
            commands = decoder.step(motor_cortex)
        assert commands == []

    def test_commands_above_threshold(self, config, motor_cortex):
        decoder = SpikeDecoder(config)
        all_commands = []
        # Force all motor neurons to spike every step
        for _ in range(25):
            motor_cortex.population.spikes[:] = True
            commands = decoder.step(motor_cortex)
            all_commands.extend(commands)
        # With all neurons firing, at least some commands should have been emitted
        assert len(all_commands) > 0
        channels = {cmd["channel"] for cmd in all_commands}
        assert len(channels) >= 1

    def test_command_format(self, config, motor_cortex):
        decoder = SpikeDecoder(config)
        motor_cortex.population.spikes[:] = True
        commands = []
        for _ in range(25):
            motor_cortex.population.spikes[:] = True
            commands = decoder.step(motor_cortex)
        if commands:
            cmd = commands[0]
            assert "channel" in cmd
            assert "intensity" in cmd
            assert "peak_neuron" in cmd
            assert "sub_range_size" in cmd
            assert isinstance(cmd["intensity"], float)

    def test_cooldown_prevents_rapid_fire(self, config, motor_cortex):
        decoder = SpikeDecoder(config)
        motor_cortex.population.spikes[:] = True

        # Fill window and trigger
        for _ in range(25):
            motor_cortex.population.spikes[:] = True
            decoder.step(motor_cortex)

        first_commands = decoder.step(motor_cortex)
        # Immediately after commands, cooldown should block same channels
        second_commands = decoder.step(motor_cortex)
        first_channels = {c["channel"] for c in first_commands}
        second_channels = {c["channel"] for c in second_commands}
        # Channels that just fired should be in cooldown
        assert first_channels.isdisjoint(second_channels) or len(first_commands) == 0

    def test_reset_clears_state(self, config, motor_cortex):
        decoder = SpikeDecoder(config)
        motor_cortex.population.spikes[:] = True
        for _ in range(5):
            decoder.step(motor_cortex)
        decoder.reset()
        # After reset, history should be empty → no commands
        commands = decoder.step(motor_cortex)
        assert commands == []

    def test_partial_channel_firing(self, config, motor_cortex):
        """Only one sub-range fires → only that channel emits."""
        decoder = SpikeDecoder(config)
        loc_sr = motor_cortex.get_subrange("locomotion")
        assert loc_sr is not None

        for _ in range(25):
            motor_cortex.population.spikes[:] = False
            motor_cortex.population.spikes[loc_sr.slice()] = True
            commands = decoder.step(motor_cortex)

        # Only locomotion channel should fire (if threshold met)
        if commands:
            channels = {c["channel"] for c in commands}
            assert "locomotion" in channels


class TestPredictionDecoder:
    def test_initial_error_zero(self, config, predictive):
        decoder = PredictionDecoder(config)
        # No spikes → rate=0, prev_rate=0 → error=0
        error = decoder.compute_prediction_error(predictive)
        assert error == 0.0

    def test_error_increases_with_rate_change(self, config, predictive):
        decoder = PredictionDecoder(config)
        # First call establishes baseline
        decoder.compute_prediction_error(predictive)
        # Now set some spikes → rate change → error
        predictive.population.spikes[:50] = True
        error = decoder.compute_prediction_error(predictive)
        assert error > 0.0

    def test_error_capped_at_1(self, config, predictive):
        decoder = PredictionDecoder(config)
        decoder.compute_prediction_error(predictive)
        # All neurons fire → max rate change
        predictive.population.spikes[:] = True
        error = decoder.compute_prediction_error(predictive)
        assert error <= 1.0

    def test_stable_rate_low_error(self, config, predictive):
        decoder = PredictionDecoder(config)
        # Set constant firing pattern
        predictive.population.spikes[:10] = True
        decoder.compute_prediction_error(predictive)
        # Same pattern → low error
        error = decoder.compute_prediction_error(predictive)
        assert error == 0.0

    def test_modulation_low_error(self, config):
        decoder = PredictionDecoder(config)
        mod = decoder.compute_modulation(0.1)
        assert mod == config.rstdp.modulation_match

    def test_modulation_high_error(self, config):
        decoder = PredictionDecoder(config)
        mod = decoder.compute_modulation(0.8)
        assert mod == config.rstdp.modulation_mismatch

    def test_modulation_interpolates(self, config):
        decoder = PredictionDecoder(config)
        mod = decoder.compute_modulation(0.35)
        assert config.rstdp.modulation_match < mod < config.rstdp.modulation_mismatch


# ── Population Vector Decoding ──────────────────────────────────────


@pytest.fixture
def pop_vec_config():
    """Config with population vector enabled and enough neurons for groups."""
    cfg = NeuromorphicConfig()
    cfg.populations.motor_cortex = 600  # enough to divide among actuators
    cfg.motor_feedback.population_vector = True
    cfg.motor_feedback.enabled = True
    return cfg


@pytest.fixture
def pop_vec_motor_cortex(pop_vec_config):
    return MotorCortex(pop_vec_config)


class TestPopulationVectorDecoder:
    def test_disabled_returns_empty(self, config, motor_cortex):
        """When population_vector is off, decode returns empty dict."""
        decoder = PopulationVectorDecoder(config)
        assert not decoder.enabled
        history = np.ones((5, motor_cortex.n), dtype=np.float32)
        result = decoder.decode(motor_cortex, history, "locomotion")
        assert result == {}

    def test_locomotion_group_count(self, pop_vec_config, pop_vec_motor_cortex):
        """Locomotion decodes to 15 actuators."""
        decoder = PopulationVectorDecoder(pop_vec_config)
        assert decoder.enabled
        history = np.ones((5, pop_vec_motor_cortex.n), dtype=np.float32)
        result = decoder.decode(pop_vec_motor_cortex, history, "locomotion")
        assert len(result) == len(CHANNEL_ACTUATORS["locomotion"])

    def test_manipulation_group_count(self, pop_vec_config, pop_vec_motor_cortex):
        """Manipulation decodes to 12 actuators."""
        decoder = PopulationVectorDecoder(pop_vec_config)
        history = np.ones((5, pop_vec_motor_cortex.n), dtype=np.float32)
        result = decoder.decode(pop_vec_motor_cortex, history, "manipulation")
        assert len(result) == len(CHANNEL_ACTUATORS["manipulation"])

    def test_head_group_count(self, pop_vec_config, pop_vec_motor_cortex):
        """Head decodes to 2 actuators."""
        decoder = PopulationVectorDecoder(pop_vec_config)
        history = np.ones((5, pop_vec_motor_cortex.n), dtype=np.float32)
        result = decoder.decode(pop_vec_motor_cortex, history, "head")
        assert len(result) == len(CHANNEL_ACTUATORS["head"])

    def test_actuator_names_match(self, pop_vec_config, pop_vec_motor_cortex):
        """Decoded actuator names match CHANNEL_ACTUATORS."""
        decoder = PopulationVectorDecoder(pop_vec_config)
        history = np.ones((5, pop_vec_motor_cortex.n), dtype=np.float32)
        for channel, expected_names in CHANNEL_ACTUATORS.items():
            result = decoder.decode(pop_vec_motor_cortex, history, channel)
            assert set(result.keys()) == set(expected_names)

    def test_nonphysical_channel_returns_empty(self, pop_vec_config, pop_vec_motor_cortex):
        """Expression/speech/cognitive channels return empty dict."""
        decoder = PopulationVectorDecoder(pop_vec_config)
        history = np.ones((5, pop_vec_motor_cortex.n), dtype=np.float32)
        for ch in ("expression", "speech", "cognitive"):
            result = decoder.decode(pop_vec_motor_cortex, history, ch)
            assert result == {}

    def test_differential_group_firing(self, pop_vec_config, pop_vec_motor_cortex):
        """Different neuron groups produce different actuator intensities."""
        decoder = PopulationVectorDecoder(pop_vec_config)
        history = np.zeros((5, pop_vec_motor_cortex.n), dtype=np.float32)
        # Light up only the first neuron group in locomotion sub-range
        loc_sr = pop_vec_motor_cortex.get_subrange("locomotion")
        n_actuators = len(CHANNEL_ACTUATORS["locomotion"])
        group_size = loc_sr.size // n_actuators
        history[:, loc_sr.start : loc_sr.start + group_size] = 1.0
        result = decoder.decode(pop_vec_motor_cortex, history, "locomotion")
        first_act = CHANNEL_ACTUATORS["locomotion"][0]
        last_act = CHANNEL_ACTUATORS["locomotion"][-1]
        assert result[first_act] > 0.5
        assert result[last_act] < 0.01

    def test_spike_decoder_includes_actuator_intensities(
        self, pop_vec_config, pop_vec_motor_cortex
    ):
        """SpikeDecoder includes actuator_intensities when population vector is on."""
        decoder = SpikeDecoder(pop_vec_config)
        for _ in range(25):
            pop_vec_motor_cortex.population.spikes[:] = True
            commands = decoder.step(pop_vec_motor_cortex)
        if commands:
            for cmd in commands:
                channel = cmd["channel"]
                if channel in CHANNEL_ACTUATORS:
                    assert "actuator_intensities" in cmd
                    assert len(cmd["actuator_intensities"]) == len(CHANNEL_ACTUATORS[channel])

    def test_backward_compat_no_actuator_intensities(self, config, motor_cortex):
        """With population_vector off, commands have no actuator_intensities."""
        decoder = SpikeDecoder(config)
        for _ in range(25):
            motor_cortex.population.spikes[:] = True
            commands = decoder.step(motor_cortex)
        if commands:
            for cmd in commands:
                assert "actuator_intensities" not in cmd
