"""Spike patterns → motor commands and predictions."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from neuromorphic.config import NeuromorphicConfig, CognitiveActionConfig
from neuromorphic.regions import MotorCortex, PredictiveLayer

logger = logging.getLogger(__name__)


class PopulationVectorDecoder:
    """Decodes per-joint motor intensities from neuron group firing rates.

    Divides each motor sub-range into N groups (N = number of actuators in
    that channel).  Each group's mean firing rate maps to one joint's torque.

    This replaces the single-intensity-per-channel approach that collapses
    thousands of neurons into one float — enabling the brain to learn
    individuated joint control (e.g., alternating left/right leg extension
    for walking gait).

    Only active when ``MotorFeedbackConfig.population_vector`` is True.
    """

    def __init__(self, config: NeuromorphicConfig):
        self._enabled = config.motor_feedback.population_vector
        self._channel_actuators = config.motor_feedback.channel_actuators
        # Pre-compute neuron→actuator mapping per channel (built lazily per
        # motor cortex instance since sub-range sizes depend on population config).
        self._group_maps: dict[str, list[tuple[int, int, str]]] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _build_group_map(
        self, motor_cortex: MotorCortex,
    ) -> dict[str, list[tuple[int, int, str]]]:
        """Build neuron-index → actuator mapping for each physical channel.

        Returns:
            {channel: [(group_start, group_end, actuator_name), ...]}
        """
        group_map: dict[str, list[tuple[int, int, str]]] = {}
        for sr in motor_cortex.sub_ranges:
            actuators = self._channel_actuators.get(sr.name)
            if actuators is None:
                continue
            n_actuators = len(actuators)
            if n_actuators == 0 or sr.size == 0:
                continue
            neurons_per_group = sr.size // n_actuators
            remainder = sr.size % n_actuators
            groups: list[tuple[int, int, str]] = []
            offset = sr.start
            for i, act_name in enumerate(actuators):
                # Distribute remainder neurons across first groups
                g_size = neurons_per_group + (1 if i < remainder else 0)
                groups.append((offset, offset + g_size, act_name))
                offset += g_size
            group_map[sr.name] = groups
        return group_map

    def decode(
        self,
        motor_cortex: MotorCortex,
        spike_history: np.ndarray,
        channel: str,
    ) -> dict[str, float]:
        """Compute per-actuator firing rates for a channel.

        Args:
            motor_cortex: The motor cortex region (for sub-range info).
            spike_history: (window, n_neurons) float32 array of spike history.
            channel: Motor channel name ("locomotion", "manipulation", "head").

        Returns:
            {actuator_name: intensity} dict.  Intensities are raw mean firing
            rates in [0, 1] — same scale as the single-intensity path.
        """
        if not self._enabled:
            return {}
        if channel not in self._channel_actuators:
            return {}

        if self._group_maps is None:
            self._group_maps = self._build_group_map(motor_cortex)

        groups = self._group_maps.get(channel)
        if not groups:
            return {}

        result: dict[str, float] = {}
        for g_start, g_end, act_name in groups:
            if g_start >= g_end:
                # Empty group (fewer neurons than actuators) → neutral
                result[act_name] = 0.0
                continue
            group_window = spike_history[:, g_start:g_end]
            result[act_name] = float(group_window.mean())
        return result


class SpikeDecoder:
    """
    Decodes motor cortex spike patterns into motor commands.

    Uses a sliding window of recent spikes to compute per-channel
    firing rates. Emits motor commands when rates exceed threshold.
    """

    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self._dec = config.decoding
        self._spike_history: deque[np.ndarray] = deque(maxlen=self._dec.window_steps)
        self._cooldowns: dict[str, int] = {}
        self._pop_vec = PopulationVectorDecoder(config)

    def step(self, motor_cortex: MotorCortex) -> list[dict[str, Any]]:
        """
        Process one step of motor cortex output.

        Args:
            motor_cortex: The motor cortex region with current spike state.

        Returns:
            List of motor command dicts (may be empty).
            When population vector is enabled, each dict includes
            ``actuator_intensities: dict[str, float]``.
        """
        spikes = motor_cortex.spikes.copy()
        self._spike_history.append(spikes)

        # Decrement cooldowns
        for ch in list(self._cooldowns):
            self._cooldowns[ch] -= 1
            if self._cooldowns[ch] <= 0:
                del self._cooldowns[ch]

        if len(self._spike_history) < 2:
            return []

        # Compute firing rates per sub-range
        history = np.array(self._spike_history, dtype=np.float32)
        commands = []

        # Skip sub-ranges that have dedicated decoders
        _DEDICATED_DECODER_RANGES = {"speech", "cognitive"}

        for sr in motor_cortex.sub_ranges:
            if sr.name in _DEDICATED_DECODER_RANGES:
                continue
            if sr.name in self._cooldowns:
                continue

            window = history[:, sr.slice()]
            rate = window.mean()

            if rate >= self._dec.firing_threshold:
                # Find peak neuron within sub-range for specificity
                mean_activity = window.mean(axis=0)
                peak_idx = int(np.argmax(mean_activity))

                cmd: dict[str, Any] = {
                    "channel": sr.name,
                    "intensity": float(rate),
                    "peak_neuron": peak_idx,
                    "sub_range_size": sr.size,
                }

                # Per-joint intensities when population vector is enabled
                if self._pop_vec.enabled:
                    actuator_intensities = self._pop_vec.decode(
                        motor_cortex, history, sr.name,
                    )
                    if actuator_intensities:
                        cmd["actuator_intensities"] = actuator_intensities

                commands.append(cmd)
                self._cooldowns[sr.name] = self._dec.cooldown_steps

        return commands

    def reset(self) -> None:
        self._spike_history.clear()
        self._cooldowns.clear()


class PredictionDecoder:
    """
    Extracts prediction confidence and surprise signals from the predictive layer.
    """

    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self._prev_rate: float = 0.0

    def compute_prediction_error(self, predictive: PredictiveLayer) -> float:
        """
        Compute prediction error as change in firing rate.

        Large changes = surprise = high prediction error.

        Returns:
            Prediction error in [0, 1] range.
        """
        rate = predictive.spikes.mean()
        error = abs(rate - self._prev_rate)
        self._prev_rate = rate
        # Normalize — typical firing rates are 0-0.1
        return min(float(error) * 10.0, 1.0)

    def compute_modulation(self, prediction_error: float) -> float:
        """
        Convert prediction error to R-STDP modulation signal.

        Returns:
            Modulation factor for R-STDP learning.
        """
        p = self.config.rstdp
        if prediction_error < 0.2:
            # Good prediction → small reinforcement
            return p.modulation_match
        elif prediction_error > 0.5:
            # High surprise → strong learning
            return p.modulation_mismatch
        else:
            # Interpolate
            t = (prediction_error - 0.2) / 0.3
            return p.modulation_match + t * (p.modulation_mismatch - p.modulation_match)


class SpeechDecoder:
    """Decodes the speech sub-range of motor cortex into token/phoneme indices.

    The brain learns to produce language output through STDP on pathways
    from association cortex → working memory → motor cortex (speech sub-range).
    This decoder monitors the speech sub-range firing rate and extracts
    a population vector that can be mapped to vocabulary tokens.

    During training, this decoder passively records activity — it does NOT
    interfere with learning. Once the network matures, the decoded patterns
    can drive TTS or text output.
    """

    # Vocabulary size for population coding — each token gets a neuron group
    VOCAB_SIZE = 256  # ASCII range initially, expandable

    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self._dec = config.decoding
        self._spike_history: deque[np.ndarray] = deque(maxlen=config.decoding.window_steps)
        self._cooldown: int = 0
        self._output_count: int = 0
        self._enabled = config.decoding.speech_end > config.decoding.head_end

    @property
    def enabled(self) -> bool:
        return self._enabled

    def step(self, motor_cortex: MotorCortex) -> list[dict[str, Any]]:
        """Process one step of speech sub-range activity.

        Returns a list of speech output events (usually empty).
        A speech event is emitted when the speech sub-range firing rate
        exceeds threshold and cooldown has elapsed.
        """
        if not self._enabled:
            return []

        sr = motor_cortex.get_subrange("speech")
        if sr is None or sr.size == 0:
            return []

        speech_spikes = motor_cortex.spikes[sr.slice()].copy()
        self._spike_history.append(speech_spikes)

        # Decrement cooldown (history still accumulated above)
        if self._cooldown > 0:
            self._cooldown -= 1

        if self._cooldown > 0 or len(self._spike_history) < 2:
            return []

        # Compute population vector over the sliding window
        history = np.array(list(self._spike_history), dtype=np.float32)
        mean_rate = float(history.mean())

        if mean_rate < self._dec.firing_threshold:
            return []

        # Population vector: average activity per neuron across window
        pop_vector = history.mean(axis=0)

        # Find peak neuron — its position within the sub-range encodes the token
        peak_idx = int(np.argmax(pop_vector))
        # Map peak neuron index to vocabulary token (population coding)
        vocab_size = min(self.VOCAB_SIZE, sr.size)
        neurons_per_token = max(1, sr.size // vocab_size)
        token_idx = min(peak_idx // neurons_per_token, vocab_size - 1)

        # Confidence: how sharply peaked is the population vector
        if pop_vector.max() > 0:
            confidence = float(pop_vector.max() / (pop_vector.mean() + 1e-8))
        else:
            confidence = 0.0

        self._cooldown = self._dec.cooldown_steps
        self._output_count += 1

        return [{
            "channel": "speech",
            "token_idx": token_idx,
            "confidence": confidence,
            "intensity": mean_rate,
            "peak_neuron": peak_idx,
            "sub_range_size": sr.size,
            "output_number": self._output_count,
        }]

    def reset(self) -> None:
        self._spike_history.clear()
        self._cooldown = 0
        self._output_count = 0

    def get_state(self) -> dict[str, Any]:
        return {"output_count": self._output_count}

    def set_state(self, state: dict[str, Any]) -> None:
        self._output_count = state.get("output_count", 0)


class CognitiveDecoder:
    """Decodes the cognitive sub-range of motor cortex into LLM query actions.

    The brain learns to "ask for help" when prediction error stays high.
    This decoder monitors the cognitive sub-range firing rate and emits
    query commands when the brain is sufficiently confused.

    The decision to query is emergent — it arises from STDP learning on
    the pathways from predictive layer → association → working memory → motor
    cognitive sub-range, not from hard-coded thresholds on prediction error.
    The thresholds here gate the output (prevent noise queries), but the
    actual "should I ask?" decision is learned by the network.
    """

    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self._cog = config.cognitive_action
        self._spike_history: deque[np.ndarray] = deque(maxlen=config.decoding.window_steps)
        self._error_history: deque[float] = deque(maxlen=self._cog.sustained_window)
        self._cooldown: int = 0
        self._query_count: int = 0

    @property
    def enabled(self) -> bool:
        return self._cog.enabled

    def step(
        self,
        motor_cortex: MotorCortex,
        prediction_error: float,
    ) -> list[dict[str, Any]]:
        """Process one step, return cognitive action commands (usually empty).

        A cognitive query is emitted when ALL conditions are met:
        1. The cognitive sub-range has a high enough firing rate
        2. Prediction error has been sustained above threshold
        3. Cooldown period has elapsed since last query
        """
        if not self._cog.enabled:
            return []

        sr = motor_cortex.get_subrange("cognitive")
        if sr is None or sr.size == 0:
            return []

        # Track cognitive sub-range spikes
        cog_spikes = motor_cortex.spikes[sr.slice()].copy()
        self._spike_history.append(cog_spikes)

        # Track prediction error
        self._error_history.append(prediction_error)

        # Decrement cooldown (history still accumulated above)
        if self._cooldown > 0:
            self._cooldown -= 1

        if self._cooldown > 0 or len(self._spike_history) < 2:
            return []

        # Compute cognitive firing rate
        history = np.array(list(self._spike_history), dtype=np.float32)
        cog_rate = float(history.mean())

        # Check sustained prediction error
        if len(self._error_history) < self._cog.sustained_window:
            return []
        mean_error = float(np.mean(list(self._error_history)))

        # Both conditions must be met: brain is firing cognitive neurons AND confused
        if cog_rate < self._cog.confidence_threshold:
            return []
        if mean_error < self._cog.error_threshold:
            return []

        # Emit cognitive query
        self._cooldown = self._cog.cooldown_steps
        self._query_count += 1

        # Determine action type from peak neuron within cognitive sub-range
        # Divide cognitive sub-range into equal parts for each action type
        mean_activity = history.mean(axis=0)
        n_types = len(self._cog.action_types)
        if n_types == 0:
            return []
        chunk = max(1, sr.size // n_types)
        peak_idx = int(np.argmax(mean_activity))
        type_idx = min(peak_idx // chunk, n_types - 1)
        action_type = self._cog.action_types[type_idx]

        return [{
            "type": action_type,
            "intensity": cog_rate,
            "prediction_error": mean_error,
            "query_number": self._query_count,
            "sub_range_size": sr.size,
        }]

    def get_state(self) -> dict[str, Any]:
        return {"query_count": self._query_count}

    def set_state(self, state: dict[str, Any]) -> None:
        self._query_count = state.get("query_count", 0)

    def reset(self) -> None:
        self._spike_history.clear()
        self._error_history.clear()
        self._cooldown = 0
        self._query_count = 0
