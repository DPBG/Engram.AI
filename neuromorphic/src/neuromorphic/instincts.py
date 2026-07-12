"""Orienting instincts — innate attention biases that nudge the brain.

These are biological-style reflexive attention mechanisms:
- Novelty: burst when a new or long-silent modality appears
- Change detection: amplify inputs that differ from recent history
- Cross-modal binding: boost when multiple modalities fire together
- Habituation: attenuate repeated identical inputs

Applied as multiplicative gain on sensory current before it enters
the cortex. Does NOT override STDP learning — just nudges attention
so the brain notices important signals during early development.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from neuromorphic.config import NeuromorphicConfig

logger = logging.getLogger(__name__)


class OrientingInstincts:
    """Innate attention biases applied to sensory current.

    Computes a per-element gain multiplier for the sensory current array
    based on novelty, change, and cross-modal signals.

    All gains are multiplicative and >= 1.0 (instincts amplify, never suppress).
    The boost decays over time as the signal becomes familiar (habituation).

    Phase-dependent: during adolescent phase, instinct gains are amplified
    to model heightened emotional reactivity. During mature phase, gains dampen.
    """

    # Known developmental phases
    _VALID_PHASES = {"infant", "toddler", "juvenile", "adolescent", "mature"}

    def __init__(self, config: NeuromorphicConfig):
        cfg = config.instincts if hasattr(config, "instincts") else None

        # Novelty: how long since each modality was last active
        self._last_active_step: dict[str, int] = {}
        self._novelty_boost = getattr(cfg, "novelty_boost", 3.0)
        self._novelty_window = getattr(cfg, "novelty_window", 500)  # steps of silence → novel

        # Change detection: compare current input to recent history
        self._recent_inputs: dict[str, deque[np.ndarray]] = {}
        self._change_history_len = getattr(cfg, "change_history_len", 5)
        self._change_boost = getattr(cfg, "change_boost", 2.0)
        self._change_threshold = getattr(cfg, "change_threshold", 0.3)  # normalized diff

        # Cross-modal binding: boost when N+ modalities fire together
        self._crossmodal_boost = getattr(cfg, "crossmodal_boost", 1.5)
        self._crossmodal_min_modalities = getattr(cfg, "crossmodal_min_modalities", 2)

        # Habituation: decreasing response to repeated identical inputs
        self._habituation: dict[str, float] = {}  # modality → habituation level [0, 1]
        self._habituation_rate = getattr(cfg, "habituation_rate", 0.02)
        self._habituation_recovery = getattr(cfg, "habituation_recovery", 0.005)
        self._habituation_cap = getattr(cfg, "habituation_cap", 0.8)

        # Phase-dependent gain scales from config
        self._phase_scales: dict[str, dict[str, float]] = {
            "infant": {"novelty": 1.0, "crossmodal": 1.0},
            "toddler": {"novelty": 1.0, "crossmodal": 1.0},
            "juvenile": {"novelty": 1.0, "crossmodal": 1.0},
            "adolescent": {
                "novelty": getattr(cfg, "adolescent_novelty_scale", 1.17),
                "crossmodal": getattr(cfg, "adolescent_crossmodal_scale", 1.33),
            },
            "mature": {
                "novelty": getattr(cfg, "mature_novelty_scale", 0.5),
                "crossmodal": getattr(cfg, "mature_crossmodal_scale", 0.8),
            },
        }

        # C3: Smooth phase transitions — interpolate over N steps
        self._phase_transition_steps = getattr(cfg, "phase_transition_steps", 10_000)
        self._phase: str = "infant"
        self._prev_phase: str = "infant"
        self._transition_step: int = 0
        self._transition_progress: float = 1.0  # 1.0 = fully transitioned

        # H7: Per-pathway instinct gains (computed each step, read by network)
        self.last_per_modality_gain: dict[str, float] = {}

    def set_phase(self, phase: str) -> None:
        """Update the developmental phase for gain scaling."""
        # E1: Validate phase name
        if phase not in self._VALID_PHASES:
            logger.warning(
                f"Unknown developmental phase '{phase}', keeping current '{self._phase}'"
            )
            return
        if phase != self._phase:
            # C3: Start smooth transition (progression driven by compute_gain)
            self._prev_phase = self._phase
            self._phase = phase
            self._transition_step = 0
            self._transition_progress = 0.0

    def compute_gain(
        self,
        sensory_current: np.ndarray,
        active_modalities: dict[str, tuple[int, int]],
        step: int,
    ) -> np.ndarray:
        """Compute instinctual gain multiplier for sensory current.

        Args:
            sensory_current: Raw encoded sensory current (n_sensory,).
            active_modalities: Dict of modality → (start, end) subrange indices
                              for modalities that have data this step.
            step: Current simulation step number.

        Returns:
            Gain array of shape (n_sensory,), all values >= 1.0.
        """
        n = len(sensory_current)
        gain = np.ones(n, dtype=np.float32)

        if not active_modalities:
            # Recover habituation for all modalities when nothing is active
            for mod in self._habituation:
                self._habituation[mod] = max(
                    0.0, self._habituation[mod] - self._habituation_recovery
                )
            return gain

        active_names = set(active_modalities.keys())
        n_active = len(active_names)

        # C3: Smooth phase transition — interpolate between previous and current scales
        if self._transition_progress < 1.0:
            self._transition_step += 1
            self._transition_progress = min(
                1.0,
                self._transition_step / max(1, self._phase_transition_steps),
            )
        prev_scales = self._phase_scales.get(self._prev_phase, {"novelty": 1.0, "crossmodal": 1.0})
        curr_scales = self._phase_scales.get(self._phase, {"novelty": 1.0, "crossmodal": 1.0})
        t = self._transition_progress
        novelty_scale = prev_scales["novelty"] * (1.0 - t) + curr_scales["novelty"] * t
        crossmodal_scale = prev_scales["crossmodal"] * (1.0 - t) + curr_scales["crossmodal"] * t

        for modality, (start, end) in active_modalities.items():
            if start >= end:
                continue
            sl = slice(start, end)
            mod_gain = 1.0

            # --- Novelty (phase-scaled) ---
            last = self._last_active_step.get(modality, -self._novelty_window - 1)
            steps_since = step - last
            if steps_since >= self._novelty_window:
                effective_novelty = 1.0 + (self._novelty_boost - 1.0) * novelty_scale
                mod_gain *= effective_novelty
                logger.debug(f"Novelty burst for {modality} (silent {steps_since} steps)")

            # --- Change detection ---
            current_data = sensory_current[sl].copy()
            if np.any(current_data > 0):
                history = self._recent_inputs.get(modality)
                if history and len(history) > 0:
                    # Filter to same-size entries (range may have been reallocated)
                    compatible = [h for h in history if len(h) == len(current_data)]
                    if compatible:
                        recent_mean = np.mean(compatible, axis=0)
                        norm = np.linalg.norm(current_data) + 1e-8
                        diff = np.linalg.norm(current_data - recent_mean) / norm
                        if diff > self._change_threshold:
                            change_factor = min(diff / self._change_threshold, self._change_boost)
                            mod_gain *= change_factor
                    else:
                        # Range was reallocated — clear stale history
                        history.clear()

                # Update history
                if modality not in self._recent_inputs:
                    self._recent_inputs[modality] = deque(maxlen=self._change_history_len)
                self._recent_inputs[modality].append(current_data)

            # --- Cross-modal binding (phase-scaled) ---
            if n_active >= self._crossmodal_min_modalities:
                effective_crossmodal = 1.0 + (self._crossmodal_boost - 1.0) * crossmodal_scale
                mod_gain *= effective_crossmodal

            # --- Habituation (attenuate the BOOST, not below baseline) ---
            # Habituation reduces the extra gain above 1.0, never below 1.0.
            # E.g. novelty=3.0, hab=0.5 → boost=(3.0-1.0)*(1-0.5)+1.0 = 2.0
            # This gives a smooth decay from boosted → baseline as habituation grows.
            hab = self._habituation.get(modality, 0.0)
            if mod_gain > 1.0:
                boost = mod_gain - 1.0
                mod_gain = 1.0 + boost * (1.0 - hab)
            # Increase habituation for this modality (cap from config)
            self._habituation[modality] = min(self._habituation_cap, hab + self._habituation_rate)

            # Apply gain to this modality's subrange (always >= 1.0)
            gain[sl] *= np.float32(mod_gain)

            # H7: Record per-modality gain for per-pathway STDP modulation
            self.last_per_modality_gain[modality] = mod_gain

            # Update last-active timestamp
            self._last_active_step[modality] = step

        # Recover habituation for inactive modalities
        for mod in list(self._habituation.keys()):
            if mod not in active_names:
                self._habituation[mod] = max(
                    0.0, self._habituation[mod] - self._habituation_recovery
                )

        return gain
