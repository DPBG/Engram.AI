"""Gamma/theta oscillatory dynamics for attention gating and memory consolidation.

Implements the oscillatory binding mechanism described in Patent §22.1
(Alternative Embodiment) and temporal coding from §26.2.

Gamma oscillations (~40 Hz, period ~25 ms / 25 steps at dt=1ms):
    Gate sensory current for attention binding. Cross-modal phase coherence
    above threshold triggers a binding boost, complementing STDP-based binding.

Theta oscillations (~6 Hz, period ~167 ms / 167 steps at dt=1ms):
    Gate STDP plasticity — encoding enhanced during theta peak, consolidation
    (lower plasticity) during theta trough. This creates natural alternation
    between learning and recall states.

Both oscillators are simple sinusoidal generators. Phase advances each step
by 2*pi*freq*dt. The functional effect is achieved through multiplicative
modulation of sensory gain (gamma) and plasticity rate (theta).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from neuromorphic.config import OscillatoryConfig


class OscillatorBank:
    """Gamma + theta oscillator pair with phase tracking.

    Thread safety: relies on external ``_net_lock`` (asyncio.Lock in service.py)
    to serialize ``step()`` writes and ``get_metrics()`` reads.  Float fields
    are atomic under CPython GIL, but ``_modality_phases`` dict requires the lock.
    """

    def __init__(self, config: OscillatoryConfig):
        self._cfg = config
        # Phase accumulators (radians, wrap at 2*pi)
        self._gamma_phase: float = 0.0
        self._theta_phase: float = 0.0
        # Pre-compute angular velocities (radians per ms at dt=1)
        self._gamma_omega: float = 2.0 * math.pi * config.gamma_freq / 1000.0
        self._theta_omega: float = 2.0 * math.pi * config.theta_freq / 1000.0
        # Per-modality phase tracking for coherence detection
        self._modality_phases: dict[str, float] = {}

    def step(self, dt: float = 1.0) -> None:
        """Advance oscillator phases by dt milliseconds."""
        TWO_PI = 2.0 * math.pi
        self._gamma_phase = (self._gamma_phase + self._gamma_omega * dt) % TWO_PI
        self._theta_phase = (self._theta_phase + self._theta_omega * dt) % TWO_PI

    def gamma_gain(self) -> float:
        """Multiplicative gain for sensory current from gamma oscillation.

        Returns 1.0 ± gamma_amplitude * sin(phase).
        Range: [1 - amplitude, 1 + amplitude] — always positive.
        Peak = attention window, trough = suppression.
        """
        return 1.0 + self._cfg.gamma_amplitude * math.sin(self._gamma_phase)

    def theta_plasticity_factor(self) -> float:
        """Multiplicative factor for STDP learning rate from theta oscillation.

        Returns 1.0 ± theta_amplitude * sin(phase).
        Peak (sin=+1): enhanced encoding — STDP learns faster.
        Trough (sin=-1): consolidation — STDP learns slower.
        Range: [1 - amplitude, 1 + amplitude] — always positive.
        """
        return 1.0 + self._cfg.theta_amplitude * math.sin(self._theta_phase)

    def update_modality_phase(self, modality: str) -> None:
        """Lock a modality's phase to the current gamma phase.

        Called when a modality provides input. Silent modalities drift
        (not updated), creating natural decoherence over time.
        """
        self._modality_phases[modality] = self._gamma_phase

    def cross_modal_coherence(self) -> float:
        """Measure phase alignment across active modalities.

        Returns mean pairwise cosine similarity of phase differences (0.0 to 1.0).
        High coherence → modalities are firing in sync → binding boost.
        cos(0) = 1.0 (in-phase), cos(pi) = -1.0 (anti-phase).
        Mapped to [0, 1] range: (cos(diff) + 1) / 2.
        """
        phases = list(self._modality_phases.values())
        n = len(phases)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = phases[i] - phases[j]
                # Map cos [-1, 1] to [0, 1]
                total += (math.cos(diff) + 1.0) / 2.0
                count += 1
        return total / count if count > 0 else 0.0

    def binding_boost(self) -> float:
        """Additional gain when cross-modal coherence exceeds threshold.

        Returns 1.0 (no boost) or up to 1.5 (strong coherence).
        Scales linearly from threshold to 1.0 coherence.
        """
        coherence = self.cross_modal_coherence()
        thresh = self._cfg.phase_coherence_threshold
        if coherence < thresh:
            return 1.0
        # Linear scale: threshold → 1.0 maps to 1.0 → 1.5 gain
        scale = (coherence - thresh) / (1.0 - thresh + 1e-9)
        return 1.0 + 0.5 * scale

    def get_metrics(self) -> dict[str, float]:
        """Public metrics snapshot for dashboard/monitoring."""
        return {
            "gamma_phase": self._gamma_phase,
            "theta_phase": self._theta_phase,
            "gamma_gain": self.gamma_gain(),
            "theta_factor": self.theta_plasticity_factor(),
            "coherence": self.cross_modal_coherence(),
        }

    # ── State persistence ──────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        return {
            "gamma_phase": self._gamma_phase,
            "theta_phase": self._theta_phase,
            "modality_phases": dict(self._modality_phases),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._gamma_phase = state.get("gamma_phase", 0.0)
        self._theta_phase = state.get("theta_phase", 0.0)
        self._modality_phases = dict(state.get("modality_phases", {}))
