"""Homeostatic drive system — energy, damage, temperature, fatigue."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from neuromorphic.config import NeuromorphicConfig, DriveConfig
from neuromorphic.regions import Brainstem

logger = logging.getLogger(__name__)


class HomeostaticDriveSystem:
    """
    Manages internal drive states that create "needs".

    Each drive is a continuous float in [0, 1]:
    - energy: 1.0 = full, decays over time → 0.0 = critical
    - damage: 0.0 = healthy, events increase → 1.0 = critical
    - temperature: 0.5 = optimal, drifts → 0.0/1.0 = extreme
    - fatigue: 0.0 = rested, accumulates → 1.0 = exhausted

    Drive signals are injected into brainstem neurons to create
    urgency-proportional activity.
    """

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        self._cfg: DriveConfig = config.drives
        self._rng = rng or np.random.default_rng()
        self.energy: float = 1.0
        self.damage: float = 0.0
        self.temperature: float = 0.5
        self.fatigue: float = 0.0
        self._update_count: int = 0

    def update(self, dt_seconds: float = 1.0) -> None:
        """Advance drives by dt_seconds of simulated time.

        Drives use exponential decay toward resting values so they
        self-regulate rather than accumulating to extremes forever.
        """
        c = self._cfg

        # Energy: decays from activity, but slowly regenerates toward resting level
        # Resting energy ~0.7 — always partially hungry, never fully depleted
        energy_rest = 0.7
        self.energy -= c.energy_decay * dt_seconds
        self.energy += (energy_rest - self.energy) * c.energy_recovery * dt_seconds
        self.energy = max(0.0, min(1.0, self.energy))

        # Damage slowly recovers (unchanged — already self-healing)
        self.damage = max(0.0, self.damage - c.damage_recovery * dt_seconds)

        # Temperature: mean-reverts toward optimal instead of random walk
        temp_pull = (c.temperature_optimal - self.temperature) * c.temperature_recovery * dt_seconds
        noise = self._rng.normal(0, c.temperature_drift * dt_seconds)
        self.temperature = float(np.clip(self.temperature + temp_pull + noise, 0.0, 1.0))

        # Fatigue: accumulates from activity, recovers toward resting level
        # Resting fatigue ~0.2 — mild baseline, never fully exhausted
        fatigue_rest = 0.2
        self.fatigue += c.fatigue_rate * dt_seconds
        self.fatigue += (fatigue_rest - self.fatigue) * c.fatigue_recovery * dt_seconds
        self.fatigue = max(0.0, min(1.0, self.fatigue))

        # Periodic drive perturbation — natural fluctuation
        self._update_count += 1
        if self._update_count % c.perturbation_interval == 0:
            self.energy = float(np.clip(self.energy + self._rng.normal(0, c.perturbation_std), 0, 1))
            self.damage = float(np.clip(self.damage + self._rng.normal(0, c.perturbation_std * 0.5), 0, 1))
            self.temperature = float(np.clip(self.temperature + self._rng.normal(0, c.perturbation_std), 0, 1))
            self.fatigue = float(np.clip(self.fatigue + self._rng.normal(0, c.perturbation_std * 0.5), 0, 1))

    def handle_event(self, event: dict[str, Any]) -> None:
        """Process an external drive event (e.g., feeding, damage)."""
        event_type = event.get("type", "")

        if event_type == "feed":
            amount = float(event.get("amount", 0.3))
            self.energy = min(1.0, self.energy + amount)
            logger.debug(f"Fed: energy now {self.energy:.2f}")

        elif event_type == "damage":
            amount = float(event.get("amount", 0.2))
            self.damage = min(1.0, self.damage + amount)
            logger.debug(f"Damaged: damage now {self.damage:.2f}")

        elif event_type == "rest":
            amount = float(event.get("amount", 0.3))
            self.fatigue = max(0.0, self.fatigue - amount)
            logger.debug(f"Rested: fatigue now {self.fatigue:.2f}")

        elif event_type == "temperature":
            self.temperature = max(0.0, min(1.0, float(event.get("value", 0.5))))

    def compute_brainstem_current(self, brainstem: Brainstem) -> np.ndarray:
        """
        Convert drive states to current injection for brainstem.

        Higher urgency (further from optimal) = stronger current.
        """
        n = brainstem.n
        current = np.full(n, np.float32(self._cfg.tonic_current), dtype=np.float32)
        gain = 30.0  # current scaling

        # Energy: urgency increases as energy drops
        energy_sr = brainstem.get_subrange("energy")
        if energy_sr:
            urgency = 1.0 - self.energy
            current[energy_sr.slice()] += np.float32(urgency * gain)

        # Damage: urgency proportional to damage
        damage_sr = brainstem.get_subrange("damage")
        if damage_sr:
            current[damage_sr.slice()] += np.float32(self.damage * gain * 1.5)

        # Temperature: urgency from deviation from optimal
        temp_sr = brainstem.get_subrange("temperature")
        if temp_sr:
            deviation = abs(self.temperature - self._cfg.temperature_optimal)
            current[temp_sr.slice()] += np.float32(deviation * gain * 2.0)

        # Fatigue
        fatigue_sr = brainstem.get_subrange("fatigue")
        if fatigue_sr:
            current[fatigue_sr.slice()] += np.float32(self.fatigue * gain)

        return current

    def get_state(self) -> dict[str, float]:
        return {
            "energy": self.energy,
            "damage": self.damage,
            "temperature": self.temperature,
            "fatigue": self.fatigue,
        }

    def set_state(self, state: dict[str, float]) -> None:
        self.energy = max(0.0, min(1.0, state.get("energy", 1.0)))
        self.damage = max(0.0, min(1.0, state.get("damage", 0.0)))
        self.temperature = max(0.0, min(1.0, state.get("temperature", 0.5)))
        self.fatigue = max(0.0, min(1.0, state.get("fatigue", 0.0)))

    def is_critical(self) -> bool:
        """Check if any drive is in critical state."""
        return (
            self.energy < self._cfg.energy_critical
            or self.damage > 0.8
            or abs(self.temperature - self._cfg.temperature_optimal) > 0.4
            or self.fatigue > self._cfg.fatigue_recovery_threshold
        )

    def summary(self) -> str:
        return (
            f"energy={self.energy:.2f} damage={self.damage:.2f} "
            f"temp={self.temperature:.2f} fatigue={self.fatigue:.2f}"
        )
