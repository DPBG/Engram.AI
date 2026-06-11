"""Astrocyte-gated plasticity (CIP-22).

Simulated glial cells that monitor metabolic activity per brain region and
gate synaptic plasticity based on energy cost.  One astrocyte per brain
region.  Each astrocyte integrates local firing rate + weight change
magnitude into a slow calcium signal (tau ~5000ms).  High calcium triggers
gliotransmitter release via a sigmoid, which suppresses plasticity on
synapses projecting to/from that region.

This implements the "fourth factor" from AGMP (Frontiers in Neuroscience,
2025): STDP x eligibility x neuromodulation x astrocyte.

Patent: CIP-22.  Extends Claim 1 (8th mechanism) and Claim 6 (continual
learning).  No conflict with existing 6 claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuromorphic.config import AstrocyteConfig


class AstrocyteNetwork:
    """Network of astrocytes, one per brain region.

    Each astrocyte monitors the metabolic cost of its region and outputs
    a plasticity gate (0..1) and an excitability gate (0..1).

    Lifecycle (called by network.py each step):
        1. step(firing_rates, weight_change_magnitudes) - update calcium
        2. get_plasticity_gate(region_name) -> float - read gate for STDP/elig
        3. get_excitability_gate(region_name) -> float - read gate for current

    The calcium dynamics use a simple leaky integrator:
        calcium += dt * (-calcium / tau_calcium + metabolic_cost)
    where metabolic_cost = firing_rate + abs(mean_weight_change).

    Gliotransmitter is a sigmoid of (calcium - threshold):
        glio = sigmoid(slope * (calcium - threshold))
    and gates plasticity:
        effective_plasticity = 1.0 - glio * (1.0 - plasticity_gate_min)
    """

    def __init__(self, config: AstrocyteConfig, region_names: list[str]):
        self._cfg = config
        self._region_names = list(region_names)
        self._region_idx = {name: i for i, name in enumerate(self._region_names)}
        n = len(region_names)

        # Per-region state
        self._calcium = np.zeros(n, dtype=np.float32)
        self._gliotransmitter = np.zeros(n, dtype=np.float32)

        # Pre-compute decay constants
        self._calcium_decay = np.float32(np.exp(-1.0 / config.tau_calcium))
        self._glio_decay = np.float32(np.exp(-1.0 / config.tau_gliotransmitter))

    @property
    def region_names(self) -> list[str]:
        return self._region_names

    def step(
        self,
        firing_rates: dict[str, float],
        weight_change_magnitudes: dict[str, float] | None = None,
    ) -> None:
        """Update astrocyte calcium and gliotransmitter based on regional activity.

        Args:
            firing_rates: {region_name: firing_rate} for each region.
            weight_change_magnitudes: {region_name: mean |dw|} for post-synaptic
                regions.  Optional; if None, only firing rates are used.
        """
        cfg = self._cfg

        # Decay calcium toward zero
        self._calcium *= self._calcium_decay

        # Accumulate metabolic cost per region
        for name, rate in firing_rates.items():
            idx = self._region_idx.get(name)
            if idx is None:
                continue
            metabolic_cost = rate
            if weight_change_magnitudes and name in weight_change_magnitudes:
                metabolic_cost += weight_change_magnitudes[name]
            # Leaky integration: calcium drifts toward metabolic_cost / tau
            self._calcium[idx] += np.float32(metabolic_cost * (1.0 - self._calcium_decay))

        # Compute gliotransmitter via sigmoid of (calcium - threshold)
        x = cfg.sigmoid_slope * (self._calcium - cfg.metabolic_threshold)
        self._gliotransmitter = 1.0 / (1.0 + np.exp(-x, dtype=np.float32))

    def get_plasticity_gate(self, region_name: str) -> float:
        """Return plasticity multiplier for a region (1.0 = full, gate_min = max suppression)."""
        idx = self._region_idx.get(region_name)
        if idx is None:
            return 1.0
        glio = float(self._gliotransmitter[idx])
        # gate = 1.0 when glio=0, gate_min when glio=1.0
        return 1.0 - glio * (1.0 - self._cfg.plasticity_gate_min)

    def get_excitability_gate(self, region_name: str) -> float:
        """Return excitability multiplier for a region."""
        idx = self._region_idx.get(region_name)
        if idx is None:
            return 1.0
        glio = float(self._gliotransmitter[idx])
        return 1.0 - glio * (1.0 - self._cfg.excitability_gate_min)

    def get_metrics(self) -> dict[str, Any]:
        """Return current astrocyte state for monitoring/dashboard."""
        return {
            "calcium": {name: float(self._calcium[i])
                        for i, name in enumerate(self._region_names)},
            "gliotransmitter": {name: float(self._gliotransmitter[i])
                                for i, name in enumerate(self._region_names)},
            "plasticity_gates": {name: self.get_plasticity_gate(name)
                                 for name in self._region_names},
        }

    def get_state(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "calcium": self._calcium.copy(),
            "gliotransmitter": self._gliotransmitter.copy(),
            "region_names": list(self._region_names),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from persistence (backward compat: missing keys OK)."""
        if "calcium" in state and len(state["calcium"]) == len(self._region_names):
            self._calcium[:] = state["calcium"]
        if "gliotransmitter" in state and len(state["gliotransmitter"]) == len(self._region_names):
            self._gliotransmitter[:] = state["gliotransmitter"]
