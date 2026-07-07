"""Phase 2.17 Block D6 (2026-06-20): offline sleep consolidation.

Periodic wall-clock-scheduled consolidation window. When `interval_s` of
wall time has passed since the last sleep, enter a `duration_s` window:

* Drop external sensory gain to `sensory_gain_during_sleep` (default 10%)
  so internally-generated replay dominates the input.
* Raise 5HT baseline by `serotonin_boost` (default 1.5x) -- 5HT is the
  consolidation-friendly neuromod profile.
* Replay eligibility traces with a HARD-CAPPED weight boost
  (default 1.2x), clipped to each synapse group's `w_max`.

The cap is non-negotiable safety. Phase 2.1.3 hit an inf-overflow when
unbounded replay multiplied weights past w_max. Both the per-step boost
factor AND the post-application w_max clip must remain.

This module is a scheduling scaffold + a HARD-CAPPED replay helper.
Integration in `service.py` calls `maybe_tick` once per brain step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from neuromorphic.config import SleepPhaseConfig

logger = logging.getLogger(__name__)


class _SynapseGroupLike(Protocol):
    """Minimal interface that the replay helper needs.

    Real `SynapseGroup` (synapses.py) has these attributes; the test
    suite uses a small stub class that conforms.
    """

    weights: np.ndarray  # CSR .data slice or dense; mutated in place
    eligibility: np.ndarray  # same shape as weights
    w_max: float


@dataclass
class _SleepState:
    last_sleep_end_mono: float = 0.0
    in_sleep: bool = False
    sleep_started_mono: float = 0.0


class SleepPhaseManager:
    """Schedules and applies offline sleep consolidation windows.

    Constructed once per brain. `maybe_tick(now)` is called every brain
    step with `time.monotonic()`. Returns True when state transitioned
    so the caller can log + adjust 5HT baseline / sensory gain.

    Callers that want the side effects but not the full replay can
    skip `apply_capped_replay`; the cap helper is a pure function and
    can be called independently from tests.
    """

    def __init__(
        self,
        config: SleepPhaseConfig,
        clock: Callable[[], float] | None = None,
    ):
        self._config = config
        self._clock = clock  # injected for tests
        self._state = _SleepState()

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        import time

        return time.monotonic()

    @property
    def in_sleep(self) -> bool:
        return self._state.in_sleep

    def maybe_tick(self, now: float | None = None) -> str | None:
        """Advance the scheduler. Returns one of:
        * "entered" -- transitioned awake -> asleep this tick
        * "exited"  -- transitioned asleep -> awake this tick
        * None      -- no transition
        """
        if not self._config.enabled:
            return None
        now = self._now() if now is None else now
        st = self._state
        if st.in_sleep:
            if now - st.sleep_started_mono >= self._config.duration_s:
                st.in_sleep = False
                st.last_sleep_end_mono = now
                logger.info(
                    "Sleep phase EXIT after %.1fs",
                    now - st.sleep_started_mono,
                )
                return "exited"
            return None
        # awake -- has interval elapsed?
        if now - st.last_sleep_end_mono >= self._config.interval_s:
            st.in_sleep = True
            st.sleep_started_mono = now
            logger.info(
                "Sleep phase ENTER (interval %.0fs elapsed)",
                self._config.interval_s,
            )
            return "entered"
        return None

    def apply_capped_replay(
        self,
        synapse_groups: list[_SynapseGroupLike],
    ) -> dict[str, float]:
        """Apply a HARD-CAPPED eligibility-trace replay to each group.

        Returns a {group_id_str: total_delta} dict for diagnostics.
        Pure-ish: mutates `weights` in place but returns the change.
        """
        boost = float(self._config.boost_factor)
        if boost > 1.5:
            # Safety net: refuse to apply boost factors that exceed the
            # Phase 2.1.3 inf-overflow safety budget.
            raise ValueError(f"Sleep replay boost_factor {boost} exceeds hard cap 1.5")
        diagnostics: dict[str, float] = {}
        for idx, grp in enumerate(synapse_groups):
            before = grp.weights.copy() if hasattr(grp.weights, "copy") else None
            delta = grp.eligibility * (boost - 1.0)
            grp.weights += delta
            np.minimum(grp.weights, grp.w_max, out=grp.weights)
            np.maximum(grp.weights, 0.0, out=grp.weights)
            if before is not None:
                diagnostics[str(idx)] = float(np.sum(grp.weights - before))
        return diagnostics


__all__ = ["SleepPhaseManager", "SleepPhaseConfig"]
