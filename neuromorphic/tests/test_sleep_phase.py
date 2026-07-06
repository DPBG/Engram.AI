"""Offline sleep-phase consolidation: scheduler transitions + capped replay.

Deterministic (injected clock, stub synapse group) — no live brain needed.
Proves: OFF by default is a no-op; enabled enters/exits on schedule; and the
eligibility-trace replay boost is HARD-CAPPED and clipped to w_max (the
non-negotiable safety property that prevents weight blow-up).
"""

from __future__ import annotations

import numpy as np
import pytest

from neuromorphic.config import SleepPhaseConfig
from neuromorphic.sleep_phase import SleepPhaseManager


class _StubGroup:
    """Minimal SynapseGroup-like object for the replay helper."""

    def __init__(self, weights, eligibility, w_max):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.eligibility = np.asarray(eligibility, dtype=np.float32)
        self.w_max = w_max


def test_disabled_is_noop():
    mgr = SleepPhaseManager(SleepPhaseConfig(enabled=False), clock=lambda: 1e9)
    assert mgr.maybe_tick() is None
    assert mgr.in_sleep is False


def test_enters_and_exits_on_schedule():
    t = {"now": 0.0}
    cfg = SleepPhaseConfig(enabled=True, interval_s=100.0, duration_s=10.0)
    mgr = SleepPhaseManager(cfg, clock=lambda: t["now"])

    assert mgr.maybe_tick() is None  # not yet time
    t["now"] = 100.0
    assert mgr.maybe_tick() == "entered"
    assert mgr.in_sleep is True
    t["now"] = 105.0
    assert mgr.maybe_tick() is None  # still within the window
    t["now"] = 110.0
    assert mgr.maybe_tick() == "exited"
    assert mgr.in_sleep is False


def test_capped_replay_boosts_toward_eligibility_and_clips_to_w_max():
    cfg = SleepPhaseConfig(enabled=True, boost_factor=1.2)
    mgr = SleepPhaseManager(cfg)
    grp = _StubGroup(weights=[0.10, 0.90], eligibility=[1.0, 1.0], w_max=1.0)

    mgr.apply_capped_replay([grp])

    # delta = eligibility * (boost - 1.0) = 1.0 * 0.2 = 0.2
    assert grp.weights[0] == pytest.approx(0.30)  # 0.10 + 0.20
    assert grp.weights[1] == pytest.approx(1.0)  # 0.90 + 0.20 -> clipped to w_max
    assert np.all(grp.weights <= grp.w_max)


def test_replay_never_drives_weights_negative():
    cfg = SleepPhaseConfig(enabled=True, boost_factor=1.2)
    mgr = SleepPhaseManager(cfg)
    grp = _StubGroup(weights=[0.05], eligibility=[-1.0], w_max=1.0)
    mgr.apply_capped_replay([grp])
    assert grp.weights[0] >= 0.0  # floored at 0


def test_boost_above_hard_cap_is_rejected():
    cfg = SleepPhaseConfig(enabled=True, boost_factor=2.0)
    mgr = SleepPhaseManager(cfg)
    grp = _StubGroup(weights=[0.1], eligibility=[1.0], w_max=1.0)
    with pytest.raises(ValueError):
        mgr.apply_capped_replay([grp])
