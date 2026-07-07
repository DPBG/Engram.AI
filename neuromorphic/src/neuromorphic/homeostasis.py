"""Tiered preventive homeostasis (Phase 1.3, 2026-05-31).

Replaces the binary emergency homeostasis pattern (one threshold -> panic
3x scaling) with a three-tier scheduler that applies continuous, graduated
braking. Most groups live in tier 1 most of the time; tier 2 / tier 3
activate only on severe saturation.

Thresholds are expressed as ratios of each group's ``w_max`` (per-group),
so groups with reduced w_max (e.g. cerebellum at 0.75 from Phase 1.5)
get proportional tiers automatically.

Hysteresis: each tier has separate enter / leave thresholds (5-point
default band) to prevent boundary oscillation.

Patent Claim 1e (homeostatic scaling) is strengthened: scaling is now
graduated by saturation level instead of binary on/off, and runs more
preventively so the brain rarely reaches the emergency tier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HomeostasisThresholds:
    """Per-tier enter/leave thresholds for one group (absolute weight values)."""

    t1_enter: float
    t1_leave: float
    t2_enter: float
    t2_leave: float
    t3_enter: float
    t3_leave: float


class HomeostasisScheduler:
    """Per-group, hysteresis-gated tier scheduler.

    Stateful: remembers each group's current tier so that hysteresis can
    keep a group in its current tier while the mean weight is inside the
    enter/leave band.

    Tier definitions (Phase 1.3 plan):

      Tier | Enter (ratio of w_max) | Leave    | Plan action
      -----|------------------------|----------|------------------------------
       1   | tier_3 enter - 0.15    | enter-5% | 0.5%/step gentle pull
       2   | tier_3 enter - 0.10    | enter-5% | 1.5%/step + BCM theta bump
       3   | emergency_ceiling      | enter-5% | overshoot-scaled, cap 0.15

    For motor-output groups (those passed in ``motor_groups``) the tier 3
    enter is ``motor_ceiling`` instead of ``emergency_ceiling`` so motor
    paths brake earlier (existing behavior preserved).

    Tier 3 may also be triggered by a high post-region firing rate
    (``post_firing_rate > 0.30``), per the plan.
    """

    def __init__(
        self,
        emergency_ceiling: float = 0.85,
        motor_ceiling: float = 0.75,
        motor_groups: frozenset[str] = frozenset(),
        tier1_offset: float = 0.15,
        tier2_offset: float = 0.10,
        leave_hysteresis: float = 0.05,
        rate_tier3_enter: float = 0.30,
        rate_tier3_leave: float = 0.25,
    ) -> None:
        self._emergency_ceiling = float(emergency_ceiling)
        self._motor_ceiling = float(motor_ceiling)
        self._motor_groups = motor_groups
        self._t1_off = float(tier1_offset)
        self._t2_off = float(tier2_offset)
        self._hyst = float(leave_hysteresis)
        self._rate_t3_enter = float(rate_tier3_enter)
        self._rate_t3_leave = float(rate_tier3_leave)

        # Per-group current tier (0 = below tier 1)
        self._tier_per_group: dict[str, int] = {}
        # Telemetry counters: index = tier number, increments each evaluate() call
        # where the group is at that tier. Reset by ``drain_telemetry``.
        self._fire_counts: list[int] = [0, 0, 0, 0]
        # Per-group BCM-bump-applied flag: True once a group entered tier 2 since
        # its last drop below tier 2.  Prevents continuous theta inflation while
        # the group sits in steady-state tier 2.
        self._bcm_bumped: dict[str, bool] = {}

    def thresholds_for(self, group_name: str, w_max: float) -> HomeostasisThresholds:
        """Compute absolute weight thresholds for a group, given its w_max."""
        is_motor = group_name in self._motor_groups
        ceiling = self._motor_ceiling if is_motor else self._emergency_ceiling
        t3_enter = ceiling * w_max
        t2_enter = max(0.0, (ceiling - self._t2_off) * w_max)
        t1_enter = max(0.0, (ceiling - self._t1_off) * w_max)
        return HomeostasisThresholds(
            t1_enter=t1_enter,
            t1_leave=max(0.0, t1_enter - self._hyst * w_max),
            t2_enter=t2_enter,
            t2_leave=max(0.0, t2_enter - self._hyst * w_max),
            t3_enter=t3_enter,
            t3_leave=max(0.0, t3_enter - self._hyst * w_max),
        )

    def evaluate(
        self,
        group_name: str,
        mean_w: float,
        w_max: float,
        post_firing_rate: float = 0.0,
    ) -> int:
        """Compute new tier for the group, applying hysteresis.

        Returns the new tier (0-3) and updates internal state.
        Multi-tier escalation (e.g. 0 -> 3 in one call when mean very high)
        is supported.  De-escalation is one tier at a time to avoid
        oscillation.
        """
        th = self.thresholds_for(group_name, w_max)
        current = self._tier_per_group.get(group_name, 0)

        if mean_w > th.t3_enter:
            new_tier = 3
        elif mean_w > th.t2_enter:
            new_tier = 2 if current < 3 else (3 if mean_w > th.t3_leave else 2)
        elif mean_w > th.t1_enter:
            new_tier = 1 if current < 2 else (2 if mean_w > th.t2_leave else 1)
        elif current >= 3 and mean_w > th.t3_leave:
            new_tier = 3
        elif current >= 2 and mean_w > th.t2_leave:
            new_tier = 2
        elif current >= 1 and mean_w > th.t1_leave:
            new_tier = 1
        else:
            new_tier = 0

        # Firing-rate override: high post-region rate forces tier 3 with its
        # own hysteresis band (enter 0.30, leave 0.25).
        if post_firing_rate > self._rate_t3_enter:
            new_tier = 3
        elif current >= 3 and post_firing_rate > self._rate_t3_leave and mean_w > th.t2_enter:
            # Firing rate still elevated AND weights still in tier 2+ range:
            # stay in tier 3.  Do not let weight-only de-escalation drop below 3
            # while firing rate is in the 0.25-0.30 hysteresis band.
            new_tier = 3

        # BCM bump bookkeeping: clear flag when group falls below tier 2 so a
        # future re-entry can bump again.
        if new_tier < 2:
            self._bcm_bumped[group_name] = False

        self._tier_per_group[group_name] = new_tier
        if 1 <= new_tier <= 3:
            self._fire_counts[new_tier] += 1
        return new_tier

    def should_bump_bcm(self, group_name: str) -> bool:
        """Return True exactly once per tier-2 entry, then False until the
        group falls below tier 2 again.

        Caller is responsible for actually performing the theta bump.
        """
        if self._tier_per_group.get(group_name, 0) < 2:
            return False
        if self._bcm_bumped.get(group_name, False):
            return False
        self._bcm_bumped[group_name] = True
        return True

    def any_tier_at_or_above(self, tier: int) -> bool:
        """True if any group is currently in ``tier`` or above.

        Used by callers to switch the evaluation interval between every-N-steps
        (calm) and every-step (responsive brake).
        """
        if not self._tier_per_group:
            return False
        return max(self._tier_per_group.values()) >= tier

    def drain_telemetry(self) -> tuple[int, int, int]:
        """Return (tier1_fires, tier2_fires, tier3_fires) since last drain and reset."""
        t1, t2, t3 = self._fire_counts[1], self._fire_counts[2], self._fire_counts[3]
        self._fire_counts = [0, 0, 0, 0]
        return t1, t2, t3

    def get_tier(self, group_name: str) -> int:
        """Read current tier without mutating state.  Used by tests / dashboard."""
        return self._tier_per_group.get(group_name, 0)

    def reset(self) -> None:
        """Clear all per-group state.  Used by tests; not called in production."""
        self._tier_per_group.clear()
        self._bcm_bumped.clear()
        self._fire_counts = [0, 0, 0, 0]
