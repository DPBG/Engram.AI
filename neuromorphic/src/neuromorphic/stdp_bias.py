"""STDP downward-bias auto-tuner (Phase 1.4, 2026-05-31).

When the watchdog reports persistent monotonic weight drift on a synapse
group, this tuner nudges the group's ``stdp_bias_compensation`` upward so
that future LTD is amplified relative to LTP.  When the watchdog goes
quiet on a group, the bias decays back toward neutral.

The tuner is purely reactive to watchdog alerts -- it does not run on
every simulation step.  Watchdog cadence is ``check_interval`` (default
100 steps), so the tuner sees alerts at most every 100 steps per group.

Patent Claim 1a (STDP) is preserved: ``a_minus`` is multiplied by
``(1 + bias)``, which is mathematically equivalent to a per-group
``a_minus`` override -- the same configurability the claim already
contemplates ("configurable tau_plus, tau_minus, a_plus, a_minus per
synapse group").  No mechanism is disabled; LTP still runs.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable


# Lazy import to avoid circulars; matches watchdog.WatchdogAlert at runtime.
class _AlertProtocol:
    check_name: str
    region: str


class StdpBiasAutoTuner:
    """Per-group adaptive LTD-bias compensator.

    Logic:
      - Maintain a rolling 1000-step window of "weight_drift_up" alert step numbers.
      - On any new alert: append the current step; trim older than ``window_steps``.
      - If the trimmed window has >= ``warning_threshold`` entries, bump bias by
        ``bump_step`` (default 0.05), capped at ``bias_max`` (default 0.5).
      - Quiet-decay: every ``window_steps`` quiet steps (no alerts), the bias
        decays multiplicatively toward 0 by ``decay_factor`` (default 0.95).

    State:
      - ``_warning_history[group]``: deque of step numbers where alerts fired
      - ``_last_quiet_check[group]``: last step we evaluated quiet-decay
      - Bias values themselves live on the SynapseGroup, not here.  This
        class only manipulates them via the ``set_bias`` callback.
    """

    def __init__(
        self,
        window_steps: int = 1000,
        warning_threshold: int = 5,
        bump_step: float = 0.05,
        bias_max: float = 0.5,
        decay_factor: float = 0.95,
        per_group_cap: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 < bump_step <= bias_max:
            raise ValueError("bump_step must be in (0, bias_max]")
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError("decay_factor must be in (0, 1]")
        self._window = int(window_steps)
        self._threshold = int(warning_threshold)
        self._bump = float(bump_step)
        self._max = float(bias_max)
        self._decay = float(decay_factor)
        # Phase 1.4.1 (2026-06-01) hotfix: per-group cap overrides.
        # Motor output groups need a much lower cap (e.g. 0.15 = 1.15x a_minus)
        # because over-amplified LTD on motor pathways drains weights below the
        # functional threshold for physical actuator drive. Non-motor groups can
        # still use the default ``bias_max`` cap. Missing key falls back to
        # ``bias_max``.
        #
        # Phase 2.12c (2026-06-11): per-group caps may now be EITHER lower OR
        # higher than ``bias_max``. Cognitive-loop groups (workspace / meta /
        # association / dg / feature families) need a higher cap (default 1.0
        # via ``NEURO_STDP_BIAS_MAX_COGNITIVE``) because 11 days of training
        # under the 0.5 cap exhausted the bias-tuner authority and cognitive
        # weights crept up to 0.93-1.00. Per-group caps now act as a pure
        # override of ``bias_max`` for their named groups.
        self._per_group_cap: dict[str, float] = {
            k: float(v) for k, v in (per_group_cap or {}).items()
        }
        # Sanity validate: per-group caps must be positive and below an
        # absolute ceiling. The 2.0 ceiling is far above any practical bias
        # (would mean LTD is 3x base a_minus) but prevents typos producing
        # runaway suppression.
        _abs_cap_ceiling = 2.0
        for k, v in self._per_group_cap.items():
            if not 0.0 < v <= _abs_cap_ceiling:
                raise ValueError(
                    f"per_group_cap[{k!r}]={v} must be in (0, {_abs_cap_ceiling}]"
                )

        self._warning_history: dict[str, deque[int]] = {}
        self._last_quiet_check: dict[str, int] = {}
        # Telemetry: count of bumps applied since last drain
        self._bump_count: int = 0
        self._decay_count: int = 0

    def handle_alerts(
        self,
        alerts: Iterable[_AlertProtocol],
        step: int,
        get_bias: Any,
        set_bias: Any,
    ) -> None:
        """Process a watchdog alert batch for one check cycle.

        Args:
            alerts: WatchdogAlert iterable from ``NeuralWatchdog.check``.
            step: current simulation step.
            get_bias: callable ``(group_name) -> float`` returning current bias.
            set_bias: callable ``(group_name, new_bias) -> None`` to write bias.
        """
        # Phase 1.4.1: One-shot snap-down for groups whose persisted bias
        # exceeds the new per-group cap (e.g. motor weights restored from
        # SQLite at 0.500 when their new cap is 0.150). Counted as a decay
        # for telemetry so the operator can see the snap happen.
        for group, cap in self._per_group_cap.items():
            current = float(get_bias(group))
            if current > cap:
                set_bias(group, cap)
                self._decay_count += 1

        # Collect this cycle's drift_up alerts, indexed by group name.
        drift_groups_this_cycle: set[str] = set()
        for alert in alerts:
            if getattr(alert, "check_name", "") == "weight_drift_up":
                group = getattr(alert, "region", "") or ""
                if group:
                    drift_groups_this_cycle.add(group)

        # Record this cycle's drift alerts in each group's history.
        for group in drift_groups_this_cycle:
            hist = self._warning_history.setdefault(group, deque())
            hist.append(step)
            self._trim_window(hist, step)
            if len(hist) >= self._threshold:
                current = float(get_bias(group))
                group_cap = self._per_group_cap.get(group, self._max)
                new_bias = min(current + self._bump, group_cap)
                if new_bias > current:
                    set_bias(group, new_bias)
                    self._bump_count += 1

        # Quiet-decay sweep across ALL groups the tuner has ever observed.
        # Only decay groups whose window has been empty for at least ``window_steps``.
        # Initialize _last_quiet_check on first contact so brand-new groups don't
        # decay immediately based on a step in the distant future.
        for group in list(self._warning_history.keys()):
            hist = self._warning_history[group]
            self._trim_window(hist, step)
            if group not in self._last_quiet_check:
                self._last_quiet_check[group] = step
            if not hist:
                elapsed = step - self._last_quiet_check[group]
                if elapsed >= self._window:
                    current = float(get_bias(group))
                    if current > 0.0:
                        new_bias = max(0.0, current * self._decay)
                        set_bias(group, new_bias)
                        self._decay_count += 1
                    self._last_quiet_check[group] = step
            else:
                # Any non-empty history resets the quiet timer.
                self._last_quiet_check[group] = step

    def _trim_window(self, hist: deque[int], current_step: int) -> None:
        """Drop alert step numbers older than ``window_steps`` from the deque."""
        cutoff = current_step - self._window
        while hist and hist[0] < cutoff:
            hist.popleft()

    def drain_telemetry(self) -> tuple[int, int]:
        """Return (bumps_applied, decays_applied) since last drain and reset."""
        b, d = self._bump_count, self._decay_count
        self._bump_count = 0
        self._decay_count = 0
        return b, d

    def get_warning_count(self, group: str) -> int:
        """Return current windowed warning count for a group (read-only)."""
        hist = self._warning_history.get(group)
        return len(hist) if hist is not None else 0

    def get_state(self) -> dict[str, Any]:
        """Serialize tuner state for persistence.

        The per-group bias values live on SynapseGroup and are persisted there.
        This serializes only the warning history and quiet-check timers so
        post-restart behavior continues smoothly.
        """
        return {
            "warning_history": {
                g: list(h) for g, h in self._warning_history.items()
            },
            "last_quiet_check": dict(self._last_quiet_check),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore tuner state from persistence."""
        hist_state = state.get("warning_history", {}) or {}
        self._warning_history = {
            g: deque(int(s) for s in v) for g, v in hist_state.items()
        }
        self._last_quiet_check = {
            g: int(s) for g, s in (state.get("last_quiet_check", {}) or {}).items()
        }
