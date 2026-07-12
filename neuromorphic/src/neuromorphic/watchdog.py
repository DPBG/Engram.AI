"""Neural Watchdog — monitors brain state for pathological patterns.

Runs periodic checks against the network's internal state (firing rates,
weight distributions, neuromodulator levels) and escalates anomalies
through a 5-level severity system.

Design constraints:
- All checks are READ-ONLY — the watchdog never modifies neural state.
- Responses at Level 2+ publish alerts but do not stop the simulation
  (only Level 4 EMERGENCY triggers a halt request).
- Patent Claim 1e alignment: monitoring IS homeostatic regulation.
- Patent Claim 6 alignment: preserves continual learning (never resets weights).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from neuromorphic.network import NeuromorphicNetwork

logger = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    """Watchdog escalation levels."""

    NOMINAL = 0  # All checks pass
    WARNING = 1  # 1 check failed — log + dashboard alert
    CAUTION = 2  # 2+ checks failed — alert + recommend plasticity reduction
    CRITICAL = 3  # Critical threshold hit — freeze motor output recommendation
    EMERGENCY = 4  # 3+ critical — request simulation halt


@dataclass
class WatchdogAlert:
    """A single watchdog alert."""

    check_name: str
    level: AlertLevel
    message: str
    value: float = 0.0  # The observed value
    threshold: float = 0.0  # The threshold that was exceeded
    region: str = ""  # Which region (if applicable)
    timestamp: float = field(default_factory=time.time)


@dataclass
class WatchdogConfig:
    """Configurable thresholds for watchdog checks."""

    # Check interval (in simulation steps)
    check_interval: int = 100

    # Firing rate caps per region type
    rate_warning: float = 0.40  # 40% → WARNING
    rate_critical: float = 0.60  # 60% → CRITICAL
    motor_rate_warning: float = 0.30  # Motor cortex is more dangerous
    motor_rate_critical: float = 0.50
    meta_rate_warning: float = 0.55  # Meta-controller is a modulator hub, naturally higher
    meta_rate_critical: float = 0.70

    # Dead region detection
    silence_steps: int = 200  # Steps of 0% rate before warning

    # Weight drift detection
    drift_window: int = 5  # Consecutive monotonic checks
    weight_floor_multiplier: float = 2.0  # Alert if mean < 2x w_min

    # Motor spam detection
    motor_spam_steps: int = 10  # Steps of >50% motor rate → CRITICAL
    motor_spam_rate: float = 0.50

    # Sensory starvation (extends existing zero-obs watchdog)
    sensory_critical_steps: int = 1000  # 1000 steps without input → CRITICAL

    # Neuromodulator collapse
    neuromod_floor_steps: int = 500  # Steps at floor → WARNING

    # Prediction error stagnation
    pred_error_stagnation_steps: int = 5000
    pred_error_stagnation_threshold: float = 0.9

    # Weight oscillation detection (rapid up/down changes = instability)
    weight_oscillation_window: int = 6  # checks to analyze for oscillation
    weight_oscillation_threshold: int = 4  # direction reversals within window → WARNING
    weight_oscillation_min_amplitude: float = 0.0  # min weight range in window to warn (0=disabled)

    # Eligibility trace saturation
    elig_saturation_threshold: float = 0.8  # fraction of non-zero entries near max → WARNING


class NeuralWatchdog:
    """Monitors brain internal state for pathological patterns.

    Usage:
        watchdog = NeuralWatchdog(config)
        # In simulation loop, every N steps:
        status = watchdog.check(network, step_count)
        if status.level >= AlertLevel.WARNING:
            publish_alerts(status.alerts)
    """

    def __init__(self, config: WatchdogConfig | None = None):
        self._config = config or WatchdogConfig()

        # Per-region firing rate history (for silence detection)
        self._silence_counters: dict[str, int] = {}

        # Weight drift tracking: per-group list of recent mean weights
        self._weight_history: dict[str, list[float]] = {}

        # Motor spam tracking
        self._motor_spam_counter: int = 0

        # Neuromod floor tracking: per-channel counter
        self._neuromod_floor_counters: dict[str, int] = {}

        # Prediction error history
        self._pred_error_high_counter: int = 0

        # Last check step (avoid redundant checks)
        self._last_check_step: int = -1

    @property
    def check_interval(self) -> int:
        return self._config.check_interval

    def check(
        self,
        network: NeuromorphicNetwork,
        step_count: int,
        sensory_steps_since_input: int = 0,
    ) -> WatchdogStatus:
        """Run all watchdog checks against current network state.

        Args:
            network: The NeuromorphicNetwork instance.
            step_count: Current simulation step.
            sensory_steps_since_input: Steps since last sensory observation.

        Returns:
            WatchdogStatus with overall level and list of alerts.
        """
        if step_count == self._last_check_step:
            return WatchdogStatus(level=AlertLevel.NOMINAL, alerts=[], step=step_count)

        self._last_check_step = step_count
        alerts: list[WatchdogAlert] = []

        # Run all checks
        alerts.extend(self._check_firing_rates(network))
        alerts.extend(self._check_weight_drift(network))
        alerts.extend(self._check_motor_spam(network))
        alerts.extend(self._check_sensory_starvation(sensory_steps_since_input))
        alerts.extend(self._check_neuromodulator_levels(network))
        alerts.extend(self._check_prediction_error(network))
        alerts.extend(self._check_weight_oscillation(network))
        alerts.extend(self._check_eligibility_saturation(network))
        alerts.extend(self._check_cross_modal_health(network))

        # Compute overall level
        level = AlertLevel.NOMINAL
        critical_count = sum(1 for a in alerts if a.level >= AlertLevel.CRITICAL)
        warning_count = sum(1 for a in alerts if a.level >= AlertLevel.WARNING)

        if critical_count >= 3:
            level = AlertLevel.EMERGENCY
        elif critical_count >= 1:
            level = AlertLevel.CRITICAL
        elif warning_count >= 2:
            level = AlertLevel.CAUTION
        elif warning_count >= 1:
            level = AlertLevel.WARNING

        return WatchdogStatus(level=level, alerts=alerts, step=step_count)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_firing_rates(self, network: NeuromorphicNetwork) -> list[WatchdogAlert]:
        """Check per-region firing rates against caps."""
        alerts: list[WatchdogAlert] = []
        cfg = self._config
        rates = network.get_firing_rates()

        for region_name, rate in rates.items():
            if region_name == "motor_cortex":
                warning_thresh = cfg.motor_rate_warning
                critical_thresh = cfg.motor_rate_critical
            elif region_name == "meta_controller":
                warning_thresh = cfg.meta_rate_warning
                critical_thresh = cfg.meta_rate_critical
            else:
                warning_thresh = cfg.rate_warning
                critical_thresh = cfg.rate_critical

            if rate >= critical_thresh:
                alerts.append(
                    WatchdogAlert(
                        check_name="firing_rate_critical",
                        level=AlertLevel.CRITICAL,
                        message=f"Region '{region_name}' firing at {rate:.1%} (critical threshold: {critical_thresh:.0%})",
                        value=rate,
                        threshold=critical_thresh,
                        region=region_name,
                    )
                )
            elif rate >= warning_thresh:
                alerts.append(
                    WatchdogAlert(
                        check_name="firing_rate_warning",
                        level=AlertLevel.WARNING,
                        message=f"Region '{region_name}' firing at {rate:.1%} (warning threshold: {warning_thresh:.0%})",
                        value=rate,
                        threshold=warning_thresh,
                        region=region_name,
                    )
                )

            # Silence detection (use epsilon, not exact float equality)
            if rate < 1e-6:
                self._silence_counters[region_name] = self._silence_counters.get(region_name, 0) + 1
                if self._silence_counters[region_name] >= cfg.silence_steps:
                    alerts.append(
                        WatchdogAlert(
                            check_name="region_silence",
                            level=AlertLevel.WARNING,
                            message=f"Region '{region_name}' has been silent for {self._silence_counters[region_name]} checks",
                            value=0.0,
                            threshold=float(cfg.silence_steps),
                            region=region_name,
                        )
                    )
            else:
                self._silence_counters[region_name] = 0

        return alerts

    def _check_weight_drift(self, network: NeuromorphicNetwork) -> list[WatchdogAlert]:
        """Check for monotonic weight drift (potential saturation)."""
        alerts: list[WatchdogAlert] = []
        cfg = self._config
        w_min = network.config.w_min if hasattr(network.config, "w_min") else 0.01

        for name, syn in network.get_synapse_groups():
            mean_w = float(syn.weights.data.mean())

            history = self._weight_history.setdefault(name, [])
            history.append(mean_w)

            # Keep bounded history
            max_history = cfg.drift_window + 1
            if len(history) > max_history:
                self._weight_history[name] = history[-max_history:]
                history = self._weight_history[name]

            # Check monotonic increase (saturation risk)
            if len(history) >= cfg.drift_window:
                window = history[-cfg.drift_window :]
                if all(window[i] < window[i + 1] for i in range(len(window) - 1)):
                    alerts.append(
                        WatchdogAlert(
                            check_name="weight_drift_up",
                            level=AlertLevel.WARNING,
                            message=f"Synapse group '{name}' mean weight rising monotonically "
                            f"({window[0]:.4f} → {window[-1]:.4f})",
                            value=mean_w,
                            region=name,
                        )
                    )

            # Check weight collapse
            if mean_w < w_min * cfg.weight_floor_multiplier:
                alerts.append(
                    WatchdogAlert(
                        check_name="weight_collapse",
                        level=AlertLevel.WARNING,
                        message=f"Synapse group '{name}' mean weight {mean_w:.4f} "
                        f"approaching floor ({w_min * cfg.weight_floor_multiplier:.4f})",
                        value=mean_w,
                        threshold=w_min * cfg.weight_floor_multiplier,
                        region=name,
                    )
                )

        return alerts

    def _check_motor_spam(self, network: NeuromorphicNetwork) -> list[WatchdogAlert]:
        """Detect pathological motor cortex over-firing."""
        alerts: list[WatchdogAlert] = []
        cfg = self._config
        rates = network.get_firing_rates()
        motor_rate = rates.get("motor_cortex", 0.0)

        if motor_rate >= cfg.motor_spam_rate:
            self._motor_spam_counter += 1
        else:
            self._motor_spam_counter = 0

        if self._motor_spam_counter >= cfg.motor_spam_steps:
            alerts.append(
                WatchdogAlert(
                    check_name="motor_spam",
                    level=AlertLevel.CRITICAL,
                    message=f"Motor cortex firing at {motor_rate:.1%} for "
                    f"{self._motor_spam_counter} consecutive checks — pathological",
                    value=motor_rate,
                    threshold=cfg.motor_spam_rate,
                    region="motor_cortex",
                )
            )

        return alerts

    def _check_sensory_starvation(self, steps_since_input: int) -> list[WatchdogAlert]:
        """Check for extended sensory deprivation."""
        alerts: list[WatchdogAlert] = []
        cfg = self._config

        if steps_since_input >= cfg.sensory_critical_steps:
            alerts.append(
                WatchdogAlert(
                    check_name="sensory_starvation",
                    level=AlertLevel.CRITICAL,
                    message=f"No sensory input for {steps_since_input} steps",
                    value=float(steps_since_input),
                    threshold=float(cfg.sensory_critical_steps),
                )
            )

        return alerts

    def _check_neuromodulator_levels(self, network: NeuromorphicNetwork) -> list[WatchdogAlert]:
        """Check for neuromodulator floor collapse."""
        alerts: list[WatchdogAlert] = []
        cfg = self._config

        if not hasattr(network, "neuromodulation"):
            return alerts

        nm = network.neuromodulation
        channels = {
            "DA": nm.da,
            "ACh": nm.ach,
            "NE": nm.ne,
            "5HT": nm.serotonin,
        }

        # Floor values from neuromodulation config
        floors = {
            "DA": getattr(nm, "_da_floor", 0.0),
            "ACh": getattr(nm, "_ach_floor", 0.0),
            "NE": getattr(nm, "_ne_floor", 0.0),
            "5HT": getattr(nm, "_5ht_floor", 0.0),
        }

        for channel, value in channels.items():
            floor = floors.get(channel, 0.0)
            # Check if value is at or very near the floor
            if floor > 0 and value <= floor * 1.05:
                counter = self._neuromod_floor_counters.get(channel, 0) + 1
                self._neuromod_floor_counters[channel] = counter
                if counter >= cfg.neuromod_floor_steps:
                    alerts.append(
                        WatchdogAlert(
                            check_name="neuromod_floor",
                            level=AlertLevel.WARNING,
                            message=f"Neuromodulator {channel} at floor ({value:.3f}) "
                            f"for {counter} checks — learning may be disabled",
                            value=value,
                            threshold=floor,
                            region=channel,
                        )
                    )
            else:
                self._neuromod_floor_counters[channel] = 0

        return alerts

    def _check_prediction_error(self, network: NeuromorphicNetwork) -> list[WatchdogAlert]:
        """Check for stuck prediction error (brain not learning)."""
        alerts: list[WatchdogAlert] = []
        cfg = self._config

        pred_error = getattr(network, "_last_prediction_error", None)
        if pred_error is None:
            return alerts

        if pred_error >= cfg.pred_error_stagnation_threshold:
            self._pred_error_high_counter += 1
        else:
            self._pred_error_high_counter = 0

        if self._pred_error_high_counter >= (
            cfg.pred_error_stagnation_steps // max(cfg.check_interval, 1)
        ):
            alerts.append(
                WatchdogAlert(
                    check_name="prediction_error_stagnation",
                    level=AlertLevel.WARNING,
                    message=f"Prediction error stuck above {cfg.pred_error_stagnation_threshold:.1f} "
                    f"for ~{self._pred_error_high_counter * cfg.check_interval} steps",
                    value=pred_error,
                    threshold=cfg.pred_error_stagnation_threshold,
                )
            )

        return alerts

    def _check_weight_oscillation(self, network: NeuromorphicNetwork) -> list[WatchdogAlert]:
        """Detect rapid weight oscillation (instability indicator).

        If the mean weight of a synapse group reverses direction frequently
        within a window, this suggests unstable STDP dynamics (e.g., LTP/LTD
        fighting each other due to mismatched rates or excessive plasticity).
        """
        alerts: list[WatchdogAlert] = []
        cfg = self._config
        window = cfg.weight_oscillation_window
        reversal_thresh = cfg.weight_oscillation_threshold

        for name, syn in network.synapses.items():
            if not syn.plastic:
                continue
            history = self._weight_history.get(name, [])
            if len(history) < window:
                continue
            # Count direction reversals in the last `window` entries
            recent = history[-window:]
            reversals = 0
            for i in range(2, len(recent)):
                d1 = recent[i - 1] - recent[i - 2]
                d2 = recent[i] - recent[i - 1]
                if d1 * d2 < 0:  # sign change = direction reversal
                    reversals += 1
            if reversals >= reversal_thresh:
                # Amplitude filter: skip warning if total weight swing is below
                # threshold (normal STDP-homeostasis equilibrium, not instability)
                amplitude = max(recent) - min(recent)
                if amplitude < cfg.weight_oscillation_min_amplitude:
                    continue
                alerts.append(
                    WatchdogAlert(
                        check_name="weight_oscillation",
                        level=AlertLevel.WARNING,
                        message=f"Synapse group '{name}' has {reversals} weight direction "
                        f"reversals in last {window} checks — possible STDP instability",
                        value=float(reversals),
                        threshold=float(reversal_thresh),
                        region=name,
                    )
                )
        return alerts

    def _check_eligibility_saturation(self, network: NeuromorphicNetwork) -> list[WatchdogAlert]:
        """Detect eligibility trace saturation.

        When a large fraction of eligibility trace values are near the maximum
        (the eligibility has saturated), the three-factor learning signal
        loses resolution — all synapses get similar updates regardless of
        their recent activity history.
        """
        alerts: list[WatchdogAlert] = []
        cfg = self._config

        for name, syn in network.synapses.items():
            if not syn.plastic or not hasattr(syn, "eligibility"):
                continue
            elig = getattr(syn, "eligibility", None)
            if elig is None:
                continue
            # Eligibility is always a dense 1D float32 array (synapses.py)
            if len(elig) == 0:
                continue
            # Check fraction of non-zero entries that are near max (>90% of max)
            nnz_data = elig[elig != 0]
            if len(nnz_data) == 0:
                continue
            max_val = np.abs(nnz_data).max()
            if max_val < 1e-8:
                continue
            near_max = np.abs(nnz_data) > 0.9 * max_val
            saturation_frac = near_max.sum() / len(nnz_data)
            if saturation_frac > cfg.elig_saturation_threshold:
                alerts.append(
                    WatchdogAlert(
                        check_name="eligibility_saturation",
                        level=AlertLevel.WARNING,
                        message=f"Synapse group '{name}' eligibility traces {saturation_frac:.0%} "
                        f"near max — three-factor learning resolution degraded",
                        value=float(saturation_frac),
                        threshold=cfg.elig_saturation_threshold,
                        region=name,
                    )
                )
        return alerts

    def _check_cross_modal_health(self, network: NeuromorphicNetwork) -> list[WatchdogAlert]:
        """Detect cross-modal binding degradation.

        If a modality has a large sensory range but zero associated neurons
        in association cortex, binding for that modality has collapsed.
        This catches silent regression (e.g. allocator state loss on restart).
        """
        alerts: list[WatchdogAlert] = []
        probe = getattr(network, "_cross_modal_probe", None)
        if probe is None:
            return alerts
        last_result = getattr(probe, "_last_result", None)
        if last_result is None:
            return alerts
        ranges = getattr(network, "allocator", None)
        if ranges is None:
            return alerts
        alloc_ranges = ranges.current_ranges
        min_neurons = 5000  # only check modalities with meaningful range
        for mod_name, (start, end) in alloc_ranges.items():
            if (end - start) < min_neurons:
                continue
            assoc_key = f"n_{mod_name}_associated"
            n_assoc = last_result.get(assoc_key, -1)
            if n_assoc == 0:
                alerts.append(
                    WatchdogAlert(
                        check_name="cross_modal_binding_lost",
                        level=AlertLevel.WARNING,
                        message=f"Cross-modal binding lost for '{mod_name}': "
                        f"{end - start} sensory neurons but 0 associated neurons",
                        value=0.0,
                        threshold=1.0,
                        region=mod_name,
                    )
                )
        return alerts

    def get_governor_corrections(
        self,
        network: NeuromorphicNetwork,
    ) -> dict[str, float]:
        """Compute inhibitory corrections for over-firing regions.

        When a region fires above its critical threshold, the governor
        returns a negative current multiplier to inject inhibitory current.
        This is homeostatic (Claim 1e) — it regulates activity without
        modifying weights (Claim 6 safe).

        Returns:
            Dict of region_name → inhibitory_gain (0.0 to -1.0).
            Empty if all regions are nominal.
        """
        corrections: dict[str, float] = {}
        cfg = self._config
        rates = network.get_firing_rates()

        # Governor corrections are in mV of inhibitory current, scaled to the
        # neuron threshold gap (~10mV) so they actually affect firing.
        threshold_gap = abs(network.config.lif.threshold - network.config.lif.resting)

        for region_name, rate in rates.items():
            is_motor = region_name == "motor_cortex"
            critical = cfg.motor_rate_critical if is_motor else cfg.rate_critical
            warning = cfg.motor_rate_warning if is_motor else cfg.rate_warning

            if rate >= critical:
                # Strong inhibition: 30-100% of threshold gap
                overshoot = min((rate - critical) / max(1.0 - critical, 0.01), 1.0)
                corrections[region_name] = -threshold_gap * (0.3 + 0.7 * overshoot)
            elif rate >= warning:
                # Mild inhibition: 5-30% of threshold gap
                overshoot = (rate - warning) / max(critical - warning, 0.01)
                corrections[region_name] = -threshold_gap * (0.05 + 0.25 * overshoot)

        return corrections

    def get_state(self) -> dict[str, Any]:
        """Serialize watchdog state for persistence."""
        return {
            "silence_counters": dict(self._silence_counters),
            "weight_history": {k: list(v) for k, v in self._weight_history.items()},
            "motor_spam_counter": self._motor_spam_counter,
            "neuromod_floor_counters": dict(self._neuromod_floor_counters),
            "pred_error_high_counter": self._pred_error_high_counter,
            "last_check_step": self._last_check_step,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore watchdog state from persistence."""
        self._silence_counters = state.get("silence_counters", {})
        self._weight_history = state.get("weight_history", {})
        self._motor_spam_counter = state.get("motor_spam_counter", 0)
        self._neuromod_floor_counters = state.get("neuromod_floor_counters", {})
        self._pred_error_high_counter = state.get("pred_error_high_counter", 0)
        self._last_check_step = state.get("last_check_step", -1)


@dataclass
class WatchdogStatus:
    """Result of a watchdog check cycle."""

    level: AlertLevel
    alerts: list[WatchdogAlert]
    step: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize for NATS publishing."""
        return {
            "level": self.level.name,
            "level_value": int(self.level),
            "alert_count": len(self.alerts),
            "step": self.step,
            "alerts": [
                {
                    "check": a.check_name,
                    "level": a.level.name,
                    "message": a.message,
                    "value": a.value,
                    "threshold": a.threshold,
                    "region": a.region,
                }
                for a in self.alerts
            ],
        }
