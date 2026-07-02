"""Neuromodulation system — 4 channels with critical period developmental schedule.

Neuromodulators are global signals read from the MetaControllerRegion's
sub-population firing rates. They gate plasticity (three-factor learning)
and modulate network dynamics.

Channels:
  DA  — dopamine: per-region plasticity multiplier [0, 5]
  ACh — acetylcholine: feedforward vs recurrent balance [0, 3]
  NE  — norepinephrine: global gain / arousal [0.5, 3]
  5-HT — serotonin: consolidation brake [0, 1]

Critical periods shift the baseline of each channel over developmental time.
The adolescent phase is entered dynamically (experience-dependent, not time-based).
"""

from __future__ import annotations

import logging

import numpy as np

from neuromorphic.config import AdolescentEntryConfig, NeuromorphicConfig

logger = logging.getLogger(__name__)


class ConceptDifferentiationTracker:
    """Track distinct concept patterns to detect when concepts have differentiated.

    Maintains a rolling buffer of association cortex firing patterns and counts
    the number of distinct clusters (patterns with cosine similarity < threshold).
    """

    def __init__(self, config: AdolescentEntryConfig, n_sample: int = 200):
        self._similarity_threshold = config.concept_similarity_threshold
        self._min_patterns = config.min_concept_patterns
        self._n_sample = n_sample  # neurons to sample (for efficiency)
        self._patterns: list[np.ndarray] = []
        self._max_stored = config.max_stored_patterns

    def record_pattern(self, firing_rates: np.ndarray) -> None:
        """Record a snapshot of firing rates (subsampled for efficiency)."""
        if len(firing_rates) > self._n_sample:
            # Deterministic subsample (evenly spaced)
            step = len(firing_rates) // self._n_sample
            sample = firing_rates[::step][: self._n_sample]
        else:
            sample = firing_rates.copy()
        norm = np.linalg.norm(sample)
        if norm > 1e-8:
            sample = sample / norm
            self._patterns.append(sample)
            if len(self._patterns) > self._max_stored:
                self._patterns = self._patterns[-self._max_stored :]

    def count_distinct(self) -> int:
        """Count distinct clusters via greedy cosine similarity."""
        if len(self._patterns) < 2:
            return len(self._patterns)

        centroids: list[np.ndarray] = [self._patterns[0]]
        for pat in self._patterns[1:]:
            is_new = True
            for c in centroids:
                sim = float(np.dot(pat, c))
                if sim > (1.0 - self._similarity_threshold):
                    is_new = False
                    break
            if is_new:
                centroids.append(pat)
        return len(centroids)

    @property
    def is_differentiated(self) -> bool:
        return self.count_distinct() >= self._min_patterns

    def get_state(self) -> dict:
        return {
            "n_patterns": len(self._patterns),
            "n_distinct": self.count_distinct(),
            # D3: Persist patterns so adolescent readiness survives checkpoint resume
            "patterns": [p.tolist() for p in self._patterns],
        }

    def set_state(self, state: dict) -> None:
        # D3: Restore patterns from persisted state
        if "patterns" in state:
            self._patterns = [np.array(p, dtype=np.float32) for p in state["patterns"]]


class NeuromodulationSystem:
    """Reads neuromodulatory outputs and applies developmental schedule.

    Phase progression: infant → toddler → juvenile → [adolescent] → mature
    The adolescent phase is entered dynamically when experience-dependent
    criteria are met (concept differentiation, sensory stability, etc.).
    """

    def __init__(self, config: NeuromorphicConfig):
        self._cp = config.critical_period
        self._adol_entry = config.adolescent_entry
        self._myelination_target = config.myelination.target_fraction

        # A3: Initialize from infant baselines (not hardcoded) — prevents first-step discontinuity
        self.da: float = self._cp.infant_da
        self.ach: float = self._cp.infant_ach
        self.ne: float = self._cp.infant_ne
        self.serotonin: float = self._cp.infant_serotonin

        # Developmental baselines (shift with critical periods)
        self._baseline_da: float = self._cp.infant_da
        self._baseline_ach: float = self._cp.infant_ach
        self._baseline_ne: float = self._cp.infant_ne
        self._baseline_serotonin: float = self._cp.infant_serotonin

        self._current_phase: str = "infant"

        # Adolescent phase tracking
        self._adolescent_active: bool = False
        self._adolescent_start_step: int = 0
        self._adolescent_entry_checks: int = 0  # consecutive checks meeting criteria
        self._concept_tracker = ConceptDifferentiationTracker(self._adol_entry)

        # C4: Staggered entry — track when each criterion was last satisfied
        self._criterion_last_met: dict[str, int] = {
            "concept_differentiation": 0,
            "sensory_stability": 0,
            "feature_stdp_decline": 0,
        }

        # External signals for entry criteria (set by network each step)
        self._sensory_rate_variance: float = 1.0
        self._feature_stdp_peak: float = 1.0
        self._feature_stdp_current: float = 1.0
        self._myelination_fraction: float = 0.0

    @property
    def concept_tracker(self) -> ConceptDifferentiationTracker:
        return self._concept_tracker

    def update_external_signals(
        self,
        sensory_rate_variance: float = 1.0,
        feature_stdp_current: float = 1.0,
        feature_stdp_peak: float | None = None,
        myelination_fraction: float = 0.0,
    ) -> None:
        """Update signals from network for adolescent entry/exit evaluation."""
        self._sensory_rate_variance = sensory_rate_variance
        self._feature_stdp_current = feature_stdp_current
        if feature_stdp_peak is not None:
            self._feature_stdp_peak = max(self._feature_stdp_peak, feature_stdp_peak)
        self._myelination_fraction = myelination_fraction

    def update(self, step: int, meta_outputs: dict[str, float] | None = None) -> None:
        """Update neuromodulator levels from meta-controller + developmental schedule."""
        # Update critical period baselines
        self._update_critical_period(step)

        # Check adolescent entry/exit (at configured intervals, after juvenile)
        if step >= self._cp.juvenile_end:
            check_interval = self._adol_entry.check_interval
            if check_interval > 0 and step % check_interval == 0:
                if not self._adolescent_active:
                    self._check_adolescent_entry(step)
                else:
                    self._check_adolescent_exit(step)

        if meta_outputs is not None:
            self.da = self._baseline_da * meta_outputs.get("da", 1.0)
            self.ach = self._baseline_ach * meta_outputs.get("ach", 1.0)
            self.ne = self._baseline_ne * meta_outputs.get("ne", 1.0)
            self.serotonin = self._baseline_serotonin * meta_outputs.get("serotonin", 1.0)
        else:
            self.da = self._baseline_da
            self.ach = self._baseline_ach
            self.ne = self._baseline_ne
            self.serotonin = self._baseline_serotonin

    def _check_adolescent_entry(self, step: int) -> None:
        """Check if all experience-dependent entry criteria are met.

        C4: Criteria are tracked independently with staggered satisfaction.
        Each criterion records the step when it was last met.
        Entry occurs when all criteria have been met within `criteria_window` steps
        AND the current check finds all criteria met simultaneously for N consecutive checks.
        """
        cfg = self._adol_entry

        if step < cfg.min_steps:
            return

        # Check each criterion independently and record when last satisfied
        concept_met = self._concept_tracker.is_differentiated
        if concept_met:
            self._criterion_last_met["concept_differentiation"] = step

        sensory_met = self._sensory_rate_variance < cfg.sensory_stability_threshold
        if sensory_met:
            self._criterion_last_met["sensory_stability"] = step

        stdp_met = (
            self._feature_stdp_peak < 1e-8
            or self._feature_stdp_current < cfg.feature_stdp_decline * self._feature_stdp_peak
        )
        if stdp_met:
            self._criterion_last_met["feature_stdp_decline"] = step

        # Check if all criteria were satisfied within the rolling window
        all_within_window = all(
            (step - last_met) <= cfg.criteria_window
            for last_met in self._criterion_last_met.values()
        )

        if all_within_window:
            self._adolescent_entry_checks += 1
            if self._adolescent_entry_checks >= cfg.consecutive_checks:
                self._enter_adolescent(step)
        else:
            self._adolescent_entry_checks = 0

        # Log criteria progress every check so operators can monitor approach
        n_concepts = self._concept_tracker.count_distinct()
        stdp_ratio = (
            self._feature_stdp_current / self._feature_stdp_peak
            if self._feature_stdp_peak > 1e-8
            else 0.0
        )
        logger.info(
            "Adolescent entry check at step %s: "
            "concepts=%d/%d %s | "
            "sensory_var=%.4f/%.4f %s | "
            "stdp_ratio=%.3f/%.1f %s | "
            "consecutive=%d/%d | "
            "all_in_window=%s",
            f"{step:,}",
            n_concepts,
            cfg.min_concept_patterns,
            "MET" if concept_met else "---",
            self._sensory_rate_variance,
            cfg.sensory_stability_threshold,
            "MET" if sensory_met else "---",
            stdp_ratio,
            cfg.feature_stdp_decline,
            "MET" if stdp_met else "---",
            self._adolescent_entry_checks,
            cfg.consecutive_checks,
            "YES" if all_within_window else "NO",
        )

    def _enter_adolescent(self, step: int) -> None:
        """Transition to adolescent phase."""
        self._adolescent_active = True
        self._adolescent_start_step = step
        self._current_phase = "adolescent"
        logger.info(
            f"ADOLESCENT PHASE ENTERED at step {step:,} — "
            f"concepts={self._concept_tracker.count_distinct()}, "
            f"sensory_var={self._sensory_rate_variance:.4f}"
        )

    def _check_adolescent_exit(self, step: int) -> None:
        """Check if adolescent phase should end."""
        cfg = self._adol_entry
        duration = step - self._adolescent_start_step

        # Exit conditions: max_duration always exits; myelination only after min_duration
        exit_reason = None
        if duration >= cfg.max_duration:
            exit_reason = f"max_duration ({duration:,} steps)"
        elif (
            duration >= cfg.min_duration and self._myelination_fraction >= self._myelination_target
        ):
            exit_reason = (
                f"myelination target ({self._myelination_fraction:.1%}) after {duration:,} steps"
            )

        if exit_reason:
            self._adolescent_active = False
            self._current_phase = "mature"
            logger.info(
                f"ADOLESCENT PHASE ENDED at step {step:,} — reason: {exit_reason}, "
                f"duration={duration:,} steps"
            )

    def _update_critical_period(self, step: int) -> None:
        """Shift neuromodulatory baselines according to developmental phase."""
        cp = self._cp

        if step < cp.infant_end:
            phase = "infant"
            baselines = (cp.infant_da, cp.infant_ach, cp.infant_ne, cp.infant_serotonin)
        elif step < cp.toddler_end:
            phase = "toddler"
            t = (step - cp.infant_end) / (cp.toddler_end - cp.infant_end)
            baselines = (
                cp.infant_da + t * (cp.toddler_da - cp.infant_da),
                cp.infant_ach + t * (cp.toddler_ach - cp.infant_ach),
                cp.infant_ne + t * (cp.toddler_ne - cp.infant_ne),
                cp.infant_serotonin + t * (cp.toddler_serotonin - cp.infant_serotonin),
            )
        elif step < cp.juvenile_end:
            phase = "juvenile"
            t = (step - cp.toddler_end) / (cp.juvenile_end - cp.toddler_end)
            baselines = (
                cp.toddler_da + t * (cp.juvenile_da - cp.toddler_da),
                cp.toddler_ach + t * (cp.juvenile_ach - cp.toddler_ach),
                cp.toddler_ne + t * (cp.juvenile_ne - cp.toddler_ne),
                cp.toddler_serotonin + t * (cp.juvenile_serotonin - cp.toddler_serotonin),
            )
        elif self._adolescent_active:
            phase = "adolescent"
            baselines = (
                cp.adolescent_da,
                cp.adolescent_ach,
                cp.adolescent_ne,
                cp.adolescent_serotonin,
            )
        else:
            phase = "mature"
            baselines = (cp.mature_da, cp.mature_ach, cp.mature_ne, cp.mature_serotonin)

        self._baseline_da, self._baseline_ach, self._baseline_ne, self._baseline_serotonin = (
            baselines
        )
        # Only update phase from time-based transitions; adolescent entry/exit handled separately
        if not self._adolescent_active:
            self._current_phase = phase
        # But keep "adolescent" if active, even if time-based would say "mature"

    @property
    def phase(self) -> str:
        """Current developmental phase name."""
        return self._current_phase

    @property
    def is_adolescent(self) -> bool:
        """Whether the brain is currently in adolescent phase."""
        return self._adolescent_active

    @property
    def plasticity_multiplier(self) -> float:
        """Combined plasticity multiplier from DA and 5-HT.

        DA amplifies learning, 5-HT suppresses it (consolidation).
        Clamped to [0, inf) — negative plasticity is not meaningful.
        """
        return max(0.0, self.da * (1.0 - self.serotonin))

    def get_state(self) -> dict:
        return {
            "da": self.da,
            "ach": self.ach,
            "ne": self.ne,
            "serotonin": self.serotonin,
            "baseline_da": self._baseline_da,
            "baseline_ach": self._baseline_ach,
            "baseline_ne": self._baseline_ne,
            "baseline_serotonin": self._baseline_serotonin,
            "phase": self._current_phase,
            "adolescent_active": self._adolescent_active,
            "adolescent_start_step": self._adolescent_start_step,
            "adolescent_entry_checks": self._adolescent_entry_checks,
            "feature_stdp_peak": self._feature_stdp_peak,
            # D3: concept tracker patterns
            "concept_tracker": self._concept_tracker.get_state(),
            # C4: staggered criterion timestamps
            "criterion_last_met": dict(self._criterion_last_met),
        }

    def set_state(self, state: dict) -> None:
        self.da = state.get("da", self._cp.infant_da)
        self.ach = state.get("ach", self._cp.infant_ach)
        self.ne = state.get("ne", self._cp.infant_ne)
        self.serotonin = state.get("serotonin", self._cp.infant_serotonin)
        self._baseline_da = state.get("baseline_da", self._cp.infant_da)
        self._baseline_ach = state.get("baseline_ach", self._cp.infant_ach)
        self._baseline_ne = state.get("baseline_ne", self._cp.infant_ne)
        self._baseline_serotonin = state.get("baseline_serotonin", self._cp.infant_serotonin)
        self._current_phase = state.get("phase", "mature")
        self._adolescent_active = state.get("adolescent_active", False)
        self._adolescent_start_step = state.get("adolescent_start_step", 0)
        self._adolescent_entry_checks = state.get("adolescent_entry_checks", 0)
        self._feature_stdp_peak = state.get("feature_stdp_peak", 1.0)
        # D3: restore concept tracker patterns
        if "concept_tracker" in state:
            self._concept_tracker.set_state(state["concept_tracker"])
        # C4: restore criterion timestamps
        if "criterion_last_met" in state:
            self._criterion_last_met.update(state["criterion_last_met"])
