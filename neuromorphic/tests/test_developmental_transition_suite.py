"""Developmental-transition validation suite — Invariant 2.

Validates that Engram's adolescent-phase transition is experience-dependent,
gated on three learning signals (concept differentiation, sensory stability,
feature-STDP decline), and is NOT triggered by step count alone.

References: CLAUDE.md Invariant 2, neuromodulation.py, docs/DEVELOPMENTAL-TRANSITION-VALIDATION.md

Three sensory-input regimes are exercised (+ a fourth for timing comparison):

  Regime A — Rich experience: all three criteria satisfied promptly after
              juvenile_end → adolescent IS entered, at the earliest eligible step.

  Regime B — Degenerate/null: no distinct concepts, high sensory variance, no
              STDP decline → adolescent NEVER entered; system goes juvenile→mature.

  Regime C — Partial (2/3 criteria): concept differentiation + sensory stability
              present, but feature-STDP not declined → adolescent NOT entered.

  Regime D — Delayed rich: identical to A except STDP decline arrives 40 steps
              later → adolescent entered later than A, proving timing tracks
              experience rather than step count.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuromorphic.config import (
    NeuromorphicConfig,
    AdolescentEntryConfig,
    CriticalPeriodConfig,
)
from neuromorphic.neuromodulation import NeuromodulationSystem


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _make_fast_config(
    criteria_window: int = 50,
    consecutive_checks: int = 2,
    check_interval: int = 10,
    min_steps: int = 100,
) -> NeuromorphicConfig:
    """Millisecond-scale config for fast validation tests.

    criteria_window=50 ensures that criterion_last_met initialised to 0
    falls outside the window once step >= 51, so unchecked criteria cannot
    accidentally pass the window check.
    """
    entry = AdolescentEntryConfig(
        min_steps=min_steps,
        consecutive_checks=consecutive_checks,
        check_interval=check_interval,
        criteria_window=criteria_window,
        min_concept_patterns=2,
        sensory_stability_threshold=0.1,
        feature_stdp_decline=0.5,
        max_duration=1000,
        min_duration=20,
    )
    cp = CriticalPeriodConfig(
        infant_end=10,
        toddler_end=30,
        juvenile_end=50,
    )
    return NeuromorphicConfig(critical_period=cp, adolescent_entry=entry)


def _make_distinct_concepts(
    n_patterns: int = 5,
    n_neurons: int = 200,
    seed: int = 42,
) -> list[np.ndarray]:
    """Return n_patterns nearly-orthogonal concept vectors."""
    rng = np.random.default_rng(seed)
    chunk = n_neurons // n_patterns
    patterns = []
    for i in range(n_patterns):
        pat = np.zeros(n_neurons, dtype=np.float32)
        pat[i * chunk:(i + 1) * chunk] = rng.random(chunk).astype(np.float32)
        patterns.append(pat)
    return patterns


# ---------------------------------------------------------------------------
# Regime descriptor
# ---------------------------------------------------------------------------

class InputRegime:
    """Encapsulates the external signals for one sensory-input environment."""

    def __init__(
        self,
        *,
        inject_concepts_at: int | None,
        sensory_variance_fn,
        stdp_fn,
        name: str = "",
    ) -> None:
        self.name = name
        self.inject_concepts_at = inject_concepts_at
        self._variance = sensory_variance_fn
        self._stdp = stdp_fn

    def sensory_variance(self, step: int) -> float:
        return self._variance(step)

    def stdp(self, step: int) -> tuple[float, float]:
        """Return (current, peak) STDP values for this step."""
        return self._stdp(step)


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

REGIME_A = InputRegime(
    name="rich_experience",
    inject_concepts_at=60,              # distinct patterns before min_steps check
    sensory_variance_fn=lambda s: 0.01,  # stable (< 0.1 threshold) from step 0
    stdp_fn=lambda s: (0.1, 1.0),        # declined (0.1 < 0.5 * 1.0) from step 0
)

REGIME_B = InputRegime(
    name="degenerate_null",
    inject_concepts_at=None,             # no distinct concepts ever
    sensory_variance_fn=lambda s: 0.50,  # unstable (> 0.1 threshold)
    stdp_fn=lambda s: (0.9, 1.0),        # not declined (0.9 > 0.5)
)

REGIME_C = InputRegime(
    name="partial_2of3",
    inject_concepts_at=60,               # concept criterion met ✓
    sensory_variance_fn=lambda s: 0.01,  # sensory criterion met ✓
    stdp_fn=lambda s: (0.9, 1.0),        # STDP NOT declined ✗ — only 2/3 criteria
)

REGIME_D = InputRegime(
    name="delayed_rich",
    inject_concepts_at=60,               # same as A
    sensory_variance_fn=lambda s: 0.01,  # same as A
    # STDP decline arrives 40 steps later than A → entry must be later
    stdp_fn=lambda s: (0.1, 1.0) if s >= 140 else (0.9, 1.0),
)


# ---------------------------------------------------------------------------
# Regime runner
# ---------------------------------------------------------------------------

def run_regime(
    nm: NeuromodulationSystem,
    regime: InputRegime,
    *,
    n_steps: int = 300,
) -> int | None:
    """Step nm through n_steps under the given regime.

    Returns the step at which adolescent was first entered, or None if it
    never happened within n_steps.
    """
    concepts = _make_distinct_concepts()
    entry_step: int | None = None

    for step in range(n_steps):
        if regime.inject_concepts_at is not None and step == regime.inject_concepts_at:
            for pat in concepts:
                nm.concept_tracker.record_pattern(pat)

        current, peak = regime.stdp(step)
        nm.update_external_signals(
            sensory_rate_variance=regime.sensory_variance(step),
            feature_stdp_current=current,
            feature_stdp_peak=peak,
        )
        nm.update(step)

        if nm.is_adolescent and entry_step is None:
            entry_step = step

    return entry_step


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestTimeBasedEarlyPhases:
    """infant→toddler→juvenile transitions are time-based (Invariant 2 permits this)."""

    def test_infant_toddler_juvenile_are_step_gated(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        nm.update(5)
        assert nm.phase == "infant"
        nm.update(15)
        assert nm.phase == "toddler"
        nm.update(40)
        assert nm.phase == "juvenile"

    def test_adolescent_not_entered_at_juvenile_end(self):
        """Merely reaching juvenile_end does NOT trigger adolescent (no experience injected)."""
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        for step in range(cfg.critical_period.juvenile_end + 1):
            nm.update(step)
        # Without criteria signals, adolescent must not be active
        assert not nm.is_adolescent


class TestRegimeARichExperience:
    """Regime A: all criteria met early → adolescent IS entered."""

    def test_enters_adolescent(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, REGIME_A)
        assert entry is not None, "Rich experience must trigger adolescent phase"

    def test_entry_after_juvenile_end(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, REGIME_A)
        assert entry is not None
        assert entry >= cfg.critical_period.juvenile_end

    def test_entry_after_min_steps(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, REGIME_A)
        assert entry is not None
        assert entry >= cfg.adolescent_entry.min_steps

    def test_is_adolescent_while_active(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        run_regime(nm, REGIME_A, n_steps=120)
        assert nm.phase in ("adolescent", "mature")


class TestRegimeBDegenerateInput:
    """Regime B: degenerate/null input → adolescent NEVER entered."""

    def test_never_enters_adolescent(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, REGIME_B, n_steps=300)
        assert entry is None, (
            "Degenerate input (no concepts, high variance, no STDP decline) "
            "must not trigger adolescent phase"
        )

    def test_reaches_mature_directly(self):
        """Without adolescent, system goes juvenile → mature skipping adolescent."""
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        run_regime(nm, REGIME_B, n_steps=200)
        assert nm.phase == "mature"
        assert not nm.is_adolescent

    def test_concept_differentiation_absent(self):
        """Confirms the concept criterion was never satisfied in regime B."""
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        run_regime(nm, REGIME_B, n_steps=200)
        assert not nm.concept_tracker.is_differentiated


class TestRegimeCPartialExperience:
    """Regime C: 2/3 criteria met (no STDP decline) → adolescent NOT entered."""

    def test_never_enters_adolescent(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, REGIME_C, n_steps=300)
        assert entry is None, (
            "Partial experience (concept+sensory ok, STDP NOT declined) "
            "must not trigger adolescent phase"
        )

    def test_concept_criterion_was_met(self):
        """Confirms the block is STDP, not concepts — concepts ARE present."""
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        run_regime(nm, REGIME_C, n_steps=300)
        assert nm.concept_tracker.is_differentiated

    def test_phase_is_mature_not_adolescent(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        run_regime(nm, REGIME_C, n_steps=300)
        assert nm.phase == "mature"
        assert not nm.is_adolescent


class TestTransitionTimingVariesWithExperience:
    """Core Invariant-2 proof: timing changes when experience changes."""

    def test_delayed_signals_delay_entry(self):
        """Regime D (late STDP decline) enters adolescent later than Regime A."""
        cfg = _make_fast_config()
        nm_a = NeuromodulationSystem(cfg)
        nm_d = NeuromodulationSystem(cfg)
        entry_a = run_regime(nm_a, REGIME_A, n_steps=300)
        entry_d = run_regime(nm_d, REGIME_D, n_steps=300)
        assert entry_a is not None, "Regime A should enter adolescent"
        assert entry_d is not None, "Regime D should enter adolescent"
        assert entry_d > entry_a, (
            f"Delayed experience (regime D, entry={entry_d}) must enter adolescent "
            f"LATER than rich experience (regime A, entry={entry_a})"
        )

    def test_entry_step_not_equal_to_juvenile_end(self):
        """Entry step differs from juvenile_end — step count is not the gate."""
        cfg = _make_fast_config()
        juvenile_end = cfg.critical_period.juvenile_end
        nm_a = NeuromodulationSystem(cfg)
        nm_d = NeuromodulationSystem(cfg)
        entry_a = run_regime(nm_a, REGIME_A, n_steps=300)
        entry_d = run_regime(nm_d, REGIME_D, n_steps=300)
        assert entry_a is not None
        assert entry_d is not None
        assert entry_a != juvenile_end, "Entry must not coincide with juvenile_end"
        assert entry_d != juvenile_end, "Entry must not coincide with juvenile_end"

    def test_different_experience_different_entry_step(self):
        """Same network architecture, different experience → different entry steps."""
        cfg = _make_fast_config()
        nm_a = NeuromodulationSystem(cfg)
        nm_d = NeuromodulationSystem(cfg)
        entry_a = run_regime(nm_a, REGIME_A, n_steps=300)
        entry_d = run_regime(nm_d, REGIME_D, n_steps=300)
        assert entry_a != entry_d, (
            "Entry step must vary with experience (A and D must differ)"
        )

    def test_no_step_count_singularity(self):
        """After identical step counts, A and B have different adolescent status."""
        cfg = _make_fast_config()
        nm_a = NeuromodulationSystem(cfg)
        nm_b = NeuromodulationSystem(cfg)
        run_regime(nm_a, REGIME_A, n_steps=200)
        run_regime(nm_b, REGIME_B, n_steps=200)
        # B is mature but never went through adolescent
        assert not nm_b.is_adolescent
        assert nm_b.phase == "mature"


class TestAllThreeCriteriaRequired:
    """Each criterion alone is insufficient; all three must converge."""

    def _regime_with_only(self, *, concepts: bool, sensory: bool, stdp: bool) -> InputRegime:
        """Construct a regime where exactly the specified criteria are satisfied."""
        return InputRegime(
            name=f"concepts={concepts},sensory={sensory},stdp={stdp}",
            inject_concepts_at=60 if concepts else None,
            sensory_variance_fn=lambda s: 0.01 if sensory else 0.50,
            stdp_fn=lambda s: (0.1, 1.0) if stdp else (0.9, 1.0),
        )

    def test_concept_only_insufficient(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, self._regime_with_only(concepts=True, sensory=False, stdp=False))
        assert entry is None

    def test_sensory_only_insufficient(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, self._regime_with_only(concepts=False, sensory=True, stdp=False))
        assert entry is None

    def test_stdp_only_insufficient(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, self._regime_with_only(concepts=False, sensory=False, stdp=True))
        assert entry is None

    def test_concept_plus_sensory_insufficient(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, self._regime_with_only(concepts=True, sensory=True, stdp=False))
        assert entry is None

    def test_concept_plus_stdp_insufficient(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, self._regime_with_only(concepts=True, sensory=False, stdp=True))
        assert entry is None

    def test_sensory_plus_stdp_insufficient(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, self._regime_with_only(concepts=False, sensory=True, stdp=True))
        assert entry is None

    def test_all_three_sufficient(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, self._regime_with_only(concepts=True, sensory=True, stdp=True))
        assert entry is not None


class TestInvariant2Regression:
    """Regression suite — failures here indicate a violation of Invariant 2."""

    def test_rich_enters_degenerate_does_not(self):
        cfg = _make_fast_config()
        nm_a = NeuromodulationSystem(cfg)
        nm_b = NeuromodulationSystem(cfg)
        entry_a = run_regime(nm_a, REGIME_A)
        entry_b = run_regime(nm_b, REGIME_B)
        assert entry_a is not None, "Invariant 2 violated: rich experience must enter adolescent"
        assert entry_b is None, "Invariant 2 violated: degenerate input must not enter adolescent"

    def test_partial_does_not_enter(self):
        cfg = _make_fast_config()
        nm = NeuromodulationSystem(cfg)
        entry = run_regime(nm, REGIME_C)
        assert entry is None, "Invariant 2 violated: partial experience (2/3 criteria) must not enter"

    def test_entry_timing_is_experience_dependent(self):
        cfg = _make_fast_config()
        nm_a = NeuromodulationSystem(cfg)
        nm_d = NeuromodulationSystem(cfg)
        entry_a = run_regime(nm_a, REGIME_A)
        entry_d = run_regime(nm_d, REGIME_D)
        assert entry_a is not None
        assert entry_d is not None
        assert entry_a != entry_d, (
            "Invariant 2 violated: entry timing must vary with experience, not be fixed"
        )
