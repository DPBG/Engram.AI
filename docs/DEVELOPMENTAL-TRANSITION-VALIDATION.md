# Developmental-Transition Validation Report

This document summarises the validation suite for **Invariant 2** of Engram's
neuromorphic architecture — the requirement that the adolescent developmental
transition is experience-dependent and never hardcoded by step count alone.

Reference: `CLAUDE.md` § 2 Invariant 2, `neuromorphic/tests/test_developmental_transition_suite.py`

---

## What Is Being Tested

Engram's developmental schedule progresses through five phases:

```
infant → toddler → juvenile → [adolescent] → mature
```

The first three phases (infant → toddler → juvenile) are **time-based** —
they advance at configurable step-count boundaries set in `CriticalPeriodConfig`.
This is permitted by Invariant 2.

The **adolescent** transition is different. It is **experience-dependent** and
is gated on three simultaneous learning signals evaluated by
`NeuromodulationSystem._check_adolescent_entry()`:

| Criterion | Signal | Pass condition |
|---|---|---|
| Concept differentiation | `concept_tracker.is_differentiated` | ≥ N distinct concept patterns formed |
| Sensory stability | `_sensory_rate_variance` | firing-rate variance < threshold |
| Feature-STDP decline | `_feature_stdp_current / _feature_stdp_peak` | ratio < `feature_stdp_decline` |

All three criteria must have been satisfied within a rolling `criteria_window`
and must hold simultaneously for `consecutive_checks` consecutive check
intervals after `min_steps` have elapsed.

---

## Input Regimes Exercised

### Regime A — Rich experience

All three criteria are satisfied promptly after `juvenile_end`:
- Distinct concept patterns injected at step 60
- Sensory rate variance = 0.01 (< 0.1 threshold)
- Feature-STDP ratio = 0.1/1.0 (< 0.5 threshold)

**Expected behaviour:** adolescent entered at the first eligible check pair
(steps 100 and 110 in the test config).

### Regime B — Degenerate / null input

No criteria are ever satisfied:
- No distinct concept patterns recorded
- Sensory rate variance = 0.50 (> threshold)
- Feature-STDP ratio = 0.9/1.0 (not declined)

**Expected behaviour:** adolescent is **never** entered. The system progresses
juvenile → mature directly, bypassing adolescence.

### Regime C — Partial experience (2 of 3 criteria)

Two criteria are met; one is permanently absent:
- Distinct concept patterns injected (criterion 1 ✓)
- Sensory rate variance = 0.01 (criterion 2 ✓)
- Feature-STDP ratio = 0.9/1.0 — NOT declined (criterion 3 ✗)

**Expected behaviour:** adolescent is **not** entered, even though two criteria
are satisfied, because the staggered-criteria window logic requires all three to
converge.

### Regime D — Delayed rich (timing comparison)

Identical to Regime A except the STDP decline signal arrives 40 steps later
(step 140 instead of step 60).

**Expected behaviour:** adolescent entered later than Regime A (step ~150 vs
step ~110 in the test config), proving that timing tracks experience rather than
being fixed at a step-count boundary.

---

## Assertions

### Positive (entry happens when it should)

- Regime A: `entry_step is not None`
- Regime A: `entry_step >= juvenile_end` and `entry_step >= min_steps`
- Regime D: `entry_step is not None` (all criteria eventually met)

### Negative (entry blocked when criteria absent)

- Regime B: `entry_step is None` (degenerate input never triggers adolescence)
- Regime C: `entry_step is None` (STDP criterion missing blocks entry)
- Each single criterion alone: `entry_step is None` (7 single/pair combinations)

### Experience-dependence (core Invariant 2 proof)

- `entry_step_D > entry_step_A`: delayed STDP → later entry
- `entry_step_A ≠ juvenile_end`: entry is not fixed at the juvenile boundary
- `entry_step_A ≠ entry_step_D`: same architecture, different experience → different timing
- After identical step counts, Regime A is in adolescent/mature (via adolescence),
  Regime B is in mature (directly, never adolescent)

---

## Test File

```
neuromorphic/tests/test_developmental_transition_suite.py
```

26 tests across 6 classes:

| Class | Focus |
|---|---|
| `TestTimeBasedEarlyPhases` | Confirms infant/toddler/juvenile remain step-based |
| `TestRegimeARichExperience` | Positive case: rich input triggers adolescence |
| `TestRegimeBDegenerateInput` | Negative case: null input skips adolescence |
| `TestRegimeCPartialExperience` | Negative case: 2/3 criteria insufficient |
| `TestTransitionTimingVariesWithExperience` | Timing comparison (A vs D) |
| `TestAllThreeCriteriaRequired` | All 7 single/pair combinations are insufficient |
| `TestInvariant2Regression` | Summary regression: catches any future violation |

Run with:
```bash
cd neuromorphic && uv run --extra dev python -m pytest tests/test_developmental_transition_suite.py -v -p no:anchorpy
```
