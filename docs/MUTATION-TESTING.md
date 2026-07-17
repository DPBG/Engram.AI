# Mutation Testing — Kernel Risk-Clamping Logic

> Scope: `kernel/src/kernel/evaluator.py`'s risk-clamping logic — the code
> that turns a Safety Supervisor `RiskAnalysis` (or its absence) into the
> `risk_score` a `KernelDecision` is actually gated on.
> Tracking issue: [#211](https://github.com/DPBG/Engram.AI/issues/211) (M1.18).

## Why mutation testing, not just more regression tests

PR #122 fixed a fail-open bug: a missing risk analysis was silently treated
as zero risk, so the Kernel could **ALLOW** an unsafe proposal instead of
denying it. Line coverage was never the gap — every line involved was already
executed by existing tests. The gap was that no test *asserted the exact
value* the risk-clamping arithmetic produced, so a subtly wrong clamp (or, as
this issue's own audit found, a wrong argument order) could silently regress
without any test noticing.

Mutation testing finds exactly that class of gap. [mutmut](https://mutmut.readthedocs.io/)
rewrites `evaluator.py` line-by-line (flips comparisons, swaps constants,
changes dict keys, etc.), reruns `tests/test_evaluator.py` against each
mutant, and reports which mutants **survive** — i.e. which code changes no
test would have caught. This audit's own investigation of `_risk_from_analysis()`
surfaced a live instance of the PR #122 bug class before any test was added:

```python
# max(0.0, min(risk_analysis.risk_score, 1.0)) with risk_score = NaN:
min(float("nan"), 1.0)   # -> nan  (NaN comparisons are always False)
max(0.0, nan)            # -> 0.0 (same reason) — silently "zero risk"
```

`kernel/service.py`'s `_get_risk_analysis()` already rejects a non-finite
`risk_score` from the wire before constructing a `RiskAnalysis` — but
`_risk_from_analysis()` is public API reachable by any caller that builds a
`RiskAnalysis` directly, and it had no defense of its own. Fixed by adding an
explicit `math.isfinite()` check (see the function's docstring for detail).

## Running it

```bash
cd kernel
uv run --extra dev mutmut run      # ~3-6 minutes; regenerates kernel/mutants/
uv run --extra dev mutmut results  # lists survived / no-tests mutants
uv run --extra dev mutmut show <mutant-id>   # view a specific mutant's diff
```

Configuration lives in `[tool.mutmut]` in `kernel/pyproject.toml`:

- `source_paths = ["src"]`, `only_mutate = ["src/kernel/evaluator.py"]` —
  scoped to this one file, per the issue.
- `pytest_add_cli_args_test_selection = ["tests/test_evaluator.py"]` — only
  this file's ~0.8s suite reruns per mutant (not the whole kernel suite),
  which is what makes a several-hundred-mutant run finish in minutes instead
  of hours.

`kernel/mutants/` and `.mutmut-cache` are generated artifacts (gitignored) —
delete `kernel/mutants/` to force a clean regeneration if results look stale
after a source change.

## Scope of the enforced threshold

`evaluator.py` mutates as a whole file (566 mutants total as of this
writing), but `evaluate_action_proposal()` and `evaluate_code_proposal()` are
large functions that also cover body-profile capability checks, envelope
validation, protected-path/self-referential-code detection, and transform
generation — concerns this issue does not scope to. Grading the *whole file*
against one kill-rate number would conflate "well-tested risk arithmetic"
with "under-tested envelope validation," and chasing 100% on the latter is a
different, larger effort than this issue asks for.

The enforced threshold is scored only on mutants that touch risk-clamping
surface: any mutated line containing `risk_score`, `deny_threshold`,
`defer_threshold`, `risk_boost`, a `min(`/`max(` clamp, `isfinite`, or the
`_UNAVAILABLE_*` fail-closed constants, within `_risk_from_analysis()`,
`evaluate_action_proposal()`, and `evaluate_code_proposal()`.

| Scope | Mutants | Killed | Survived | Kill rate |
|---|---|---|---|---|
| **Risk-clamping surface (enforced)** | 71 | 71 | 0 | **100%** |
| `_risk_from_analysis()` alone | 15 | 15 | 0 | 100% |
| Same 3 functions, non-risk mutants (trace_id, reason strings, unrelated dict keys, action-type comparisons) | 213 | 122 | 91 | 57% (informational only, not enforced) |
| Whole file (`evaluator.py`) | 566 | 323 | 236 (+7 no-tests) | 57% (informational only) |

**Enforced floor: 100% kill rate on the risk-clamping surface, as defined
above.** A PR that lowers it must either add tests restoring 100% or, if a
survivor is a genuine equivalent mutant (see below), document why in this
file and get maintainer sign-off (CLAUDE.md §3 — changes to `kernel/` require
review).

### Equivalent mutants encountered

Two mutants in `evaluate_action_proposal()` change the *default* argument of
`profile_deny.get("risk_score", 1.0)` (e.g. to `None`, or drop it entirely).
`_check_body_profile_denials()` always includes `"risk_score"` in the dicts
it returns today, so that default is unreachable through the public
`evaluate_action_proposal()` path alone — both mutants are equivalent under
current behavior. `tests/test_evaluator.py::test_profile_deny_risk_score_defaults_when_dict_omits_it`
kills them anyway by monkeypatching `_check_body_profile_denials` to return a
dict that omits the key, exercising the fallback directly rather than
leaving it as an accepted equivalent-mutant exception.

## Adding coverage for a new risk-clamping mutant

1. `mutmut show <mutant-id>` to see exactly what changed.
2. Add or extend a test in `kernel/tests/test_evaluator.py` that asserts the
   **exact** `d.risk_score` (or `_risk_from_analysis()` return value)
   distinguishing the mutant from the original — not just `d.type`, which
   several of these mutants leave unchanged.
3. Re-run `mutmut run <mutant-id>` (accepts specific IDs, much faster than a
   full run) to confirm it's now killed.
4. Update the table above if the scoped totals changed.

## References

- `kernel/tests/test_evaluator.py` — regression tests, including the
  `# Risk-clamping logic (issue #211...)` section added for this issue.
- `kernel/src/kernel/evaluator.py::_risk_from_analysis` — the function's
  docstring explains the NaN/non-finite fail-closed fix in detail.
- PR #122 — the original fail-open bug (missing analysis treated as zero
  risk) this issue generalizes a systematic check for.
