# Testing Guide

How tests are structured in Engram, how CI runs them, and the conventions to
follow when adding new ones. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the
full contribution workflow and [CLAUDE.md](../CLAUDE.md) for the architecture
invariants a change must preserve.

## Structure

Every service keeps its own tests under `<service>/tests/`, next to the package
it exercises:

```
neuromorphic/tests/       # the brain — largest suite; required for brain changes
sdk/tests/                # BaseService, EventBus, DB, plugins (+ red_team/ authz)
kernel/tests/             # safety decisions (ALLOW/TRANSFORM/DENY/DEFER)
safety-supervisor/tests/  # advisory risk analysis, evasion cases
beliefs/tests/            # constitutional VALUEs, confidence floor
meta-programmer/tests/    # sandbox containment, staging, fail-closed
coordinator/, memory/, cache/, external-api/, planner/, dashboard/, launcher/
```

- One test file per module/behavior, named `test_<subject>.py`.
- Pytest is the runner everywhere. The SDK enables `asyncio_mode = "auto"`
  (see `sdk/pyproject.toml`), so `async def test_*` functions run without an
  explicit marker.
- Put shared fixtures in a `conftest.py` at the appropriate scope (e.g.
  `sdk/tests/conftest.py`, `sdk/tests/red_team/conftest.py`).

## Running tests locally

Match what CI runs so a green local run means a green PR:

```bash
# Neuromorphic (primary suite)
cd neuromorphic && uv run --extra dev python -m pytest tests/ -v -p no:anchorpy

# SDK (with coverage, as CI runs it)
cd sdk && uv run --extra dev python -m pytest tests/ -v

# A single service that has no third-party deps (loads its module directly):
PYTHONPATH=kernel/src python -m pytest kernel/tests -v
```

### Two required gotchas

- **`-p no:anchorpy`** — the neuromorphic suite disables the `anchorpy` pytest
  plugin, which otherwise interferes with collection. Always pass it there.
- **`ENGRAM_SKIP_MUJOCO_LOOP=1`** — the continuous MuJoCo physics-loop tests
  abort nondeterministically on headless machines. CI sets this env var; set it
  locally too if the MuJoCo loop tests are flaky on your box.

## What CI runs (`.github/workflows/test.yml`)

The **Tests** workflow runs on every PR to `dev` and `main`:

| Job | What it covers |
|---|---|
| `neuromorphic` | Full brain suite on Python 3.11 + 3.12 |
| `benchmark-regression` | Step-timing gate vs. the committed baseline (fails > 25% regression) |
| `sdk` | SDK suite on 3.11 + 3.12, with a non-blocking coverage summary in the run report |
| `governance` | Kernel, meta-programmer, coordinator, dashboard, safety-supervisor, beliefs, external-api, and NATS authz red-team tests |
| `sandbox` | Builds the hardened sandbox image and asserts containment |
| `lint` | Ruff (bug subset + full) and Black — **blocking** |
| `typecheck` | Mypy on `sdk/src/activelearning` — **blocking** |

## Automated PR quality bots

These run alongside the test suite and comment on the PR (they do **not** gate
merges — the `Tests` workflow is the gate):

- **PR Review Bot** (`.github/workflows/pr-review.yml`) — runs Ruff and Mypy via
  reviewdog and posts findings as **inline review comments** on the changed
  lines.
- **CodeQL** (`.github/workflows/codeql.yml`) — security + quality static
  analysis for Python and JavaScript; results appear in the Security tab.
- **pre-commit.ci** — if enabled for the repo, runs the hooks in
  `.pre-commit-config.yaml` on each PR and auto-commits formatting fixes. Run the
  same hooks locally with `pre-commit install`.
- **Dependabot** (`.github/dependabot.yml`) — weekly grouped dependency and
  Actions update PRs against `dev`.

## Writing new tests

- **Neuromorphic changes require tests** (CONTRIBUTING.md). Prefer a test that
  asserts the behavior, not the implementation.
- When a change touches a learning mechanism updated on a fixed interval, add or
  extend an **equivalence test** proving the interval result matches
  step-by-step (see CLAUDE.md Invariant 1).
- Safety-layer changes (`kernel/`, `safety-supervisor/`, `beliefs/`,
  `meta-programmer/`) must keep the **fail-closed** tests passing — a check that
  cannot run must deny/halt, never degrade open.
- Keep tests deterministic and offline: inject fakes/mocks for NATS, Qdrant, and
  LLM clients rather than requiring a live service.
