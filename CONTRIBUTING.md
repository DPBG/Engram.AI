# Contributing to Engram

Thanks for your interest in contributing! Engram is an open-source (MIT)
neuromorphic AI system, and we welcome contributions of all kinds — code,
tests, documentation, bug reports, and ideas.

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).

## Branching model (read this first)

Engram uses a two-branch model:

- **`dev`** — the **integration branch**. **All pull requests target `dev`.** It is
  the repository's default branch, so GitHub's "New pull request" page selects it
  automatically.
- **`main`** — the **stable / release branch**. Maintainers merge `dev` into `main`
  for releases. **Do not open feature PRs against `main`** — they'll be asked to
  retarget.

> 🟢 **The golden rule: branch _from_ `dev`, and open your PR _back into_ `dev`.**

## Contribution workflow

### 1. Fork & clone (one-time setup)

Fork the repo on GitHub, then:

```bash
git clone https://github.com/<your-username>/Engram.AI.git
cd Engram.AI
git remote add upstream https://github.com/DPBG/Engram.AI.git   # track the original repo
```

Set up your environment (full guide in [RUN-LOCAL.md](RUN-LOCAL.md)):

```bash
python run.py --install     # one-time: install dependencies
python run.py               # start the core profile (dashboard on http://localhost:8080)
```

### 2. Start from an up-to-date `dev`

Always branch from the latest `dev`:

```bash
git fetch upstream
git checkout -B dev upstream/dev     # make local dev match the latest upstream dev
```

### 3. Create a branch

Name it `type/short-kebab-description` (see [Branch naming](#branch-naming)):

```bash
git checkout -b feat/concept-probe-export
```

### 4. Make your change

Keep it **focused** — one feature or fix per branch/PR. Follow the
[Code Standards](#code-standards) and the [Architecture Invariants](#architecture-invariants).

### 5. Run tests & checks locally — *before* you commit

These are exactly what CI runs, so fixing them now avoids a red PR:

```bash
# Neuromorphic suite (primary)
cd neuromorphic && uv run --extra dev python -m pytest tests/ -v -p no:anchorpy && cd ..

# SDK suite
cd sdk && uv run --extra dev python -m pytest tests/ -v && cd ..

# Lint — ALL three gates are blocking in CI
uvx ruff check --select E9,F63,F7,F82 .  # fast "real bug" gate
uvx ruff check .                           # full style gate
uvx black --check --line-length 100 .     # formatting gate
cd sdk && .venv/bin/mypy src/activelearning/ --ignore-missing-imports  # type gate
```

### 6. Commit

Write a clear, [conventional commit message](#commit-messages):

```bash
git add <files>
git commit -m "feat(neuromorphic): export concept-probe results as JSON"
```

### 7. Push to your fork

```bash
git push -u origin feat/concept-probe-export
```

### 8. Open the Pull Request — base branch **`dev`**

1. On GitHub, open a PR from your branch into **`dev`** (it should be the default base
   — double-check the base dropdown says `base: dev`).
2. **Fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md)** that auto-populates:
   summary, type of change, components affected, how you tested, and the checklist.
3. Wait for **CI to go green** (see [Running tests & CI](#running-tests--ci)). A
   maintainer reviews and merges.

### 9. Respond to review & keep your branch current

If `dev` advances while your PR is open, rebase onto it:

```bash
git fetch upstream
git rebase upstream/dev          # replay your commits on top of the latest dev
git push --force-with-lease      # update the PR (safe force-push)
```

## Branch naming

Prefix the branch with the kind of change:

| Prefix | Use for |
|---|---|
| `feat/` | a new feature |
| `fix/` | a bug fix |
| `docs/` | documentation only |
| `test/` | tests only |
| `refactor/` | restructuring with no behavior change |
| `chore/` | tooling, CI, dependencies, housekeeping |

Examples: `feat/sensor-imu-driver`, `fix/memory-recall-reply`, `docs/sdk-readme`.

## Commit messages

Every commit message uses the **Conventional Commits** form:

```
type(scope): short imperative summary

Optional body — explain WHAT changed and WHY (wrap at ~72 columns).
Reference issues in the body/footer with "Closes #123".
```

**Rules**

- **`type`** — *required*, and must be one of the types in the table below.
- **`scope`** — *optional* — the area touched, e.g. `neuromorphic`, `sdk`,
  `dashboard`, `kernel`, `brain-viz`, `deploy`.
- **`summary`** — *required* — imperative mood ("add", "fix", "remove"), lower-case
  start, **no trailing period**, ≤ ~72 characters.
- One logical change per commit; squash noisy "WIP" commits before opening the PR.

**Allowed commit types** — pick the one that matches your change:

| Type | Use it when you… | Example |
|---|---|---|
| `feat` | add a new feature or capability | `feat(sensory-gateway): add IMU sensor driver` |
| `fix` | fix a bug | `fix(memory): publish recall results over NATS` |
| `docs` | change documentation only | `docs: document the dev-branch PR workflow` |
| `test` | add or fix tests | `test(kernel): cover envelope-clamp boundaries` |
| `refactor` | restructure code with **no** behavior change | `refactor(sdk): unify the NATS client setup` |
| `perf` | improve performance | `perf(neuromorphic): cache per-step routing tables` |
| `chore` | tooling, dependencies, housekeeping | `chore: add uv.lock and constraints.txt` |
| `ci` | change CI / workflows | `ci: run the SDK suite on Python 3.11 and 3.12` |

> 💡 The commit **type usually matches your branch prefix** — a `fix/…` branch should
> contain `fix:` commits, a `docs/…` branch `docs:` commits, and so on.

## Pull request rules

- **Base branch is always `dev`** (not `main`).
- **One focused change per PR** — easier to review and to revert.
- **Fill the PR template** completely (summary, type, components, testing, checklist).
- **CI must be green** before merge — the neuromorphic + SDK suites on Python 3.11 &
  3.12 and the blocking lint gate. Don't merge a red PR.
- **Add/maintain tests for behavior changes** — required for `neuromorphic/` changes.
- **No secrets** — never commit credentials, IPs, SSH keys, or tokens (see
  [Security & Secrets](#security--secrets)).
- **Safety-critical areas** (`kernel/`, `safety-supervisor/`, `meta-programmer/`,
  `beliefs/`) require maintainer review — see
  [Changes That Need Extra Review](#changes-that-need-extra-review).

## Running tests & CI

The neuromorphic core is the primary suite (also run automatically in CI):

```bash
cd neuromorphic && uv run --extra dev python -m pytest tests/ -v -p no:anchorpy
```

On every pull request to `dev` (and `main`), the **`Tests`** workflow runs:

- the **neuromorphic** test suite on **Python 3.11 and 3.12**,
- the **SDK** test suite on **Python 3.11 and 3.12**,
- a **blocking lint gate** (two ruff passes: `--select E9,F63,F7,F82` for real
  bugs, then the full rule set),
- a **blocking formatter gate** (`black --check --line-length 100`), and
- a **blocking type gate** (`mypy` on `sdk/src/activelearning/`).

Your PR can be merged once **all** blocking jobs are green.

## Code Standards

- **Python**: `async`/`await` patterns; type hints on public APIs.
- **Neuromorphic code**: NumPy/SciPy, `float32` throughout, CSR sparse matrices.
- **Dashboard**: standalone FastAPI + vanilla JS (no SDK dependency).
- **All services**: communicate via NATS, persist to SQLite.
- Keep PRs focused — one feature or fix per PR.
- Include test coverage for neuromorphic changes.

## Architecture Invariants

All neuromorphic code must conform to the 6 design invariants defined in
[CLAUDE.md](CLAUDE.md). These are non-negotiable architectural constraints that
define what Engram is:

1. All 6 learning mechanisms must operate simultaneously every step.
2. Developmental critical periods cannot be skipped or hardcoded.
3. LLM queries must be emergent via STDP, not hardcoded IF-THEN.
4. Instinctual gain must be multiplicative (>= 1.0, never suppress).
5. Every synapse group must target a specific dendritic compartment.
6. Continual learning is required — no batch-only training.

If you're unsure whether a change violates an invariant, open an issue and ask
before implementing.

## Areas Open for Contribution

- `neuromorphic/` — brain code, learning algorithms, tests
- `sdk/` — SDK runtime, plugins, event bus
- `dashboard/` — web UI, API endpoints, visualizations
- `sensory-gateway/` — sensor discovery, encoding, aggregation
- `brain-viz/` — 3D visualization demos
- Component services (`coordinator/`, `memory/`, `beliefs/`, etc.)
- Documentation and examples — always welcome, great for first-time contributors

## Changes That Need Extra Review

These areas affect safety or system integrity and require maintainer review
before merge:

- `kernel/` and `safety-supervisor/` — the safety layer. Changes must not weaken
  the ability to gate, deny, or transform unsafe actions.
- `meta-programmer/` — self-evolution / code-generation logic.
- New third-party dependencies or major architectural shifts.
- Changes to synapse/neuron data structures (they affect persisted weights).

## Security & Secrets

- Never commit credentials, server IPs, SSH keys, or API tokens. All secrets go
  in `.env` (git-ignored). See `.env.example`.
- To report a security vulnerability, follow [SECURITY.md](SECURITY.md) — please
  do **not** open a public issue for vulnerabilities.

## Reporting Bugs & Requesting Features

Use the issue templates (Bug Report / Feature Request). For questions and
open-ended discussion, use GitHub Discussions.

## Questions?

Open a discussion or an issue. We're happy to help new contributors get started.
