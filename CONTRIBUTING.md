# Contributing to Engram

Thanks for your interest in contributing! Engram is an open-source (MIT)
neuromorphic AI system, and we welcome contributions of all kinds — code,
tests, documentation, bug reports, and ideas.

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).

## Getting Started

1. **Fork** the repository and clone your fork.
2. Set up your environment (see [README.md](README.md) and [RUN-LOCAL.md](RUN-LOCAL.md)):
   ```bash
   python run.py --install     # one-time dependency install
   python run.py               # start the core profile (dashboard on :8080)
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
4. Make your changes.
5. Run the tests (see below).
6. Commit, push to your fork, and open a Pull Request against `main`.

## Running Tests

The neuromorphic core is the primary test suite:

```bash
cd neuromorphic && uv run python -m pytest tests/ -v -p no:anchorpy
```

CI runs this suite on every pull request (Python 3.11 and 3.12). PRs must pass
before they can be merged.

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
