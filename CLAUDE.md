# CLAUDE.md — Engram Architecture & Contributor Guide

> This file is the **authoritative source** for Engram's non-negotiable
> architectural constraints. It is cited by [CONTRIBUTING.md](CONTRIBUTING.md),
> [DESIGN-PRINCIPLES.md](DESIGN-PRINCIPLES.md), and the issue templates. It also
> serves as the working context for Claude Code and other agents operating in
> this repo.
>
> If a proposed change conflicts with anything here, **stop and open an issue**
> before implementing.

---

## 1. What Engram Is

Engram is a self-aware, continuously-learning neuromorphic AI system. Its
intelligence lives in a spiking neural network (the `neuromorphic/` "brain"),
surrounded by microservices that provide sensory input, safety governance,
memory, planning, and a web dashboard. Services are independent processes that
communicate over **NATS** and persist to **SQLite** (+ **Qdrant** for vectors).

Two ways to run the same system:
- **Pure Python** (`python run.py`) — the launcher downloads NATS and runs each
  service as a subprocess. Best for local development. See [RUN-LOCAL.md](RUN-LOCAL.md).
- **Docker Compose** (`docker compose up`) — each service is a container. The
  Hetzner deployment layers `docker-compose.yml` + `deploy/docker-compose.1m.yml`.

---

## 2. The Six Architectural Invariants (non-negotiable)

All neuromorphic code MUST conform to these. They define what Engram *is*; a
change that violates one is a change to a different system. Primary
implementation files are listed for each.

### Invariant 1 — Integrated Multi-Mechanism Learning
All **6 learning mechanisms** operate together and are never individually
disabled: **STDP, eligibility traces, BCM metaplasticity, 4-channel
neuromodulation (DA/ACh/NE/5-HT), homeostatic scaling, and R-STDP.**
- *Files:* `neuromorphic/src/neuromorphic/synapses.py`, `neuromodulation.py`, `network.py`
- *Enforcement note:* For performance, some mechanisms update on a fixed
  interval (e.g. STDP every N steps) using **compensated decay** so the result
  is mathematically equivalent to running every step. This is "logically every
  step." Any change to those intervals MUST preserve equivalence and be covered
  by an equivalence test. Mechanisms may never be turned *off*.

### Invariant 2 — Developmental Critical Periods
Five phases (**infant → toddler → juvenile → adolescent → mature**) with
**experience-dependent** transitions. The adolescent transition in particular is
gated on learning signals (concept differentiation, sensory variance, feature-
STDP decline), **never hardcoded by step count alone.**
- *Files:* `neuromorphic/src/neuromorphic/neuromodulation.py`, `network.py`

### Invariant 3 — Hybrid SNN-LLM Cognitive Architecture
LLM queries are **emergent** — driven by learned STDP pathways into the
cognitive motor range — **not hardcoded IF-THEN logic.** The LLM is a "teacher /
book," consulted by the brain, never the brain itself.
- *Files:* `neuromorphic/src/neuromorphic/decoding.py`, `cognitive_bridge.py`, `service.py`

### Invariant 4 — Cross-Modal Binding with Instinctual Gain
Arbitrary modalities bind via temporal correlation. Instinctual/orienting gain
is **always multiplicative and ≥ 1.0** — it may amplify salient input but must
**never suppress** sensory input below baseline.
- *Files:* `neuromorphic/src/neuromorphic/instincts.py`, `encoding.py`, `network.py`

### Invariant 5 — Multi-Compartment Dendritic Processing
Cortical neurons have **4 dendritic compartments**; **every cortical synapse
group targets a specific compartment**, with supralinear dendritic spikes in
apical compartments. (Subcortical point-neuron regions — brainstem, reflex —
are intentionally somatic.)
- *Files:* `neuromorphic/src/neuromorphic/neurons.py`, `config.py`, `synapses.py`

### Invariant 6 — Neuromodulatory Continual Learning
Eligibility traces persist **≥ 1000 ms** for delayed credit assignment;
homeostatic scaling prevents catastrophic forgetting; learned weights and
plasticity state **persist via SQLite/disk and survive restarts.** No
batch-only training — learning is online and continual.
- *Files:* `neuromorphic/src/neuromorphic/neuromodulation.py`, `synapses.py`, `persistence.py`

---

## 3. Safety Architecture (treat as critical)

Engram can generate and run its own code (`meta-programmer/`), so the safety
layer is load-bearing. The intended invariant is: **every action and code
proposal passes through the Kernel, which is the sole authority that may emit a
decision.**

- The **Kernel** (`kernel/`) evaluates proposals → `ALLOW / TRANSFORM / DENY /
  DEFER` and **fails safe (DENY) on internal error.**
- The **Safety Supervisor** (`safety-supervisor/`) provides advisory risk
  analysis; it does not decide.
- The **Beliefs** graph (`beliefs/`) holds constitutional VALUEs with a
  confidence floor that cannot be lowered.
- The **Meta-Programmer** (`meta-programmer/`) must only deploy code that the
  Kernel approved, and only into the sandbox/allowlisted paths.

**Rules for changes in `kernel/`, `safety-supervisor/`, `meta-programmer/`,
`overrides/`, and `beliefs/`:**
1. Never weaken the ability to gate, deny, or transform an unsafe action.
2. Safety dependencies are **fail-closed**: if a check cannot run, deny/halt —
   never degrade open.
3. Changes here require maintainer review (see CONTRIBUTING.md).

> **Known hardening work (tracked in [ROADMAP.md](ROADMAP.md) Phase 1):** the
> decision bus is not yet authenticated/signed, and the sandbox is being made
> fail-closed. Until that lands, do not build new autonomy on top of the gate.

---

## 4. Repository Layout

| Path | Role |
|---|---|
| `neuromorphic/` | The spiking-neural-network brain (the intelligence). |
| `sdk/` | Shared `activelearning` SDK: `BaseService`, `EventBus` (NATS), DB, plugins. |
| `kernel/`, `safety-supervisor/`, `beliefs/` | Safety governance. |
| `coordinator/`, `planner/`, `meta-programmer/` | Task routing, planning, self-evolution. |
| `memory/`, `cache/`, `external-api/`, `overrides/` | Supporting services. |
| `sensory-gateway/` | Sensor discovery + video training pipeline → spikes. |
| `dashboard/` | FastAPI + vanilla-JS web UI (`localhost:8080`). |
| `brain-viz/` | Three.js 3D brain visualizations. |
| `launcher/`, `run.py` | Pure-Python process supervisor + NATS bootstrap. |
| `deploy/` | Docker/compose overrides, cloud-init, watchdogs for Hetzner. |
| `docs/` | Architecture, scaling, training, and safety design docs. |

Each Python service is a package under `<service>/src/<pkg>/`, launched as
`python -m <module>` (see `launcher/registry.py`). `sensory-gateway/` uses a flat
layout (module `gateway`).

---

## 5. Development Commands

```bash
# Pure-Python (recommended for dev)
python run.py --install            # one-time: install requirements-local.txt
python run.py                      # start the 'core' profile (dashboard :8080)
python run.py --profile full       # add Qdrant/Ollama-backed services
python run.py --only kernel,planner
python run.py --list               # list services and their needs

# Docker
docker compose up                  # core services
docker compose --profile full up   # + qdrant, memory, cache, coordinator
docker compose --profile extra up  # + sensory-gateway, overrides, meta-programmer

# Tests
cd neuromorphic && uv run --extra dev python -m pytest tests/ -v -p no:anchorpy
cd sdk && python -m pytest tests/ -v
```

---

## 6. Coding Standards

- **Python**: `async`/`await`; type hints on public APIs; target 3.11/3.12.
- **Neuromorphic**: NumPy/SciPy, `float32` throughout, CSR sparse matrices;
  keep hot paths vectorized (no per-neuron Python loops).
- **Services**: communicate via NATS (`EventBus`), persist to SQLite. Prefer the
  SDK `BaseService`/`EventBus` over raw `nats.connect()`.
- **Dashboard**: standalone FastAPI + vanilla JS (no SDK dependency).
- **Lint/format/type**: `ruff`, `black` (line length 100), `mypy`. These are
  enforced in CI — run them before pushing.
- Keep PRs focused — one feature or fix per PR; include tests for behavior
  changes (required for neuromorphic changes).

---

## 7. Shared Dependency Versions

Shared runtime dependencies are pinned to common floors across all services
(see `constraints.txt`). Notably: `numpy>=1.26.0`, `pydantic>=2.5.0`,
`nats-py>=2.6.0`, `scipy>=1.12.0`. When adding or bumping a shared dependency,
update `constraints.txt` and keep the per-service `pyproject.toml`/
`requirements.txt` consistent with it.

---

## 8. Pointers

- Run guide: [RUN-LOCAL.md](RUN-LOCAL.md)
- Design rationale: [DESIGN-PRINCIPLES.md](DESIGN-PRINCIPLES.md)
- Contribution workflow: [CONTRIBUTING.md](CONTRIBUTING.md)
- Technical plan & current-state audit: [ROADMAP.md](ROADMAP.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Deeper architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
