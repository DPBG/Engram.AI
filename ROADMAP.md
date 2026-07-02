# Engram Roadmap

> Technical plan & current-state audit. This is the file [CLAUDE.md](CLAUDE.md)
> (§3, §8) and other docs point to for "what phase are we in and what's left."
> It tracks the same Phase/Milestone numbering used by GitHub milestones
> (`M1`–`M7`) and issue labels (`phase-1`–`phase-7`), and by the task
> numbering (`Task N.N`) cited in ADRs such as
> [docs/adr/0001-nats-authz.md](docs/adr/0001-nats-authz.md).
>
> **How this stays current:** each milestone below lists its tracking issues
> and their state at time of writing. When a milestone's issues close, update
> its Status line here — don't let this drift the way the original
> `ROADMAP.md` did (see issue [#144](https://github.com/DPBG/Engram.AI/issues/144)).
> Status legend: ✅ Complete · 🟡 In Progress · ⚪ Not Started.

---

## Overview

| Phase | Milestone | Status | Focus |
|---|---|---|---|
| 1 | [M1 — Safety Real](#phase-1--safety-real-m1) | ✅ Complete | Unforgeable Kernel gate, real sandbox containment, tested safety stack |
| 2 | [M2 — Reliable Backbone](#phase-2--reliable-backbone-m2) | ✅ Complete | Messaging, persistence, and supervision are trustworthy |
| 3 | [M3 — Validated Intelligence](#phase-3--validated-intelligence-m3) | 🟡 In Progress | Prove emergent learning produces useful, measurable abstractions |
| 4 | [M4 — Real-Time Performance](#phase-4--real-time-performance-m4) | 🟡 In Progress | Close the real-time gap at ~1M-neuron scale |
| 5 | [M5 — Embodied Autonomy](#phase-5--embodied-autonomy-m5) | ✅ Complete | Move from MuJoCo-only to physical-world sensing and actuation |
| 6 | [M6 — Autonomous Self-Evolution](#phase-6--autonomous-self-evolution-m6) | 🟡 In Progress | Mature the meta-programmer's autonomy, gated on M1 |
| 7 | [M7 — Community & Ecosystem Growth](#phase-7--community--ecosystem-growth-m7) | 🟡 In Progress | Lower the barrier to contribution |

Phases are **not strictly sequential** — M3/M4 (intelligence validation,
performance) and M5–M7 (embodiment, autonomy, community) proceed in
parallel once M1/M2 (safety, backbone) are solid. M6 (self-evolution
autonomy) is explicitly gated on M1, per [CLAUDE.md](CLAUDE.md) §3: *"do not
build new autonomy on top of [the decision] gate"* until M1's hardening work
is fully landed.

---

## Phase 1 — Safety Real (M1)

**Status: ✅ Complete** — 14/14 tracking issues closed.

Unforgeable gate, real containment, tested safety stack. Blocks all
autonomy (see M6). This is the phase [CLAUDE.md](CLAUDE.md) §3 and
[docs/adr/0001-nats-authz.md](docs/adr/0001-nats-authz.md) reference by task
number.

| Task | Deliverable | Issues |
|---|---|---|
| 1.1 | NATS account & permission model — per-service credentials, subject allowlists (Kernel is the sole `decision.*`/`policy.*` publisher), red-team regression | [#64](https://github.com/DPBG/Engram.AI/issues/64), [#94](https://github.com/DPBG/Engram.AI/issues/94), [#95](https://github.com/DPBG/Engram.AI/issues/95), [#106](https://github.com/DPBG/Engram.AI/issues/106) |
| 1.2 | Decision-bus signing (defends the application layer independent of broker ACLs) | [PR #36](https://github.com/DPBG/Engram.AI/pull/36) |
| 1.3 | Sandbox containment — minimal internal-only image, fail-closed when Docker/image unavailable, containment integration tests (no network, read-only FS, pid/mem/cpu limits) | [#65](https://github.com/DPBG/Engram.AI/issues/65), [#66](https://github.com/DPBG/Engram.AI/issues/66), [#107](https://github.com/DPBG/Engram.AI/issues/107) |
| 1.7 | Operator/dashboard publisher scope | folded into the Kernel-hardening PRs [#36](https://github.com/DPBG/Engram.AI/pull/36), [#37](https://github.com/DPBG/Engram.AI/pull/37) |
| 1.8 | Safety-supervisor & beliefs testing — non-lowerable VALUE confidence floor, risk-heuristic unit tests, code-path/AST evasion regressions | [#67](https://github.com/DPBG/Engram.AI/issues/67), [#68](https://github.com/DPBG/Engram.AI/issues/68), [#98](https://github.com/DPBG/Engram.AI/issues/98) |
| 1.9 | Human-in-the-loop controls — approval consumer, post-deploy health-check/auto-rollback, Kernel-loss watchdog → `SAFE_HALT`, dashboard emergency-stop button | [#69](https://github.com/DPBG/Engram.AI/issues/69), [#70](https://github.com/DPBG/Engram.AI/issues/70), [#108](https://github.com/DPBG/Engram.AI/issues/108), [#109](https://github.com/DPBG/Engram.AI/issues/109) |

**Related ADR:** [0001 — NATS account & permission model](docs/adr/0001-nats-authz.md).

**Follow-up:** end-to-end regression coverage for the Task 1.3 fail-closed
sandbox deploy path is tracked separately under M6 ([#142](https://github.com/DPBG/Engram.AI/issues/142)),
since it depends on the meta-programmer deploy path M6 is hardening.

---

## Phase 2 — Reliable Backbone (M2)

**Status: ✅ Complete** — 6/6 tracking issues closed.

Messaging, persistence, and supervision are trustworthy.

| Task | Deliverable | Issues |
|---|---|---|
| 2.1 | `BaseService`/`EventBus` migration — contract + parity checklist, migrate `external-api` | [#96](https://github.com/DPBG/Engram.AI/issues/96), [#110](https://github.com/DPBG/Engram.AI/issues/110) |
| 2.2 | Validate-on-receive in the `EventBus` subscribe wrapper | [#71](https://github.com/DPBG/Engram.AI/issues/71) |
| 2.3 | Durable JetStream consumers — explicit ack + redelivery handling | [#97](https://github.com/DPBG/Engram.AI/issues/97) |
| 2.6 | Zero-vector-on-error guard in embeddings/recall | [#72](https://github.com/DPBG/Engram.AI/issues/72) |
| 2.7 | SDK `nats_client` tests (reconnect, request-reply, subscribe wrapper) | [#73](https://github.com/DPBG/Engram.AI/issues/73) |

---

## Phase 3 — Validated Intelligence (M3)

**Status: 🟡 In Progress** — 2/4 tracking issues closed.

Prove emergent learning produces useful, measurable abstractions —
concept-probe metrics, developmental-transition validation, cross-modal
binding benchmarks.

| Issue | Deliverable | State |
|---|---|---|
| [#128](https://github.com/DPBG/Engram.AI/issues/128) | Turn `test_concept_probe.py` into a scored concept-separability benchmark | ✅ Closed |
| [#129](https://github.com/DPBG/Engram.AI/issues/129) | Developmental-transition validation suite (Invariant 2) | ✅ Closed |
| [#130](https://github.com/DPBG/Engram.AI/issues/130) | Cross-modal binding accuracy benchmark | ⚪ Open |
| [#131](https://github.com/DPBG/Engram.AI/issues/131) | Publish a "Learning Evidence" panel in the dashboard | ⚪ Open |

---

## Phase 4 — Real-Time Performance (M4)

**Status: 🟡 In Progress** — 2/4 tracking issues closed.

Close the real-time gap at ~1M-neuron scale — profiling, a CI
perf-regression gate, compiled/GPU kernels for hot paths. This is the
milestone tracked by README's Known Limitations "Performance / scale" note.

| Issue | Deliverable | State |
|---|---|---|
| [#132](https://github.com/DPBG/Engram.AI/issues/132) | CI performance-regression gate using `scripts/benchmark.py` | ⚪ Open |
| [#133](https://github.com/DPBG/Engram.AI/issues/133) | Profile the neuromorphic hot path and publish findings as an ADR | ⚪ Open |
| [#134](https://github.com/DPBG/Engram.AI/issues/134) | Prototype a compiled kernel for the STDP + eligibility-trace update path | ✅ Closed |
| [#135](https://github.com/DPBG/Engram.AI/issues/135) | Investigate a GPU backend (CuPy/JAX) for CSR sparse synapse operations | ✅ Closed |

**Related ADRs:** [0002 — Neuromorphic hot-path profiling & optimization
strategy](docs/adr/0002-neuromorphic-hotpath-profiling.md) (once merged, see
[#133](https://github.com/DPBG/Engram.AI/issues/133));
[docs/GPU-SYNAPSE-BACKEND-FEASIBILITY.md](docs/GPU-SYNAPSE-BACKEND-FEASIBILITY.md)
(#135's finding: on CPU, a JAX backend for `compute_current` was 12–37×
*slower* than the existing SciPy implementation at every scale tested).

---

## Phase 5 — Embodied Autonomy (M5)

**Status: ✅ Complete** — 4/4 tracking issues closed.

Move from MuJoCo-only to physical-world sensing and actuation — IMU/depth
drivers, real-actuator adapters, sim-to-real transfer.

| Issue | Deliverable | State |
|---|---|---|
| [#136](https://github.com/DPBG/Engram.AI/issues/136) | IMU sensor driver under `sensory-gateway/sensors/` | ✅ Closed |
| [#137](https://github.com/DPBG/Engram.AI/issues/137) | Depth-camera sensor driver for cross-modal binding | ✅ Closed |
| [#138](https://github.com/DPBG/Engram.AI/issues/138) | Extend MuJoCo motor feedback to a real-actuator adapter | ✅ Closed |
| [#139](https://github.com/DPBG/Engram.AI/issues/139) | Sim-to-real transfer plan (MuJoCo → physical robot) | ✅ Closed |

**Related doc:** [docs/SIM-TO-REAL.md](docs/SIM-TO-REAL.md).

---

## Phase 6 — Autonomous Self-Evolution (M6)

**Status: 🟡 In Progress** — 3/4 tracking issues closed.

Mature the meta-programmer's autonomy, **strictly gated on M1 (Safety Real)
completion** — no new autonomy on top of an unsigned decision bus.

| Issue | Deliverable | State |
|---|---|---|
| [#140](https://github.com/DPBG/Engram.AI/issues/140) | Gate meta-programmer autonomy on M1 completion | ✅ Closed |
| [#141](https://github.com/DPBG/Engram.AI/issues/141) | Audit and close meta-programmer-adjacent stubs/TODOs in `neuromorphic/src` | ✅ Closed |
| [#142](https://github.com/DPBG/Engram.AI/issues/142) | End-to-end test coverage for sandbox "fail-closed deploy" (Task 1.3.2) | ⚪ Open |
| [#143](https://github.com/DPBG/Engram.AI/issues/143) | Track Kernel `ALLOW`/`TRANSFORM`/`DENY` rates as a meta-programmer quality signal | ✅ Closed |

---

## Phase 7 — Community & Ecosystem Growth (M7)

**Status: 🟡 In Progress** — 1/4 tracking issues closed.

Lower the barrier to contribution — restore `ROADMAP.md`, promote advisory
CI checks to blocking, onboarding issues, SDK examples.

| Issue | Deliverable | State |
|---|---|---|
| [#144](https://github.com/DPBG/Engram.AI/issues/144) | Restore/author `ROADMAP.md` | 🟡 This document |
| [#145](https://github.com/DPBG/Engram.AI/issues/145) | Promote `ruff`/`black`/`mypy` from advisory to blocking in CI | ⚪ Open |
| [#146](https://github.com/DPBG/Engram.AI/issues/146) | Audit for "good first issue" candidates across `neuromorphic/`, `sdk/`, `sensory-gateway/` | ⚪ Open |
| [#147](https://github.com/DPBG/Engram.AI/issues/147) | Minimal `SensorPlugin`/`ActuatorPlugin` example in the SDK | ✅ Closed |

---

## References

- [CLAUDE.md](CLAUDE.md) — architectural invariants, safety rules, this
  repo's non-negotiable constraints.
- [DESIGN-PRINCIPLES.md](DESIGN-PRINCIPLES.md) — the 6 architecture
  invariants and implementation file map.
- [CONTRIBUTING.md](CONTRIBUTING.md) — branching model, PR workflow, code
  standards.
- [README.md](README.md) — Known Limitations section tracks the same gaps
  M3/M4/M5 close.
- `docs/adr/` — architecture decision records referenced by task number
  above.
