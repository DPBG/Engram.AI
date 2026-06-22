# 0001 — NATS account & permission model

- **Status:** Proposed (awaiting maintainer security review)
- **Date:** 2026-06-21
- **Tracking issue:** [#64](https://github.com/DPBG/Engram.AI/issues/64) (E1.1.1)
- **Milestone:** M1 — Safety Real
- **Phase:** 1 (Safety hardening). Blocks the rest of Task 1.1 (per-service
  credentials, enforced allowlists, red-team test).
- **Related:** Decision signing (Task 1.2, [#36](https://github.com/DPBG/Engram.AI/pull/36)),
  operator/dashboard publisher scope (Task 1.7).

---

## Context

`deploy/nats-1m.conf` authenticates every client with a **single shared token**:

```hocon
authorization {
  token: $NATS_TOKEN
}
```

There are no per-service users and no subject permissions, so the broker
enforces *nothing* about **who** may publish **what**. Any process holding
`$NATS_TOKEN` can publish on any subject — including the safety-critical ones.
Concretely, a compromised or buggy non-Kernel service could publish a forged
`decision.<trace_id>` and a waiter (`EventBus.wait_for_decision`) would receive
it off the bus.

Decision signing (Task 1.2) already defends the *application* layer: a
`decision.*` payload without a valid signature is rejected by `verify_decision`
in `sdk/.../nats_client.py`. That assumes a **hostile bus**. This ADR closes the
complementary gap: make the **bus itself** least-privilege so unauthorized
publishes are refused at the transport layer, before any service has to reason
about them. Defense in depth — signing and transport authz are independent
layers, and neither replaces the other.

This ADR is a **design** deliverable. It specifies the model and the complete
subject × service matrix. Generating per-service credentials, switching
`nats-1m.conf` to the account model, and the red-team test are follow-up issues
that this one unblocks.

### Architectural invariants

This change touches the safety architecture (CLAUDE.md §3). It only ever
**adds** restrictions to the bus; it never weakens the ability to gate, deny, or
transform an action, and it is **fail-closed** (see *Dev-mode vs enforced-mode*).
No neuromorphic invariant is affected.

---

## Decision

### 1. One account per trust domain, one user per service

NATS has two nesting levels of isolation:

- **Accounts** are hard isolation boundaries: subjects do **not** cross account
  lines unless explicitly exported/imported. Two services in different accounts
  cannot see each other's subjects at all.
- **Users** live inside an account, share its subject space, and each carry
  their own publish/subscribe permission allowlist.

Engram's services form **one cooperating message fabric** — almost every subject
is meant to be visible to the right counterpart service — so hard account
isolation per service would require dozens of export/import pairs and fight the
design. We therefore use:

- A single **`ENGRAM`** account containing the whole service fabric, with **one
  user per service**, each with an explicit publish/subscribe allowlist
  (sections 3–4). This is where least-privilege is enforced.
- A separate **`SYS`** account (NATS system account) for operations/monitoring
  only, not used by application services.

This keeps the subject namespace flat (no import/export plumbing) while still
giving every service its own credential and its own allowlist. If a future
service must be *hard*-isolated (e.g. an untrusted third-party plugin), it can be
promoted to its own account with explicit exports — the model leaves room for
that without reworking the common case.

### 2. Credential mechanism: NKEY + signed user JWT (decentralized auth)

Use NATS **decentralized auth** (operator → account → user JWTs, NKEY-signed),
not the static `authorization { users: [...] }` block:

- Each service gets a **user JWT** + its **NKEY seed**, distributed as a single
  `.creds` file. The broker is configured with the account's public key only; it
  never holds any service's private seed.
- Permissions are baked into the signed JWT, so they cannot be tampered with in
  transit and rotating one service's credential does not require editing a
  central server config or restarting the broker.
- This mirrors the credential-as-config pattern already used for decision
  signing (`ENGRAM_DECISION_KEY`): the secret lives in a file/env var supplied
  to the process, not in shared source.

Rejected alternative — static `users` list in `nats-1m.conf`: simpler, but every
credential and permission lives in one server-side file, rotation means editing
+ reloading the server, and there is no per-service private key. Acceptable for a
first cut but it does not scale to per-service rotation; decentralized JWT is the
target. (A static-users variant MAY be used as a transitional implementation in
the follow-up issue, as long as the same matrix below is enforced.)

#### Credential distribution

| Runtime | How the `.creds` reaches the service |
|---|---|
| `run.py` (pure Python) | Launcher generates/loads a `.creds` per service into a gitignored `secrets/` dir and passes its path via `NATS_CREDS` env var; `EventBus.connect()` passes it to `nats.connect(..., user_credentials=...)`. |
| Docker Compose | Each service container mounts its own `.creds` (Docker secret / read-only bind mount); `NATS_CREDS` points at the mount path. The broker mounts only the operator/account public JWTs. |

No service ever receives another service's seed. The `secrets/` directory and
`*.creds` are gitignored; only the **account public** material is committed.

### 3. The privileged Kernel publisher set (global rule)

**Only the `kernel` user may publish** the following subjects. Every other
service's allowlist explicitly omits publish on them; the Kernel is the sole
authority that may emit a decision (CLAUDE.md §3):

| Privileged subject | Meaning |
|---|---|
| `decision.>` | Per-trace action decisions (`decision.<trace_id>`). |
| `code.decision.>` | Per-trace code-proposal decisions. |
| `policy.*` | Policy state the Kernel owns (`policy.restrict`, `policy.update`, `policy.rollback`, `policy.load_profile`, and `policy.*.status`). |
| `cognitive.response.validated` | Kernel's attestation that a cognitive/LLM response passed validation. |

> **⚠ Conflict surfaced by the audit — must be resolved by the implementation
> issue.** `neuromorphic/src/neuromorphic/service.py` currently publishes
> `policy.restrict` directly (lines ~1496 and ~1509). Under this rule the brain
> may **not** publish `policy.*`. The brain must instead *request* a restriction
> via a non-privileged subject (proposal: `policy.restrict.request`) that the
> Kernel subscribes to and, if it agrees, re-publishes as the authoritative
> `policy.restrict`. The matrix below encodes the **target** state
> (`neuromorphic` → publish `policy.restrict.request`, not `policy.restrict`).
> Shipping enforcement before this refactor would break the brain's halt path,
> so the two land together.

### 4. The operator / dashboard publisher set

The `dashboard` user is the human operator's hands. It may publish the
operator-control and command subjects (halt/resume, policy commands, probes,
sensory-gateway commands, motor guidance, approval responses) but is **not** in
the privileged Kernel set — an operator asks the Kernel to halt
(`safety.halt`); the operator does not forge a `decision.*`. This ties to Task
1.7 (operator publisher scope). The dashboard's broad `>` subscription
(read-everything, for the live UI) is **subscribe-only** and stays; it grants no
publish rights.

### 5. Deny-by-default allowlists

Every user's JWT sets `publish`/`subscribe` to an **explicit allow list**.
Anything not listed is denied. The default NATS behavior (allow-all) is never
used for an application user. The matrix in the next section is the source of
truth for these lists.

### 6. Dev-mode vs enforced-mode (fail-safe-by-config)

Mirror the `ENGRAM_DECISION_KEY` pattern (signing is enforced when a key is
configured, and the system fails closed when it should be on but isn't):

- **Enforced mode (production, default for deploy):** the broker uses the
  account model; services must present a valid `.creds`. This is the Hetzner /
  `deploy/` configuration. A service with no/invalid creds cannot connect — fail
  closed.
- **Dev-mode (local `run.py`, opt-in):** a single permissive credential is
  allowed *only* when an explicit env flag (e.g. `ENGRAM_NATS_DEV=1`) is set, so
  contributors can run the stack without minting per-service creds. The launcher
  logs a clear warning that authz is not enforced.
- **Fail-safe rule:** absence of configuration must not silently degrade to
  allow-all in a deployed environment. The deploy path sets enforced mode
  explicitly; if the enforced config is selected but creds are missing, the
  service fails to start rather than falling back to the shared token.

---

## Subject × service permission matrix

This is the acceptance-criteria artifact: for **every subject in the
codebase**, exactly which service(s) may **publish** (P) and **subscribe** (S).
Service keys: `kernel`, `safety` (safety-supervisor), `beliefs`, `memory`,
`coordinator`, `planner`, `neuro` (neuromorphic), `meta` (meta-programmer),
`cache`, `extapi` (external-api), `overrides`, `dashboard` (also the operator),
`gateway` (sensory-gateway). "operator" = the human, acting through `dashboard`.

`>` and `*` are NATS wildcards. A row whose subject ends in `>` / `*` is a
prefix/token family; the allowlist entry uses the wildcard form.

> Source-of-truth note: the named constants in this table are mirrored in
> `sdk/src/activelearning/subjects.py`. `sdk/tests/test_adr_subject_matrix.py`
> fails CI if a constant there is missing from this document, so the matrix and
> the code cannot silently drift.

### Governance / Kernel

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `proposal.new` | coordinator, overrides, neuro | kernel | Action proposals. JetStream (SAFETY_CRITICAL). dashboard subscribes read-only via `>`. |
| `code.proposal` | meta | kernel | Code proposals. JetStream. |
| `decision.>` | **kernel only** | planner, neuro, overrides, waiters | Per-trace decisions. JetStream. Signed (Task 1.2). |
| `code.decision.>` | **kernel only** | meta, waiters | Per-trace code decisions. JetStream. |
| `kernel.status` | dashboard (request) | kernel | Request/reply status. |
| `kernel.status.response` | kernel | dashboard | Reply payload. |

### Policy (Kernel-owned; see §3)

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `policy.load_profile` | dashboard | kernel | Operator loads a policy profile. |
| `policy.restrict` | **kernel only** | kernel, planner, neuro | Authoritative restriction. |
| `policy.restrict.request` | neuro, dashboard | kernel | **New** — non-privileged ask; replaces neuro's direct `policy.restrict` publish (§3 conflict). |
| `policy.rollback` | dashboard | kernel | |
| `policy.update` | dashboard | kernel | |
| `policy.profile.status` | kernel | dashboard | |
| `policy.rollback.status` | kernel | dashboard | |
| `policy.update.status` | kernel | dashboard | |
| `policy.restrict.status` | kernel | dashboard | |
| `cognitive.response.validate` | neuro | kernel | Brain asks Kernel to validate an LLM response. |
| `cognitive.response.validated` | **kernel only** | neuro | Kernel's attestation. |
| `cognitive.response.rejected` | kernel | neuro, dashboard | |
| `cognitive.query` | neuro | cognitive-bridge | Legacy direct query; superseded by `proposal.new` → `cognitive.execute`. |

### Safety

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `safety.analyze.action` | kernel (request) | safety | Risk analysis request. |
| `safety.analyze.code` | kernel (request) | safety | |
| `safety.status` | dashboard (request) | safety | |
| `safety.status.response` | safety | dashboard | |
| `safety.halt` | dashboard | kernel | Operator kill switch (Phase 1.9). |
| `safety.resume` | dashboard | kernel | |
| `safety.halt.status` | kernel | dashboard | |
| `safety.deny_escalation` | kernel | dashboard | |
| `safety.watchdog.status` | neuro | dashboard | Brain watchdog heartbeat. |

### Beliefs

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `beliefs.add_node` | neuro | beliefs | |
| `beliefs.add_edge` | neuro, operator | beliefs | |
| `beliefs.update` | operator | beliefs | |
| `beliefs.query` | operator | beliefs | |
| `beliefs.contradictions` | operator | beliefs | |
| `beliefs.query.request` | kernel (request) | beliefs | Kernel queries norms during decisions. |
| `beliefs.query.response` | beliefs | kernel | Reply payload. |

### Memory

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `memory.store` | neuro | memory | |
| `memory.query` | operator, services (request) | memory | |
| `memory.recall` | operator, services (request) | memory | |

### Coordinator / tasks

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `task.request` | operator, services | coordinator | |
| `task.result` | coordinator | dashboard, requesters | |
| `coordinator.status` | dashboard (request) | coordinator | |
| `coordinator.status.result` | coordinator | dashboard | |
| `demo.start` | dashboard | coordinator | |
| `demo.observation` | dashboard | coordinator | |
| `demo.finish` | dashboard | coordinator | |
| `demo.started` | coordinator | dashboard | |
| `demo.finished` | coordinator | dashboard | |
| `demo.failed` | coordinator | dashboard | |
| `device.unknown` | gateway | coordinator | |

### Planner

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `planner.mode` | kernel, dashboard | planner | Kernel sets `SAFE_HALT`; operator may change mode. |
| `planner.status` | dashboard (request) | planner | |
| `observation.*` | gateway, dashboard, sensors | planner, neuro | Per-sensor observations (`observation.<sensor_id>`). |

### Neuromorphic (brain)

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `neuromorphic.metrics` | neuro | dashboard, gateway | |
| `neuromorphic.status` | dashboard (request) | neuro | |
| `neuromorphic.drives.event` | dashboard, operator | neuro | |
| `neuromorphic.teach` | dashboard, operator | neuro | |
| `neuromorphic.train_bulk` | dashboard, operator | neuro | |
| `neuromorphic.concept.probe` | dashboard | neuro | |
| `neuromorphic.concept.result` | neuro | dashboard | |
| `neuromod.teach.da` | dashboard, operator | neuro | |
| `body.posture` | gateway, mujoco | neuro | |
| `motor.outcome.>` | mujoco, motor | neuro | Motor feedback. |
| `motor.guidance` | dashboard | neuro, mujoco | Operator motor guidance. |
| `approval.request` | neuro | dashboard | |
| `approval.response.>` | dashboard | neuro | Operator approves/denies. |
| `cognitive.execute` | kernel | neuro, dashboard | |
| `speech.execute` | kernel | neuro, dashboard | |
| `mujoco.body.state` | mujoco | dashboard | |
| `observation.visual.body` | gateway, mujoco | dashboard, planner, neuro | Subset of `observation.*`. |
| `actuator.heartbeat.>` | neuro | actuators/hardware | Per-channel actuator routing. |

### Meta-programmer / learning

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `knowledge.gap` | neuro | meta | |
| `metaprogrammer.status` | dashboard (request) | meta | |
| `metaprogrammer.status.response` | meta | dashboard | |
| `device.driver.ready` | meta | gateway | |

### Cache

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `cache.query` | dashboard, services (request) | cache | |
| `cache.setting` | dashboard | cache | |
| `cache.status` | dashboard (request) | cache | |
| `autopilot.setting` | dashboard | cache | |

### External API

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `external.query` | services, operator (request) | extapi | |
| `external.status` | dashboard (request) | extapi | |

### Overrides

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `override.request` | operator | overrides | Overrides itself publishes `proposal.new` + waits on `decision.<trace>`. |
| `override.status` | dashboard (request) | overrides | |

### Sensory gateway

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `sensory.gateway.command` | dashboard | gateway | Operator commands. |
| `sensory.gateway.status` | gateway | dashboard | |
| `video.training.status` | gateway | dashboard | |

### System / observability

| Subject | Publish | Subscribe | Notes |
|---|---|---|---|
| `system.shutdown` | operator | all services | Broadcast. |
| `system.health` | all services | dashboard | Liveness. |
| `heartbeat.*` | all services | dashboard | Per-service heartbeat (`heartbeat.<service>`). |

> The dashboard additionally holds a single broad **subscribe-only** grant on
> `>` to drive the live UI. It conveys **no** publish rights. This is the one
> intentional wildcard subscriber; every other subscribe entry is scoped.

---

## Consequences

### Positive

- A compromised non-Kernel service can no longer forge `decision.*`,
  `code.decision.*`, `policy.*`, or `cognitive.response.validated` at the
  transport layer — defense in depth on top of decision signing.
- Every service has its own rotatable credential; revoking one does not affect
  others and does not require sharing a global token.
- The matrix above is an executable contract: the CI test ties it to
  `subjects.py`, so adding a subject without documenting its authz fails the
  build.

### Negative / costs

- **Operational surface:** per-service `.creds` must be generated, distributed,
  and rotated. The `run.py` dev-mode fallback mitigates local friction.
- **Refactor required before enforcement:** `neuromorphic` must stop publishing
  `policy.restrict` directly (§3). Enforcing the matrix before that lands would
  break the brain's restriction path, so they ship together.
- **Wildcard subscribers:** the dashboard's `>` subscription is broad by
  necessity (live UI). It is read-only, but it does mean the operator credential
  can observe all traffic — acceptable, since the operator is trusted, and it
  cannot publish outside its allowlist.
- Request/reply replies go to ephemeral `_INBOX.>` subjects; every service that
  uses `request()` needs publish on `_INBOX.>` and subscribe on its own inbox.
  The implementation issue must include the standard `_INBOX.>` grant in each
  user's allowlist (NATS convention) so request/reply keeps working.

### Follow-up (unblocked by this ADR)

1. Generate per-service NKEY/JWT creds + wire `NATS_CREDS` into `EventBus` and
   both runtimes.
2. Replace the shared-token block in `deploy/nats-1m.conf` with the account
   model and commit the account public JWT only.
3. Refactor `neuromorphic` `policy.restrict` → `policy.restrict.request`.
4. Red-team test: assert a non-Kernel credential is **refused** when it tries to
   publish `decision.*` / `policy.*`.

---

## References

- CLAUDE.md §3 (Safety architecture — Kernel is the sole decision authority).
- ROADMAP.md Phase 1 (decision bus not yet authenticated/signed).
- `sdk/src/activelearning/subjects.py` (subject registry).
- `sdk/src/activelearning/nats_client.py` (`EventBus`, JetStream, decision verify).
- Issue [#64](https://github.com/DPBG/Engram.AI/issues/64); decision signing
  [#36](https://github.com/DPBG/Engram.AI/pull/36).
</content>
