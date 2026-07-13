# Unforgeable Gate Threat Model

> Scope: the **Kernel decision gate** — the end-to-end path by which a proposal
> (action or code) is evaluated and an `ALLOW / TRANSFORM / DENY / DEFER`
> decision reaches the service that must obey it, plus the operator-command
> path (`safety.halt` / `safety.resume`) that can override it. This document
> consolidates what was previously spread across
> [ADR 0001](adr/0001-nats-authz.md), [CLAUDE.md §3](../CLAUDE.md#3-safety-architecture-treat-as-critical),
> and PR #194's NATS-scope audit.
> Tracking issue: [#212](https://github.com/DPBG/Engram.AI/issues/212) (M1.19).

## 1. What we are defending against

CLAUDE.md's governing invariant is: **every action and code proposal passes
through the Kernel, which is the sole authority that may emit a decision.**
Engram can generate and run its own code (`meta-programmer/`), and every
service communicates over open NATS pub/sub — so a compromised, buggy, or
adversarially-generated non-Kernel process sharing the bus is an in-scope
attacker, not a hypothetical one.

### Assets to protect

- **The decision itself** — `decision.<trace_id>` / `code.decision.<trace_id>`.
  A forged `ALLOW` lets an unsafe action or unsafe generated code run as if the
  Kernel approved it.
- **The kill switch** — `safety.halt` / `safety.resume`. A forged `resume`
  silently un-halts a system an operator believed was stopped; a forged `halt`
  is a denial-of-service against the system (lower severity — fails safe, not
  unsafe).
- **The privileged policy/cognitive subjects** the Kernel alone may publish
  (`policy.*`, `cognitive.response.validated`) — see §3 below.

### Attacker model

- **Can:** run arbitrary code as any non-Kernel Engram service (the
  Meta-Programmer's whole job is to deploy LLM-generated code); connect to the
  NATS bus with whatever credentials that compromised service holds; observe
  all bus traffic the dashboard's broad subscribe grant would also see.
- **Cannot (in scope):** obtain the Kernel's own signing secrets
  (`ENGRAM_DECISION_KEY`, `ENGRAM_OPERATOR_KEY`) or NATS credentials without a
  *separate* host/credential-store compromise — that is a different, larger
  threat model (host security) and out of scope here.
- **Out of scope:** compromise of the NATS server binary/host itself, and
  supply-chain compromise of the `nats-py` / `nats-server` dependencies.

Two independent layers defend this path — **neither replaces the other**:

| Layer | Mechanism | Defends against |
|---|---|---|
| Transport (bus) | Per-identity NATS publish/subscribe allowlists | A compromised service *publishing on a subject it has no business touching*, before any application code runs. |
| Application (payload) | HMAC-SHA256 signing of decisions and operator actions | A message that *did* reach a subscriber being accepted as authentic — covers the case where transport authz is misconfigured, absent (dev mode), or the forger already holds a legitimately-scoped credential for some other reason. |

## 2. Transport layer — NATS subject authorization

Full design rationale and the complete subject × service matrix live in
[ADR 0001](adr/0001-nats-authz.md); this section summarizes the threat-relevant
rule and its **current implementation state**, which the ADR (still "Proposed")
does not track.

**Rule (ADR 0001 §3):** only the `kernel` identity may publish `decision.>`,
`code.decision.>`, `policy.*`, and `cognitive.response.validated`. Every other
identity's allowlist omits them; new subjects default to denied, not allowed
(deny-by-default, ADR 0001 §5).

### What's actually shipped today (two parallel implementations)

| Runtime | Credential model | Config | Status |
|---|---|---|---|
| `run.py` (local dev) | Decentralized NKEY/JWT (ADR 0001 §2 target model) | `deploy/scripts/gen-creds.sh` generates per-service `.creds` via `nsc`; `.localrun/nats/resolver.conf` | Implemented. Falls back to the permissive `dev_default` user (see below) when creds haven't been generated. |
| Docker Compose / Hetzner (`deploy/nats-1m.conf`) | Static per-user `password` block (ADR 0001 §2 "transitional" variant, explicitly sanctioned) | Per-service `*_NATS_PASS` env vars | Implemented and enforced — the privileged subjects are denied even to the anonymous `dev_default` fallback user (belt-and-suspenders). |

Both configs implement the same matrix; they differ only in credential
mechanism, per the ADR's explicit allowance for a transitional static-users
config to ship ahead of the full decentralized model.

### Verification

`sdk/tests/red_team/test_broker_authz.py` starts a real, isolated
`nats-server` per test module (not mocked — a mock cannot reject an
unauthorized publish; only the broker can) and asserts:

| Test | Asserts |
|---|---|
| `test_non_kernel_publish_privileged_is_broker_rejected` | A non-`kernel` identity's publish to a privileged subject is refused **by the broker**, never reaching a subscriber. |
| `test_kernel_can_publish_all_privileged_subjects` | The `kernel` identity *can* publish every privileged subject (the allowlist isn't accidentally over-restrictive). |
| `test_unknown_identity_cannot_connect` | A credential not in the config cannot connect at all. |

## 3. Application layer — decision & operator-action signing

Implemented in `sdk/src/activelearning/signing.py`; both signers use the same
pattern (HMAC-SHA256 over a canonical JSON subset of security-relevant fields,
`hmac.compare_digest` for constant-time verification, fail-safe-by-configuration).

### 3a. Decision signing (`ENGRAM_DECISION_KEY`)

- **Signed fields:** `trace_id`, `type`, `risk_score`, `expires_at`,
  `issued_at`. Every Kernel decision publish path
  (`_publish_and_log_decision`, `_signed_code_decision`, the internal-error
  fail-safe DENY paths in `_handle_action_proposal` /
  `_handle_code_proposal`) signs through `sign_decision` before publishing —
  there is no unsigned decision-publish code path.
- **Verification:** `EventBus.wait_for_decision` calls `verify_decision` on
  every message before resolving the waiter. A message that fails to verify is
  **ignored, not rejected-with-error** — it cannot satisfy the wait, so the
  caller times out and must fail closed (deny/halt), never fail open.
- **Fail-safe-by-configuration:** when `ENGRAM_DECISION_KEY` is set, a
  missing/invalid signature is rejected. When unset, verification passes
  unconditionally (legacy/dev mode) with a one-time warning — this is the
  "decision bus not yet authenticated" state CLAUDE.md §3 still warns about
  when the key is absent; it is opt-in hardening, not yet mandatory.

### 3b. Operator-action signing (`ENGRAM_OPERATOR_KEY`)

- Covers `safety.resume` (and is the pattern to extend to other privileged
  operator commands as they're added).
- **Signed fields:** `operator_id`, `action`, `timestamp`. Binding `action`
  into the signature means a captured `halt` signature cannot be replayed as a
  `resume` — the two commands are cryptographically distinct, not just
  distinguished by subject name.
- **Replay window:** `timestamp` must be within
  `OPERATOR_TIMESTAMP_TOLERANCE_MS` (5 minutes) of now, checked at verify time
  in *addition* to the signature — an old, validly-signed `resume` cannot be
  replayed after that window even by someone who captured it off the bus.
- **Fail-closed checks, in order:** missing/zero timestamp → reject; timestamp
  outside tolerance → reject; missing/invalid signature → reject. Same
  fail-safe-by-configuration rule as decision signing when the key is unset.
- **Verified by** `kernel/tests/test_safe_halt_resume_auth.py` (dev-mode
  accept, signed-payload accept, missing/wrong-key/tampered/stale/future
  timestamp reject, cross-action replay reject, resume-releases-halt,
  rejected-resume-still-published).

### Residual risk — decision replay

Decision signing authenticates *origin and content integrity*, not recency: a
captured, validly-signed decision could in principle be replayed for the
**same `trace_id`** within its consumer's delivery window, and `expires_at` is
signed but not independently checked against wall-clock time by
`wait_for_decision` (unlike the operator-action path, which does check
`timestamp` at verify time). In practice this is low-severity because
`trace_id`s are single-use UUIDs generated per proposal and each waiter
subscribes to one specific trace's subject and resolves on first delivery —
there is no second consumer for an attacker to replay into. Tightening this
(e.g. an explicit `issued_at` freshness check mirroring the operator-action
path) is worth doing but is not yet implemented; tracked as follow-up below.

## 4. What is NOT protected (explicitly out of scope)

- **Bus transport encryption.** Neither config in §2 configures TLS between
  services and the broker; on a shared/untrusted network, a passive observer
  could read (but, per §2, not forge) bus traffic. Mitigated operationally by
  running the broker on a private network in production.
- **Key distribution/rotation.** `ENGRAM_DECISION_KEY` and
  `ENGRAM_OPERATOR_KEY` are shared secrets set via environment variable; there
  is no automatic rotation. Compromise of either key requires a separate
  host/secret-store breach (out of scope per the attacker model in §1) but,
  once compromised, is not detected or auto-revoked.
- **The dashboard's broad `>` subscribe grant.** Intentional (drives the live
  UI) and read-only — see ADR 0001 "Consequences" — but it does mean the
  operator credential can observe all bus traffic.
- **Everything in [SANDBOX-THREAT-MODEL.md](SANDBOX-THREAT-MODEL.md).** Code
  containment (network/filesystem/capability isolation for LLM-generated code
  under test) is a separate boundary from the message-authenticity boundary
  documented here.

## 5. Follow-up

1. Decision replay hardening: an explicit freshness check on `issued_at` at
   `wait_for_decision` verify time (§3a residual risk).
2. Complete the ADR 0001 target state: retire the transitional static-users
   `nats-1m.conf` config in favor of the decentralized NKEY/JWT model already
   used by `run.py`, so both runtimes share one credential mechanism.
3. Extend operator-action signing (§3b) to the remaining dashboard-originated
   privileged commands beyond `safety.resume`, as they're identified.

## References

- [ADR 0001 — NATS account & permission model](adr/0001-nats-authz.md) — full
  subject × service matrix and design rationale.
- [CLAUDE.md §3](../CLAUDE.md#3-safety-architecture-treat-as-critical) — the
  governing safety invariant this document defends.
- [SANDBOX-THREAT-MODEL.md](SANDBOX-THREAT-MODEL.md) — the sibling threat
  model for code containment.
- `sdk/src/activelearning/signing.py` — decision and operator-action signing.
- `sdk/tests/red_team/test_broker_authz.py` — transport-layer enforcement tests.
- `kernel/tests/test_safe_halt_resume_auth.py` — operator-action signing tests.
- `sdk/tests/test_signing.py` — decision signing unit + latency-budget tests.
- Issue [#212](https://github.com/DPBG/Engram.AI/issues/212) (M1.19); PR #194
  (NATS-scope audit that surfaced the `policy.restrict` conflict resolved in
  ADR 0001 §3).
