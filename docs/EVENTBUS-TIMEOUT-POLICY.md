# EventBus Request-Reply Timeout Policy

> **Milestone:** M2 — Reliable Backbone
> **Issue:** [M2.10] Formalize the EventBus request-reply timeout policy across all services (#233)
> **Status:** Canonical — all EventBus timeout usage MUST conform to this document.

Request-reply and decision-wait timeouts on the `EventBus`
([`sdk/src/activelearning/nats_client.py`](../sdk/src/activelearning/nats_client.py))
used to be hardcoded ad hoc at each call site (`30.0`, `10.0`, `5.0`, and
caller-supplied values with no shared default), which made failure behavior
inconsistent and hard to reason about across services. This document defines the
single source of truth for those timeouts and how to override them.

---

## 1. Centralized defaults

All EventBus timeouts derive from module-level constants in `nats_client.py`.
Nothing in the client hardcodes a timeout literal anymore; each constant is
resolved once at import time from an environment variable, falling back to a
documented default.

| Constant | Env override | Default | Applies to |
|----------|--------------|---------|------------|
| `DEFAULT_REQUEST_TIMEOUT_S` | `ENGRAM_REQUEST_TIMEOUT_S` | `30.0` s | `EventBus.request()` request-reply RPCs |
| `DEFAULT_DECISION_TIMEOUT_S` | `ENGRAM_DECISION_TIMEOUT_S` | `30.0` s | `EventBus.wait_for_decision()` Kernel gate waits |
| `RECONNECT_WAIT_TIMEOUT_S` | `ENGRAM_RECONNECT_WAIT_TIMEOUT_S` | `10.0` s | `_ensure_connected()` wait for auto-reconnect |
| `CONNECTION_DRAIN_TIMEOUT_S` | `ENGRAM_CONNECTION_DRAIN_TIMEOUT_S` | `5.0` s | draining a dead connection in `force_reconnect()` |

All four constants are also re-exported from the `activelearning` package for
discoverability:

```python
from activelearning import DEFAULT_REQUEST_TIMEOUT_S, DEFAULT_DECISION_TIMEOUT_S
```

---

## 2. Override mechanism

There are two override levels, in increasing precedence:

### 2.1 Deployment-wide (environment variable)

Set the relevant env var before the process starts. This changes the default for
**every** call in that process that does not pass an explicit timeout:

```bash
# e.g. a slow/loaded deployment that needs more headroom
export ENGRAM_REQUEST_TIMEOUT_S=45
export ENGRAM_DECISION_TIMEOUT_S=45
```

Validation: a missing, non-numeric, or non-positive value logs a warning and
falls back to the built-in default. A timeout can therefore never be silently set
to `0` or a negative number (which would make a call hang forever or fail
instantly).

### 2.2 Per-call (explicit argument)

Pass `timeout=<seconds>` to override a single call. Reserve this for calls whose
latency profile genuinely differs from the default — e.g. the Kernel's fast
safety-analysis probes intentionally use short timeouts so the gate fails closed
quickly:

```python
# Kernel: a deliberately short probe (documented deviation from the default)
resp = await self.event_bus.request(subject, payload, timeout=2.0)
```

`timeout=None` (the default value of the argument) means "use the policy
default". Prefer `None` over restating `30.0` at a call site so a deployment-wide
override actually takes effect.

---

## 3. Failure semantics

- `EventBus.request()` raises `asyncio.TimeoutError` when no reply arrives in
  time. Callers should catch it and degrade safely (retry, return an error, or
  fail closed depending on context).
- `EventBus.wait_for_decision()` raises `asyncio.TimeoutError` on timeout. Callers
  **MUST fail closed** (deny / halt) — never default-allow. This is a safety
  invariant (see [CLAUDE.md](../CLAUDE.md) §3 and
  [`docs/SDK-BASESERVICE-CONTRACT.md`](SDK-BASESERVICE-CONTRACT.md) §3.10).

---

## 4. Guidance for call sites

- **Default first.** Call `request()` / `wait_for_decision()` without a `timeout`
  argument unless you have a specific reason not to.
- **Document deviations.** If you pass an explicit timeout, add a short comment
  explaining why the default is inappropriate (e.g. "fast fail-closed probe").
- **Tune per deployment, not per call.** If a whole environment needs a different
  budget, set the env var rather than editing many call sites.
- **Never pass `0` or a negative timeout.** Use a small positive value if you need
  a near-immediate check.

---

*Document owner: SDK / backbone team.*
