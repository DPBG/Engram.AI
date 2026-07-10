# Kernel Crash-Recovery Threat Model

This document threat-models what happens to an in-flight proposal if the
**Kernel process itself** dies mid-evaluation — a crash, an OOM kill, a
deploy restart, anything that ends the process, not just an exception
inside it. CLAUDE.md §3 makes the Kernel the sole ALLOW/TRANSFORM/DENY/DEFER
authority; a crash mid-decision is a gap in the "fails safe" guarantee if a
waiting caller doesn't correctly time out and fail closed.

Reference: `kernel/src/kernel/service.py` (`_handle_action_proposal`,
`_handle_code_proposal`, `_publish_and_log_decision`),
`sdk/src/activelearning/nats_client.py` (`EventBus.wait_for_decision`,
`EventBus.js_subscribe`), `launcher/watchdog.py` (E1.9.3),
`kernel/tests/test_kernel_crash_chaos.py` (M1.3, issue #186).

---

## Why an in-process exception handler isn't enough

`_handle_action_proposal` and `_handle_code_proposal` each wrap their body
in `try/except Exception`, and on any internal error publish a fail-safe
`DENY` so "the caller's Future doesn't hang." That's real coverage — see
`kernel/tests/test_service.py::test_code_proposal_publishes_fail_safe_deny_on_internal_error`
— but it only helps when the Kernel process is still *alive* to run the
`except` block. If the process itself dies (crash, OOM kill, restart), no
Python code runs at all: no `except`, no fail-safe publish, nothing. From a
decision-waiter's perspective, a **dead** Kernel and a **never-started**
Kernel are indistinguishable — no `decision.<trace_id>` message is ever
published, full stop. This document is about that case specifically.

---

## The three crash phases

### 1. Before evaluation (proposal received, not yet processed)

`proposal.new` and `code.proposal` are subscribed via
`EventBus.js_subscribe` — a **durable, explicit-ack** JetStream consumer.
Per its docstring: *"the message is acked exactly once only after the
handler returns. A consumer killed mid-processing never acks, so the
broker redelivers; the message is not lost."* If the Kernel dies before
(or during) `_handle_action_proposal`/`_handle_code_proposal`, the message
is simply redelivered to the next Kernel instance that attaches to the
same durable consumer name (`kernel-action-proposals` /
`kernel-code-proposals`) on restart. **No decision is lost — it's just
delayed until the Kernel comes back**, exactly as `js_subscribe`'s
docstring intends.

**Caller-side behavior:** the original waiter never received a decision
and is still blocked on `wait_for_decision(trace_id, timeout=...)`. It has
no idea the Kernel crashed; it just experiences elevated latency. If the
Kernel restarts and redelivers within the waiter's timeout window, the
waiter gets a correct, freshly-evaluated decision as if nothing happened.
If the Kernel is down longer than the waiter's timeout, the waiter times
out and fails closed (verified by
`test_kernel_crash_chaos.py::test_decision_waiter_fails_closed_when_kernel_dies_mid_evaluation`,
which freezes a real Kernel subprocess with SIGSTOP right before publishing
a real proposal so it can never be dequeued or acked, and confirms the
waiter's `wait_for_decision` raises promptly instead of hanging).

### 2. During evaluation (risk analysis / belief-norm queries / scoring in flight)

Same redelivery guarantee as above — the JetStream message still isn't
acked, since ack only happens after the handler *returns*. A crash here is
observationally identical to phase 1 from the waiter's perspective: no
decision published, eventual redelivery once a Kernel instance is running
again.

One consequence worth naming: `_get_risk_analysis()` and
`_check_belief_norms()` are themselves NATS request/reply calls to Safety
Supervisor and Beliefs (5.0s and 2.0s hardcoded timeouts respectively). If
the Kernel crashes mid-request, those in-flight requests are simply
abandoned along with the rest of the handler — no special handling needed,
since the whole evaluation restarts cleanly from scratch on redelivery.

### 3. After the decision is published, before the handler returns

This is the interesting one. `_publish_and_log_decision` (the action path)
does, **in this order**: publish the signed decision → deny-escalation
tracking / auto-disable-channel → forward-on-ALLOW → log to the SQLite
audit trail. `_handle_code_proposal` (the code path) does the *opposite*
order: log first, then publish. If the Kernel crashes between the publish
and the point where the JetStream handler function returns (and thus
acks):

- **The waiter already got its decision.** It already unsubscribed
  (`wait_for_decision`'s `finally: await sub.unsubscribe()` runs
  immediately once `decision_received.set()` fires) and has moved on —
  this is the common, benign case.
- **The `proposal.new`/`code.proposal` message was never acked**, so it
  gets redelivered on restart, and the Kernel evaluates the *same
  trace_id* a second time, publishing a **second** decision for it. This
  is a real, structural possibility of this design, not a bug — JetStream
  redelivery is at-least-once by design, and idempotent redelivery is the
  price of "no decision is ever silently lost." A second published
  decision for an already-resolved trace_id is inert: nothing is still
  listening on that specific durable waiter subject once the first
  decision resolved it, so the duplicate just ages out of the stream.
- **The audit-trail write can be lost.** On the action path specifically,
  a crash between "decision published" and "`_log_decision` runs" means a
  decision was delivered and (potentially) acted on, but has no SQLite
  record — despite `_publish_and_log_decision`'s docstring claiming it
  "combines publish + log in one call to prevent inconsistency." That
  claim holds against an *exception* (the surrounding `except` still runs
  `_log_decision` best-effort — see `_log_decision`'s own `try/except`,
  which never propagates), but not against a genuine process death between
  the two steps. This is an accepted gap, not a silent one: the audit
  trail is explicitly best-effort (`_log_decision` never blocks or retries
  on failure), and the redelivery-on-restart path above means the *next*
  evaluation attempt for the same trace_id, if one happens, does get
  logged. Closing it completely would need transactional publish+log
  (e.g. log first inside the same durable-consumer redelivery guarantee,
  which is closer to what the code path already does) — out of scope for
  this issue, flagged here for anyone taking on stronger audit-trail
  durability later.

---

## The two independent safety nets

1. **Per-caller timeout (verified here).** Every `wait_for_decision(trace_id,
   timeout=...)` caller fails closed on its own schedule, independent of
   why no decision arrived — dead Kernel, network partition, or a Kernel
   that's simply slow. This is the mechanism this issue's acceptance
   criteria asks to confirm, and `test_kernel_crash_chaos.py` now does so
   against a real killed process rather than by code inspection alone.
2. **System-wide kernel-loss watchdog (E1.9.3, pre-existing).**
   Independent of any specific in-flight proposal, `launcher/watchdog.py`
   subscribes to `kernel.heartbeat` (published every
   `KERNEL_HEARTBEAT_INTERVAL_S`, default 5s) and publishes `safety.halt`
   once no heartbeat has arrived for `KERNEL_WATCHDOG_TIMEOUT_S` (default
   15s — 3× the heartbeat interval). This doesn't resolve any
   *already-pending* `wait_for_decision` call (those still rely on their
   own timeout, safety net #1), but it stops the *system* from continuing
   to submit new proposals into a Kernel-shaped void while it's down,
   which is the broader protective effect a single caller's timeout can't
   provide on its own.

Together: an individual in-flight proposal always resolves (redelivery) or
times out (caller-side timeout) within a bounded window: the system as a
whole is also driven to `SAFE_HALT` within ~15s of Kernel loss regardless
of whether any proposal happens to be in flight at that moment.

---

## Verification

- `kernel/tests/test_kernel_crash_chaos.py` — real `nats-server` +
  `KernelService` subprocess, frozen with `SIGSTOP` before it can dequeue a
  real `proposal.new`, confirms `wait_for_decision` times out within its
  configured window (not a hang), then `SIGKILL`s it for cleanup. Mirrors
  `kernel/tests/test_safety_supervisor_chaos.py` (#187)'s approach and its
  rationale for `SIGSTOP`-then-`SIGKILL` over a timing-based kill.
- `kernel/tests/test_service.py::test_code_proposal_publishes_fail_safe_deny_on_internal_error`
  — the complementary in-process case: an exception during evaluation
  (Kernel alive) still publishes a fail-safe `DENY`.
