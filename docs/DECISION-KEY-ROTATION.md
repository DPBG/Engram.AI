# Decision-Bus Signing Key Rotation Runbook

Task 1.2 added HMAC decision-bus signing (`ENGRAM_DECISION_KEY`,
`sdk/src/activelearning/signing.py`) so a forged `decision.<trace_id>`
can't beat the real Kernel (CLAUDE.md §3). Until now, no rotation procedure
existed anywhere — the key was referenced only in ADR/architecture docs,
never operationally. A signing key with no rotation procedure is a
long-lived secret by default: if it's ever compromised, there was no
documented, tested path to rotate it without either an outage (verifiers
briefly unable to verify anything) or a window of forged-decision risk
(rotating everywhere at once with no fallback).

Reference: `sdk/src/activelearning/signing.py` (`verify_decision`,
`DECISION_KEY_SECONDARY_ENV`), `sdk/tests/test_signing.py`,
`sdk/tests/test_nats_client.py` (issue #206).

---

## The mechanism: a verify-only secondary key

`verify_decision` accepts a signature made with **either** of two keys:

- `ENGRAM_DECISION_KEY` — the primary key. This is the **only** key
  `sign_decision` ever signs with.
- `ENGRAM_DECISION_KEY_SECONDARY` — optional, verify-only. Never used for
  signing; exists solely so a verifier can accept decisions signed with a
  second key during rotation.

This is a **verify-only** dual-key window, not dual-signing: at any moment
exactly one key is producing signatures (the Kernel's `ENGRAM_DECISION_KEY`),
but every verifier (every service that calls `EventBus.wait_for_decision` —
currently: `planner`, `neuromorphic`, `overrides`, `external-api`,
`coordinator`, `meta-programmer`) can be configured to accept two keys at
once while the rollout is in flight.

---

## Why the rollout order matters

A decision's `expires_at` (issue #190) bounds how long a signed decision
stays acceptable at all — **5 minutes**, the longest of the Kernel's two
TTLs (`KernelEvaluator.decision_ttl_ms` = 1 min, `defer_ttl_ms` = 5 min,
`kernel/src/kernel/evaluator.py`). That bound is what makes a *timed*
overlap window safe: once it's passed, no decision signed before the
Kernel's key flip can still be validly in flight — one would need to
survive delivery/redelivery *and* still have an unexpired signature, and
the expiry check rejects the second half of that regardless of which key
signed it.

The unsafe failure mode this procedure avoids: flipping `ENGRAM_DECISION_KEY`
on the Kernel to a new value **before** every verifier can accept that new
value would make every decision published in that gap unverifiable —
verifiers reject it as a bad signature, callers time out, and everything
fails closed (safe, but a real outage, not a graceful rotation).

---

## Procedure

Generate a new key first: `openssl rand -hex 32` (or your secrets
manager's equivalent). Call it `K_new`; call the current key `K_old`.

### Step 1 — Roll out `K_new` as the secondary key everywhere

On **every node that verifies decisions** (any service calling
`wait_for_decision` — see the list above) *and* the Kernel itself:

```bash
export ENGRAM_DECISION_KEY_SECONDARY=$K_new
# ENGRAM_DECISION_KEY is untouched — still $K_old
```

Restart/redeploy each node with this change. Signing behavior is
unchanged (still `K_old`); every verifier can now additionally accept
`K_new`, but nothing signs with it yet.

**Do not proceed to Step 2 until every node has confirmed this rollout.**
This is the step that makes the whole procedure safe — it's the one
`test_wait_for_decision_overlap_window_accepts_both_keys` exercises.

### Step 2 — Flip the Kernel to sign with `K_new`

On the Kernel only:

```bash
export ENGRAM_DECISION_KEY=$K_new
export ENGRAM_DECISION_KEY_SECONDARY=$K_old
```

(Swapping `K_old` into the Kernel's own secondary slot costs nothing and
means the Kernel itself would still accept a `K_old`-signed message on any
subject it happens to verify, though today the Kernel only *signs*
decisions, it doesn't verify its own.)

Restart the Kernel. From this point, every new decision is signed with
`K_new`. Every verifier (updated in Step 1) accepts it via their secondary
slot. Any decision signed with `K_old` moments before this restart is
still accepted too, via each verifier's primary key — nothing is dropped.

### Step 3 — Wait out the overlap window

Wait at least **30–60 minutes** after Step 2 before proceeding — comfortably
longer than the 5-minute maximum decision TTL plus a generous margin for
confirming Step 1's rollout genuinely reached every node (don't start the
clock until you're sure of that). There is no benefit to rushing this step;
the only cost of waiting longer is `K_old` remaining a valid secondary
key for a bit longer.

### Step 4 — Retire `K_old` everywhere

On **every node** (Kernel and all verifiers):

```bash
export ENGRAM_DECISION_KEY=$K_new
unset ENGRAM_DECISION_KEY_SECONDARY   # or remove the env var entirely
```

Restart/redeploy. `K_old` is no longer accepted anywhere — rotation is
complete. This is what
`test_wait_for_decision_retires_old_key_after_rotation_completes` proves:
a decision signed with a retired key is rejected, not grandfathered in
forever.

---

## Rollback

If something goes wrong **before Step 4**, rollback is a no-op: `K_old` is
still the accepted primary (or, after Step 2, still an accepted secondary)
everywhere, so simply reverting `ENGRAM_DECISION_KEY`/`_SECONDARY` to their
pre-rotation values on any node that was updated out of order restores the
previous state with no verification gap.

If a mistake is discovered **after Step 4** (e.g. `K_new` itself needs to
be replaced, or was rotated to in error), treat it as a **new rotation**:
run this procedure again from Step 1 with a fresh key. Never reintroduce a
previously-retired key — an attacker who captured it during its retirement
window would regain forging capability.

---

## Verification

- `sdk/tests/test_signing.py` — unit-level: `verify_decision` accepts a
  signature made with either the primary or secondary key and rejects one
  made with neither; `sign_decision` never uses the secondary key; removing
  the secondary key retires it.
- `sdk/tests/test_nats_client.py` — end-to-end over a real embedded
  `nats-server`: a decision signed with the old key and one signed with the
  new key, published while a verifier has both configured (the Step 1→3
  state), are **both** accepted by `wait_for_decision`
  (`test_wait_for_decision_overlap_window_accepts_both_keys`); after the
  secondary key is removed (the Step 4 state), a decision signed with the
  now-retired key is rejected — the waiter times out rather than accepting
  it (`test_wait_for_decision_retires_old_key_after_rotation_completes`).

## Out of scope

Operator-action signing (`ENGRAM_OPERATOR_KEY`, `verify_operator_action` —
used for `safety.resume` and similar privileged commands) has no rotation
support yet. It's a structurally similar secret with no documented rotation
path either, but is a separate signing domain from decision-bus signing and
outside this issue's scope — worth a follow-up issue if it needs one.
