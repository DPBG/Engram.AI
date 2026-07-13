# JetStream Durable-Consumer Naming Convention

No documented scheme existed for JetStream durable consumer names
(`EventBus.js_subscribe`'s `durable` parameter, `sdk/src/activelearning/nats_client.py`).
Two services — or two instances of the same service — choosing the same
durable name would silently share/steal each other's delivery cursor: the
broker treats a durable name as a stable identity, so a second subscriber
attaching with an existing durable name joins (or corrupts, if the consumer
config differs) that consumer's position in the stream rather than getting
its own. This document is the convention, plus the audit issue #255 asked
for.

Reference: `sdk/src/activelearning/nats_client.py` (`EventBus.js_subscribe`,
`EventBus.wait_for_decision`), `kernel/src/kernel/service.py`,
`sdk/tests/test_durable_naming_convention.py` (issue #255).

---

## The convention

There are two distinct shapes of durable consumer in this codebase, and
each gets its own naming rule.

### 1. Static, long-lived consumers — `<service>-<purpose>`

A service that calls `js_subscribe()` directly at startup (today: only the
Kernel, on `proposal.new` and `code.proposal`) uses a **stable, kebab-case
name of the form `<service>-<purpose>`**:

- `<service>` — the calling service's directory/registry name exactly as it
  appears in `launcher/registry.py` (`kernel`, `beliefs`, `coordinator`,
  `overrides`, `meta-programmer`, ...). This is what makes collisions
  between *different* services structurally impossible: two services can
  never produce the same prefix.
- `<purpose>` — a short, kebab-case description of what's being consumed
  (`action-proposals`, not `proposals` — specific enough that a second
  durable added later for a different purpose in the same service can't
  collide either).

Current durables of this shape (from `sdk/tests/test_durable_naming_convention.py`'s
audit — this table is kept in sync with the code by that test):

| Durable name | Service | Subject | Purpose |
|---|---|---|---|
| `kernel-action-proposals` | kernel | `proposal.new` | Action-proposal evaluation queue |
| `kernel-code-proposals` | kernel | `code.proposal` | Code-proposal evaluation queue |

No collisions found. If a service needs a second static durable, extend the
`<purpose>` suffix (e.g. a hypothetical second Kernel consumer would be
`kernel-<something-else>`, never a second `kernel-action-proposals`).

### 2. Ephemeral, per-request consumers — `waiter-<type>-<trace_id>`

`EventBus.wait_for_decision()` is called by *any* service waiting on a
specific Kernel decision (`overrides`, `external-api`, `coordinator`,
`meta-programmer`, `planner`, `neuromorphic`, ...) — it is not a
single-owner consumer the way the Kernel's proposal queues are. Its durable
name is generated dynamically:

```python
durable = f"waiter-{'code' if code else 'action'}-{trace_id}"
```

This deliberately does **not** include the calling service's name. Two
different services waiting on the same `trace_id` at the same time would
collide — but `trace_id` is a UUID4 (`generate_trace_id()`) minted once per
proposal, and in normal operation only the service that originated the
proposal waits on its own decision, so this doesn't happen in practice.
Uniqueness here comes from the trace_id, not the caller's identity — that's
the intentional exception to rule 1, not an oversight.

These consumers are also self-cleaning: `js_subscribe`'s
`inactive_threshold` (`_CONSUMER_INACTIVE_THRESHOLD_S` = 60s) means the
broker prunes them automatically once the waiter stops polling, so they
don't accumulate the way a mis-named static consumer would.

---

## Adding a new static durable

1. Name it `<service>-<purpose>` per the rule above.
2. Run `grep -rn 'durable\s*=\s*"' --include="*.py" .` first — confirm
   nothing else already uses that exact string.
3. Add a row to the table in this document.
   `sdk/tests/test_durable_naming_convention.py` fails CI if a durable
   literal in the code isn't documented here, so this step isn't optional.

---

## Verification

`sdk/tests/test_durable_naming_convention.py`, mirroring
`sdk/tests/test_adr_subject_matrix.py`'s doc-vs-code sync pattern:

- Scans every `*.py` file in the repo for `durable="..."` literal
  arguments (the static-consumer case; the dynamic `waiter-*` pattern is
  checked separately since it's an f-string, not a literal).
- Asserts every literal found is globally unique (the collision this issue
  is about).
- Asserts every literal matches `<service>-<purpose>` with `<service>`
  drawn from the real service list in `launcher/registry.py` (plus
  `meta-programmer`, which `launcher/registry.py` intentionally omits —
  see `deploy/scripts/gen-creds.sh`'s comment on why).
- Asserts every literal found in code is documented in this file's table,
  and vice versa — so the table above can't silently drift from the code.
