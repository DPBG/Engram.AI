# Raw `nats.connect()` exemption list

Almost every process in this repo should reach NATS through
`activelearning.EventBus` (`sdk/src/activelearning/nats_client.py`), not by
calling `nats.connect()` / `nats.aio.client.Client().connect()` directly.
`EventBus` wraps the raw client with JetStream stream/consumer setup,
decision signing and verification, reconnection policy, and the DLQ monitor —
bypassing it silently loses all of that.

A small number of files have a genuine, reviewed reason to connect raw. This
document is the source of truth for that list; `sdk/tests/test_raw_nats_connect_exemptions.py`
enforces it in CI in both directions:

- a **new** raw connect call site anywhere else in the repo fails the build;
- an exemption that no longer contains a raw connect call must be removed
  from the list (so it can't silently rot into an aspirational allowlist).

## The list

| File | Why it's exempt |
|---|---|
| `sdk/src/activelearning/nats_client.py` | This **is** `EventBus` — the canonical wrapper around `nats.connect()`. Everything else's compliance is defined relative to this file. |
| `sdk/tests/test_event_bus_validation.py` | SDK's own white-box tests, which connect a second raw client alongside `EventBus` to observe wire-level behavior (e.g. that invalid messages are rejected) that `EventBus` itself can't easily assert on itself. |
| `sdk/tests/conftest.py` | SDK test fixture for the same reason — needs a raw client independent of the `EventBus` instance under test. |
| `neuromorphic/src/neuromorphic/cognitive_bridge.py` | `neuromorphic/` does not depend on the `activelearning` SDK (kept installable standalone — see the comment on `CognitiveBridgeService._llm`), so it cannot import `EventBus`. |
| `neuromorphic/scripts/train_pump.py` | Same package, same constraint — a standalone operator script, not a service, run outside the SDK's dependency graph. |
| `dashboard/src/dashboard/api.py` | Dashboard is a standalone FastAPI + vanilla-JS app with no SDK dependency by design (see `CLAUDE.md` §6: "Dashboard: standalone FastAPI + vanilla JS (no SDK dependency)"). |
| `deploy/gateway_restart_listener.py` | An infrastructure script that runs alongside the gateway tmux session on the deploy host, not a microservice — it has no `activelearning` install available. |
| `test-runner/src/test_runner/tests/conftest.py` | The integration test harness (`test-runner/`) intentionally depends only on `nats-py`, not on `activelearning` — it black-box tests the real services (including `EventBus` itself) over the wire, so it must not share the library it's validating. |
| `test-runner/src/test_runner/tests/test_jetstream_durability.py` | Same harness, same reasoning — asserts on raw JetStream redelivery/DLQ behavior independent of `EventBus`'s own bookkeeping. |

## Adding a new exemption

Adding a file to this list is a reviewed, deliberate decision, not a way to
silence a failing test:

1. Add a row above explaining *why* the file cannot or should not use
   `EventBus`.
2. Add the file's repo-relative path to `APPROVED_RAW_CONNECT_FILES` in
   `sdk/tests/test_raw_nats_connect_exemptions.py`.

If you're not sure whether a new raw connect belongs here, it probably
doesn't — use `activelearning.EventBus`/`BaseService` instead (see
`CLAUDE.md` §6).
