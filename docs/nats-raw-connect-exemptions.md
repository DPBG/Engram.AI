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
| `sdk/tests/test_base_service_migration.py` | Migration-parity regression template (see `docs/SDK-BASESERVICE-CONTRACT.md`); opens an independent raw probe client alongside `BaseService`/`EventBus` to observe behavior from outside, same white-box reasoning as the two rows above. |
| `sdk/scripts/bench_eventbus_publish.py` | Load-test script with a deliberate `raw_nats` scenario whose entire purpose is measuring `EventBus`'s overhead *against* a raw connection — it must bypass the wrapper to have something to compare it to. |
| `sdk/tests/red_team/test_broker_authz.py` | Red-team transport-authz regression (E1.1.9, ADR 0001 §3): connects with several different NATS identities (including a fabricated one) to prove the broker itself refuses non-Kernel privileged publishes. `EventBus` always connects as "this service" — testing *other* identities' permissions requires raw connects under those identities. |
| `neuromorphic/scripts/train_pump.py` | A standalone operator script, not a service, run outside the SDK's dependency graph. |
| `dashboard/src/dashboard/api.py` | Dashboard is a standalone FastAPI + vanilla-JS app with no SDK dependency by design (see `CLAUDE.md` §6: "Dashboard: standalone FastAPI + vanilla JS (no SDK dependency)"). |
| `dashboard/src/dashboard/nats_stream.py` | The dashboard's dedicated NATS-connection module — same "no SDK dependency by design" reasoning as `api.py`, just factored into its own file. |
| `deploy/gateway_restart_listener.py` | An infrastructure script that runs alongside the gateway tmux session on the deploy host, not a microservice — it has no `activelearning` install available. |
| `launcher/watchdog.py` | The Kernel-loss watchdog (E1.9.3) must be able to start even if the rest of the codebase — including the SDK — is broken, so it deliberately keeps zero import-time dependency on `activelearning` (subject strings are duplicated locally rather than imported from `activelearning.subjects.Subjects`; see the module's own comment). |
| `launcher/tests/test_watchdog_chaos.py` | Chaos test for the watchdog above: needs independent raw clients to play the heartbeat publisher, an observer, and the watchdog itself as separate NATS connections whose reconnect timing is manipulated individually — `EventBus` doesn't expose that level of connection control. |
| `test-runner/src/test_runner/tests/conftest.py` | The integration test harness (`test-runner/`) intentionally depends only on `nats-py`, not on `activelearning` — it black-box tests the real services (including `EventBus` itself) over the wire, so it must not share the library it's validating. |
| `test-runner/src/test_runner/tests/test_jetstream_durability.py` | Same harness, same reasoning — asserts on raw JetStream redelivery/DLQ behavior independent of `EventBus`'s own bookkeeping. |
| `tests/red_team/test_broker_rejects_privileged_publish.py` | Second red-team transport-authz suite (top-level `tests/red_team/`, distinct from the `sdk/tests/red_team/` one above) — same multi-identity rationale: proves the broker rejects non-Kernel privileged publishes at the transport layer, which requires connecting as those other identities directly. |

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
