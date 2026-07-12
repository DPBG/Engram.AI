# SDK BaseService Migration Contract

> **Milestone:** M2 — Reliable Backbone  
> **Issue:** [E2.1.1] Define the BaseService migration contract + parity checklist  
> **Status:** Canonical — migrations in E2.1.2–E2.1.5 MUST conform to this document.

This document is the normative reference for `BaseService` and `EventBus` behaviour.
It defines what every adopting service can rely on, what it must not assume, and what
gaps require workarounds or separate fixes.

---

## 1. Scope

| In scope | Out of scope |
|----------|-------------|
| `BaseService` lifecycle contract | Neuromorphic SNN logic |
| `EventBus` (pub/sub, request-reply, JetStream) | Dashboard (standalone FastAPI, no SDK) |
| `ServiceConfig`, `Database` singleton | `sensory-gateway/` flat-layout service |
| Decision signing / verification | Kernel internals |
| Adoption checklist for migrating services | Actual per-service migrations (E2.1.2–E2.1.5) |

---

## 2. BaseService Lifecycle Contract

### 2.1 Initialisation (`__init__`)

Guaranteed at `__init__` time (synchronous, no I/O):

- `self.service_name` — set to the `service_name` argument.
- `self.config` — a `ServiceConfig` populated from environment variables (see §5).
- `self.logger` — a `logging.Logger` named after the service. Logging is configured immediately.
- `self._shutdown_event` — an `asyncio.Event` initialised to **clear** (not set).
- `self.event_bus` and `self.database` — **`None`** until `start()` is awaited.

**Nothing performs network I/O or creates coroutines in `__init__`.**

### 2.2 Start sequence (`start`)

`await service.start()` executes in this exact order and completes before returning:

1. If `use_event_bus=True`: create `EventBus`, call `EventBus.connect()`.  
   On success `self.event_bus` is connected and `_connected` is set.
2. If `use_database=True`: call `get_database()` (initialises the singleton if needed).  
   `self.database` is the shared `Database` instance.
3. Call `await self._setup()` (service extension point).

If any step raises, the exception propagates out of `start()`. The caller (`run()`)
catches it, logs it, and still calls `stop()` in the `finally` block.

### 2.3 Extension points

| Method | When called | Guarantee |
|--------|-------------|-----------|
| `async _setup(self)` | After event bus + DB are ready, inside `start()` | `self.event_bus` and `self.database` are non-`None` (subject to `use_*` flags). |
| `async _cleanup(self)` | First thing inside `stop()`, before bus/DB teardown | The event bus is still open; the service may `unsubscribe` or flush outstanding work. |

Both methods have a **no-op default**. Services override them; they must not call
`super()._setup()` / `super()._cleanup()` (no super implementation to compose with).

### 2.4 Run loop (`run`)

`await service.run()` is the **main entry point** and blocks until shutdown:

1. Registers `signal.SIGTERM` and `signal.SIGINT` handlers (silently skips on Windows).
2. Calls `start()`.
3. Awaits `_shutdown_event`.
4. In a `finally` block: calls `stop()`.

Exceptions raised by `start()` propagate after `stop()` runs; exceptions raised by
`stop()` are **not** suppressed.

### 2.5 Graceful shutdown

To initiate shutdown from inside the service (e.g. a fatal error handler):

```python
self.shutdown()  # synchronous; sets _shutdown_event
```

`run()` will unblock and call `stop()`. The OS signal handler calls `shutdown()` too.

**The shutdown sequence inside `stop()` is:**

1. `await self._cleanup()` — service drains / unsubscribes.
2. `await self.event_bus.close()` — NATS drain + close, subscriptions cleared.
3. Database: **not closed**. The database is a process-level singleton; it closes when
   the process exits. See §4.3 for testing implications.

---

## 3. EventBus Guarantees

### 3.1 Connection lifecycle

| State | Meaning |
|-------|---------|
| Before `connect()` | `_nc` is `None`; all operations raise `RuntimeError`. |
| After `connect()` | `_connected` event is set; `is_connected` returns `True`. |
| Transient disconnect | `_connected` is cleared; `_ensure_connected()` waits up to 10 s for auto-reconnect. |
| After `close()` | `_nc`, `_js`, subscriptions, and handlers are all cleared. |

`connect()` is idempotent: calling it on an already-connected bus is a no-op.

### 3.2 Automatic reconnection

The NATS client is created with `max_reconnect_attempts=-1` (unlimited). On each
disconnect/reconnect cycle:

- `_disconnected_callback` clears `_connected`.
- `_reconnected_callback` sets `_connected`.

**Subscriptions survive automatic reconnection** — nats-py re-registers them on the
new connection internally.

Error messages are throttled: the same error type is logged at most once every 15 s
to prevent log flooding during sustained outages.

### 3.3 `_ensure_connected()` — publish-side protection

Every `publish()`, `subscribe()`, `request()`, and `js_subscribe()` call
`_ensure_connected()` before touching the NATS client:

- If already connected: returns immediately.
- If reconnecting: waits up to **10 s** for `_connected` to be set.
- If still not connected after 10 s: raises `RuntimeError("Not connected to NATS")`.

This means a brief outage (< 10 s) is transparently absorbed; a sustained outage
surfaces as a `RuntimeError` to the caller.

### 3.4 `force_reconnect()` — manual recovery

`nats-py`'s built-in reconnection can fail silently in some edge cases
(`is_connected` returns `False` but no reconnect fires). Services may call
`await self.event_bus.force_reconnect()` to manually tear down and recreate the
connection. This method:

1. Saves all current subscription handlers (core and JS durable names).
2. Closes the dead connection (best-effort, 5 s timeout).
3. Calls `connect()` (creates a fresh client, re-ensures the safety JetStream stream).
4. Re-registers all saved handlers via `subscribe()` / `js_subscribe()`.

`force_reconnect()` is safe to call on a healthy connection; it becomes an
uneventful reconnect. Services that implement their own health-monitoring loop may
call it on detecting prolonged `is_connected == False`.

### 3.5 Publish semantics

```python
await self.event_bus.publish(subject, data)
```

| Subject type | Transport | Persistence |
|---|---|---|
| `proposal.new`, `code.proposal` | JetStream (`SAFETY_CRITICAL` stream) | Yes — survives broker restart |
| `decision.<trace_id>`, `code.decision.<trace_id>` | JetStream (`SAFETY_CRITICAL` stream) | Yes |
| All other subjects | Core NATS | No (at-most-once) |

The routing decision is made automatically inside `publish()`. Callers do not need to
know which transport is used.

`publish()` validates `dict` payloads against the registered wire model for the subject
(see §3.7) before serialising. Invalid outbound payloads raise `MessageValidationError`.

### 3.6 Subscribe semantics

```python
await self.event_bus.subscribe(
    subject,
    handler,
    queue=None,               # optional queue group for load balancing
    is_request_handler=False, # set True if handler must reply
    message_model=None,       # defaults to SUBJECT_SCHEMAS registry
)
```

- Supports NATS wildcards (`observation.*`, `decision.>`).
- If `subject` is already subscribed, the existing subscription is **replaced** with a warning.
- Pending message buffer: 65 536 messages / 128 MiB (slow-consumer protection).
- The callback wrapper handles deserialisation, validation, and error replies before
  invoking the handler.

**Handler signatures:**

```python
# Fire-and-forget handler
async def handler(data: dict) -> None: ...

# Request-reply handler (is_request_handler=True)
async def handler(data: dict, msg: nats.aio.msg.Msg) -> None:
    await msg.respond(serialize_message(result))
```

### 3.7 Message validation

All incoming messages are validated against a `WireModel` (Pydantic) before the
handler is invoked:

- The registry in `messages.SUBJECT_SCHEMAS` maps known subjects to their models.
- Decision subjects (`decision.*`, `code.decision.*`) are matched by prefix.
- **Unknown subjects pass through unvalidated** — extra fields in known models are
  preserved (`extra="allow"`).

On validation failure:

- The error is logged.
- Fire-and-forget subscriptions: message is **silently dropped** (handler not called).
- Request handlers: caller receives `{"error": "validation_failed", "detail": "...", "type": "error"}`.

### 3.8 Request-reply pattern

**Caller side:**

```python
response = await self.event_bus.request(subject, payload, timeout=30.0)
# response is a dict; raises asyncio.TimeoutError if no reply within timeout
```

**Handler side** (registered with `is_request_handler=True`):

```python
async def _handle(self, data: dict, msg) -> None:
    result = await self._do_work(data)
    if msg.reply:   # guard: message may arrive via plain subscribe, not request()
        await msg.respond(serialize_message(result))
```

Always guard `msg.respond()` with `if msg.reply` so the handler is safe whether
invoked via `request()` or via a plain subscription.

If the handler raises an unhandled exception, the EventBus wrapper sends an error
reply `{"error": "<str(e)>", "type": "error"}` so callers do not hang until timeout.

### 3.9 JetStream (`js_subscribe`)

For durable consumers beyond the safety stream:

```python
await self.event_bus.js_subscribe(subject, handler, durable="my-consumer-name")
```

- The consumer persists across broker restarts under `durable`.
- Successful handler completion → automatic `msg.ack()`.
- Validation failure → `msg.term()` (message terminated, not redelivered).
- Handler exception → `msg.nak()` (message redelivered after NAK delay).

### 3.10 Decision waiting

```python
decision = await self.event_bus.wait_for_decision(trace_id, timeout=30.0)
```

- Creates a per-trace JetStream consumer with `deliver_all` policy so decisions
  published before the call is made are not missed.
- Consumers auto-expire after 60 s of inactivity.
- Signature verification: if `ENGRAM_DECISION_KEY` is set, unsigned or tampered
  decisions are **rejected** (logged as forgery attempts); the wait continues until
  a valid decision arrives or timeout fires. A decision signed with
  `ENGRAM_DECISION_KEY_SECONDARY`, if configured, is also accepted — see
  [`docs/DECISION-KEY-ROTATION.md`](DECISION-KEY-ROTATION.md) for the key-rotation
  procedure this enables.
- On timeout: `asyncio.TimeoutError` is raised — the caller **MUST fail closed**
  (deny / halt, never default-allow).

---

## 4. Known Limitations and Gaps

The items below are **not bugs in the current codebase** but constraints adopters must
account for. Gaps that require code changes are filed as separate blocking issues.

### 4.1 No built-in health check endpoint

`BaseService` provides no standard `*.health` or `*.status` subject. Services that
expose health information implement their own `<service>.status` request handler in
`_setup()`. **Gap:** A unified health protocol would allow the dashboard and launcher
to query all services uniformly. *(Tracked separately.)*

### 4.2 `force_reconnect()` is not automatic

The auto-reconnect built into nats-py covers most cases, but silent-failure scenarios
require a manual call to `force_reconnect()`. Services that rely on continuous message
flow should implement a watchdog (e.g. last-message timestamp check) and call
`force_reconnect()` when stalled. *(No built-in watchdog in `BaseService`.)*

### 4.3 Database singleton is not closed in `stop()`

`self.database` is a process-level singleton (`_db` global in `database.py`). It is
**intentionally not closed** in `stop()` to avoid closing a shared connection mid-flight
when multiple services share a process. In **tests**, this means the same `Database`
object is reused across test cases; use `pytest` fixtures that reset table state rather
than relying on connection teardown.

### 4.4 No timeout on `_setup()`

If `_setup()` blocks indefinitely (e.g. a hung external dependency), `start()` and
`run()` will hang. Services must add their own `asyncio.wait_for()` wrappers where
external calls are made inside `_setup()`. *(Not enforced by `BaseService`.)*

### 4.5 Duplicate subject warning — not an error

Calling `subscribe()` on an already-subscribed subject logs a warning and replaces the
subscription. This is intentional for reconnect scenarios but can mask programming
errors (accidentally subscribing twice). The warning level makes it detectable in logs.

### 4.6 Signal handling not available on Windows

`loop.add_signal_handler()` raises `NotImplementedError` on Windows. `BaseService.run()`
silently ignores this. On Windows, `Ctrl+C` raises `KeyboardInterrupt`, which inherits
from `BaseException` — **not** `Exception` — so it is not caught by `run()`'s
`except Exception` handler. It propagates out of `run()`, but `stop()` is still
guaranteed to execute via the `finally` block. `SIGTERM` is unavailable on Windows.

---

## 5. Configuration Reference

All configuration is loaded from environment variables by `ServiceConfig.from_env()`.

| Variable | Default | Purpose |
|---|---|---|
| `NATS_URL` | `nats://localhost:4222` | NATS server address |
| `SQLITE_PATH` | `~/.engram/engram.db` | SQLite database file |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama (LLM) endpoint |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB endpoint |
| `TASKS_ROOT` | `/data/tasks` | Filesystem path for task artefacts |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `ENGRAM_DECISION_KEY` | *(unset)* | HMAC signing secret. **Unset = signing disabled (dev mode).** |
| `ENGRAM_DECISION_KEY_SECONDARY` | *(unset)* | Verify-only second signing key, accepted alongside `ENGRAM_DECISION_KEY`. Used only during key rotation — see [`docs/DECISION-KEY-ROTATION.md`](DECISION-KEY-ROTATION.md). |

---

## 6. Parity Checklist — Service Adoption

Use this checklist when migrating a service to `BaseService` / `EventBus`. Each item
maps to a guarantee in this document. Tick all boxes before opening the migration PR.

### 6.1 Class structure

- [ ] Service class extends `BaseService` and calls `super().__init__(service_name, ...)`.
- [ ] Constructor (`__init__`) performs **no I/O** — only stores configuration and
      creates plain Python objects.
- [ ] All NATS subscriptions are set up in `_setup()`, not `__init__`.
- [ ] All cleanup (flush, cancel background tasks, unsubscribe) is in `_cleanup()`.
- [ ] Entry point is `asyncio.run(MyService().run())` with no manual
      `start()` / `stop()` calls in `__main__`.

### 6.2 Event bus usage

- [ ] No raw `nats.connect()` or `nats.aio.client.Client` references remain.
- [ ] All publish calls use `self.event_bus.publish(subject, data)`.
- [ ] All subscribe calls use `self.event_bus.subscribe(subject, handler, ...)`.
- [ ] Request handlers are registered with `is_request_handler=True` and call
      `await msg.respond(serialize_message(result))` guarded by `if msg.reply`.
- [ ] Durable JetStream consumers use `self.event_bus.js_subscribe(..., durable=...)`.
- [ ] `wait_for_decision()` callers **fail closed** (deny / halt) on `TimeoutError`.

### 6.3 Message models

- [ ] Every new subject the service publishes **or** subscribes to has a `WireModel`
      in `sdk/src/activelearning/messages.py` and an entry in `SUBJECT_SCHEMAS`.
- [ ] Wire model fields match the actual payload keys (no silent validation drift).

### 6.4 Database

- [ ] Database access uses `self.database` (the `Database` instance set up by `BaseService`).
- [ ] No raw `sqlite3.connect()` calls remain.
- [ ] Required tables are created (via `CREATE TABLE IF NOT EXISTS`) in `_setup()` or
      the `Database.initialize()` call.

### 6.5 Safety subjects

- [ ] Any proposal published to the Kernel uses `Subjects.PROPOSAL_NEW` or
      `Subjects.CODE_PROPOSAL` (these are automatically routed through JetStream).
- [ ] The service does not consume decision subjects directly via `subscribe()`;
      it uses `wait_for_decision(trace_id)` instead.

### 6.6 Regression tests

- [ ] `test_lifecycle_order` — verifies `_setup()` and `_cleanup()` are called in order.
- [ ] `test_graceful_shutdown` — verifies the service stops cleanly on `shutdown()`.
- [ ] `test_subscribe_fire_and_forget` — verifies normal pub/sub works end-to-end.
- [ ] `test_request_reply` — verifies a handler responds correctly to `event_bus.request()`.
- [ ] `test_request_reply_error_reply` — verifies handler exceptions return an error dict.
- [ ] `test_reconnect_restores_subscriptions` — verifies `force_reconnect()` re-registers handlers.
- [ ] `test_validation_drops_bad_messages` — verifies invalid payloads do not reach handlers.
- [ ] `test_decision_wait_timeout_fails_closed` — verifies the service denies / halts on timeout.

---

## 7. Regression Test Template

The canonical regression test suite lives at
`sdk/tests/test_base_service_migration.py`.  Copy it into the service's own
`tests/` directory as `test_migration_parity.py` and adapt the fixtures.

See [sdk/tests/test_base_service_migration.py](../sdk/tests/test_base_service_migration.py)
for the full executable template.

---

## 8. Minimal Adopter Skeleton

```python
# my_service/src/my_service/service.py
from activelearning.base_service import BaseService
from activelearning.nats_client import serialize_message


class MyService(BaseService):
    def __init__(self):
        super().__init__("my-service", use_database=True, use_event_bus=True)
        self._some_resource = None

    async def _setup(self) -> None:
        # Create tables
        await self.database.execute(
            "CREATE TABLE IF NOT EXISTS my_table (id TEXT PRIMARY KEY)"
        )
        # Subscribe to subjects
        await self.event_bus.subscribe(
            "my.topic", self._handle_topic
        )
        await self.event_bus.subscribe(
            "my.status", self._handle_status, is_request_handler=True
        )

    async def _cleanup(self) -> None:
        if self._some_resource:
            await self._some_resource.close()

    async def _handle_topic(self, data: dict) -> None:
        self.logger.info("Received: %s", data)

    async def _handle_status(self, data: dict, msg) -> None:
        status = {"service": self.service_name, "ok": True}
        if msg.reply:
            await msg.respond(serialize_message(status))


if __name__ == "__main__":
    import asyncio
    asyncio.run(MyService().run())
```

---

*Document owner: SDK / backbone team. Last updated: 2026-06-22.*
