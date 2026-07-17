# Changelog

All notable changes to the `activelearning` SDK are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Bump policy

The SDK is shared by every Engram service, so version discipline matters.
Apply these rules whenever you change `sdk/`:

| Change | Version segment |
|---|---|
| New public symbol, new module, new behaviour (backwards-compatible) | **minor** (`0.y.0`) |
| Bug fix, internal refactor, performance, docs | **patch** (`0.y.z`) |
| Removed or renamed public API, changed semantics of existing API | **minor** while `< 1.0`; **major** once `>= 1.0` |

**Mechanics:**
1. Edit `sdk/pyproject.toml` → `version = "X.Y.Z"`.
2. Edit `sdk/src/activelearning/__init__.py` → `__version__ = "X.Y.Z"` (must match).
3. Add a `## [X.Y.Z]` section to this file, dated today, before the previous release.
4. Move any bullet points from `## [Unreleased]` into the new section.
5. Include the bump in the same PR as the change that triggered it.

---

## [Unreleased]

## [0.2.0] — 2026-07-16

### Added

- **Wire-schema versioning** (`WireModel`, `WIRE_SCHEMA_VERSION`): every NATS payload now carries a `version` field; `validate_payload()` enforces the envelope schema per subject (#227).
- **EventBus metrics** (`bus_metrics.py`): publish/subscribe counts and per-subject latency histograms; `EventBus.get_metrics()` and a periodic `MetricsReporter` (#243).
- **JetStream consumer-lag alerting** (`consumer_lag.py`): monitors durable consumers and emits `system.health` events when lag exceeds threshold (#224).
- **Dead-letter queue monitor** (`dlq_monitor.py`): subscribes to `$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>` and re-publishes failed messages to a configurable DLQ subject (#342).
- **EventBus connection logging** (`connection_logging.py`): structured JSON log entries on connect, disconnect, and reconnect events (#256).
- **Batch embeddings** (`embed_batch()`): calls Ollama `/api/embed` for multi-text batches with a single-text fallback (#174).
- **Decision signing with key rotation** (`signing.py`): HMAC-signed kernel decisions; `sign_decision()` / `verify_decision()` with dual-key overlap window for zero-downtime rotation; latency budget regression gate (#185, #190, #192, #206).
- **Unified DB schema migration** (`database.py`): `PRAGMA user_version` tracking, `_MIGRATIONS` list, and ALTER TABLE path with safe skip on fresh databases; `latency_ms` column added to `kernel_decisions` (#262).
- **Safety-critical subject allowlist** (`subjects.py`, `nats_client.py`): `_is_safety_critical()` gates JetStream publish of privileged subjects (#340).
- **`Database.insert()` returns row ID**: captures `cursor.lastrowid` so autoincrement callers get the assigned ID instead of `""` (#245).

### Changed

- `EventBus.subscribe()` now flushes the NATS write buffer after subscribing to prevent `NoRespondersError` on fast producers.
- `EventBus.force_reconnect()` preserves the `is_request_handler` flag on re-subscribe so request-reply handlers are correctly re-registered.

### Fixed

- Expired kernel decisions are now rejected at accept time in all subscribers (replay-attack prevention, #190).
- `Database._migrate()` duplicate statement execution removed (bad merge artifact).
- `Database.execute()` / `executemany()` use `is None` guard to satisfy mypy `union-attr`.

---

## [0.1.0] — 2026-02-03

Initial SDK release extracted from the monolith.

### Added

- `BaseService`: async lifecycle (connect → setup → run → graceful shutdown).
- `EventBus`: NATS pub/sub, request-reply, reconnect, dataclass serialisation.
- `Database`: async SQLite helper (`aiosqlite`) with unified schema.
- `EmbeddingService`: Ollama embedding client with in-memory LRU cache.
- `SensorPlugin` / `ActuatorPlugin`: base classes and registries for sensor and actuator plugins.
- Core types: `Observation`, `ActionProposal`, `KernelDecision`, `BeliefNode`, `BeliefEdge`, `RiskAnalysis`, `Outcome`.
- `Subjects`: NATS subject registry with `decision_subject()`, `code_decision_subject()`, `observation_subject()`.
- `ServiceConfig`: per-service configuration with `NATS_CREDS_FILE` credential loading.
- `QdrantStore`: thin async wrapper for the Qdrant vector database client.
- `LLMClient`: async HTTP client for Ollama-compatible LLM inference.
