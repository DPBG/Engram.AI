# Slow-subscriber buffering behavior under sustained publish load

No test previously covered what happens when a service can't keep up with
the publish rate on its subscription(s) — issue #240 asked for one, plus a
record of what actually happens: silent drop, or block? The honest answer is
**it depends entirely on which transport the subject uses**, and `EventBus`
uses two, chosen per-subject by `EventBus._is_safety_critical()`. The stress
tests backing the numbers below live in
`sdk/tests/test_slow_subscriber_buffering.py`; they run against a real
embedded `nats-server`, not mocks.

## Core NATS (`subscribe()`) — most subjects: bounded queue, silent drop

Every subject that isn't safety-critical (`proposal.new`, `code.proposal`,
`decision.>`, `code.decision.>`, `policy.*`, `cognitive.response.*`) goes
through core NATS pub/sub — fire-and-forget, no persistence, no
acknowledgment. Each subscription has a **bounded client-side queue**
(`pending_msgs_limit=65536` / `pending_bytes_limit=128MB` by default,
configurable per `EventBus.subscribe()` call). When a handler falls behind
the publish rate long enough for that queue to fill, nats-py's client
library — not the broker — **drops the new incoming message outright**
(`nats/aio/client.py`, `_process_msg`) rather than blocking the publisher or
evicting an older queued message.

**Measured** (`pending_msgs_limit=20`, handler sleeping 50ms/message, 300
messages published back-to-back with no delay):

- **20/300 received, 280/300 dropped** — exactly at the configured queue
  capacity, not a fuzzy threshold.
- **Publish loop: ~2ms** for all 300 messages — the publisher is completely
  decoupled from subscriber drain speed; `publish()` never blocks or raises
  because of a slow subscriber.
- **Exactly 1 log line** for the entire episode, despite 280 individual drop
  events: `EventBus._error_callback` receives a `SlowConsumerError` from
  nats-py for every drop, but throttles identical error types to one log
  line per 15 seconds (a pre-existing anti-flood measure). **This means a
  sustained slow-consumer incident is dramatically under-represented in the
  logs relative to how many messages were actually lost** — worth knowing
  before using log volume as a proxy for drop volume during an incident.

### Recommendation

For a subject where dropped messages are unacceptable, either move it into
the safety-critical/JetStream set (`EventBus._is_safety_critical`,
`_SAFETY_STREAM_SUBJECTS`), or size `pending_msgs_limit`/`pending_bytes_limit`
generously and monitor for `SlowConsumerError` in that service's own logs —
don't rely on drop volume matching log line count.

## JetStream (`js_subscribe()`) — safety-critical subjects: persisted backlog, no loss

Safety-critical subjects are persisted to the `SAFETY_CRITICAL` stream
(`RetentionPolicy.LIMITS`, 30 days / 1,000,000 messages, file storage —
issue #247) with an explicit-ack durable consumer. A slow handler does not
cause drops: the broker keeps redelivering/holding the message until it's
acked, so the backlog simply grows server-side, bounded only by the stream's
retention limits (which a slow *handler* — as opposed to a genuinely stalled
or crashed one — will never come close to in practice).

**Measured** (handler sleeping 50ms/message, 40 messages published
back-to-back):

- **40/40 received, 0 lost** — every message eventually arrived exactly
  once, confirmed by trace ID.
- **Publish loop: ~11ms** for all 40 messages — like core NATS, the
  publisher is not gated on subscriber speed (JetStream `publish()` only
  waits for the broker's own ack that the message was persisted).
- **Backlog is directly observable mid-drain**: sampling
  `fetch_consumer_lag_snapshots()` while the handler was still working
  through the queue showed `num_ack_pending=34` (34 delivered-but-unacked
  messages) — exactly the existing issue #224 lag monitor's `lag` metric
  (`max(num_pending, num_ack_pending)`). A slow JetStream consumer is
  already wired into the alerting this repo has, not a blind spot.

### The actual limit

A slow-but-eventually-successful handler never gets dead-lettered — poisoning
only happens after `max_deliver` *failed* (raised) attempts, which is a
processing-failure path, not a pure-slowness one. The real limit is the
stream's retention (30 days / 1M messages): a subscriber that falls behind by
more than that would start losing its oldest backlog, but this is far beyond
what "slow" realistically means for any of this repo's safety-critical
handlers today.

## Summary

| | Core NATS (most subjects) | JetStream (safety-critical subjects) |
|---|---|---|
| Slow subscriber's effect | New messages **dropped** once the client-side queue fills | Backlog **persists and grows** server-side |
| Publisher blocked? | No | No |
| Data loss? | **Yes**, silently past the queue limit (drop is logged, but throttled to 1 line/15s regardless of drop volume) | No, short of exhausting stream retention (30d / 1M msgs) |
| Existing observability | `SlowConsumerError` via `EventBus._error_callback` (throttled) | `fetch_consumer_lag_snapshots()` / `check_consumer_lags()` (issue #224) |
