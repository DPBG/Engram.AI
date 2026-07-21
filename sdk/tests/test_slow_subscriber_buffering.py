"""Slow-subscriber buffering behavior under sustained publish load (issue #240).

No test previously covered what happens when a service can't keep up with the
publish rate. The answer differs completely between the two transports
``EventBus`` uses (see ``EventBus._is_safety_critical``):

- **Core NATS** (``subscribe()``, most subjects): a bounded client-side queue
  per subscription (``pending_msgs_limit`` / ``pending_bytes_limit``). A slow
  handler that falls behind causes *new* incoming messages to be dropped once
  the queue is full -- this is fire-and-forget messaging with no redelivery.
  The publisher is never blocked or informed; only the subscriber's own
  process logs a ``nats.errors.SlowConsumerError`` (via ``EventBus``'s
  ``error_cb``, itself throttled to one log line per 15s per error type).
- **JetStream** (``js_subscribe()``, safety-critical subjects only:
  ``proposal.new``, ``code.proposal``, ``decision.>``, ``code.decision.>``,
  ``policy.*``, ``cognitive.response.*``): the broker persists every message
  to disk and redelivers until acked. A slow handler causes a growing
  server-side backlog (visible via ``fetch_consumer_lag_snapshots`` /
  ``check_consumer_lags``, issue #224's existing lag monitor) but -- short of
  exhausting the stream's retention limits (issue #247: 30 days / 1M msgs) --
  no data loss.

See docs/nats-slow-subscriber-buffering.md for the measured numbers.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

import pytest

from activelearning.nats_client import EventBus


def _load_test_subject() -> str:
    # Deliberately outside every Subjects prefix EventBus routes through
    # JetStream, and unregistered in SUBJECT_SCHEMAS, so payloads pass
    # through validate_payload() unchanged -- this test is about transport
    # buffering, not schema validation.
    return f"loadtest.slow_subscriber.{uuid.uuid4().hex}"


def _decision_subject() -> str:
    return f"decision.{uuid.uuid4().hex}"


class TestCoreNatsSlowSubscriber:
    """Core NATS: bounded client-side queue, drop-on-full, publisher unaffected."""

    @pytest.mark.asyncio
    async def test_sustained_publish_drops_messages_once_pending_limit_hit(
        self, event_bus: EventBus
    ) -> None:
        subject = _load_test_subject()
        received: list[dict] = []

        async def slow_handler(data: dict) -> None:
            await asyncio.sleep(0.05)
            received.append(data)

        await event_bus.subscribe(
            subject, slow_handler, pending_msgs_limit=20, pending_bytes_limit=0
        )

        n_published = 300
        for i in range(n_published):
            await event_bus.publish(subject, {"seq": i})

        # Bounded drain window for whatever survived the drop -- generous
        # relative to how long it would take to process every message
        # (n_published * handler sleep), so this never races the drop itself.
        await asyncio.sleep(2.0)

        assert 0 < len(received) < n_published, (
            f"expected the slow subscriber's bounded queue (limit=20) to drop "
            f"some of {n_published} messages while still processing some; got "
            f"{len(received)} received"
        )

    @pytest.mark.asyncio
    async def test_publisher_is_never_blocked_by_a_slow_subscriber(
        self, event_bus: EventBus
    ) -> None:
        subject = _load_test_subject()

        async def very_slow_handler(data: dict) -> None:
            await asyncio.sleep(1.0)

        await event_bus.subscribe(
            subject, very_slow_handler, pending_msgs_limit=10, pending_bytes_limit=0
        )

        n_published = 200
        t0 = asyncio.get_event_loop().time()
        for i in range(n_published):
            await event_bus.publish(subject, {"seq": i})
        elapsed = asyncio.get_event_loop().time() - t0

        # If publish() were coupled to subscriber drain rate this would take
        # >= n_published * 1.0s (200s). Fire-and-forget core NATS publish
        # should finish in well under a second regardless of subscriber speed.
        assert elapsed < 5.0, (
            f"publish() took {elapsed:.2f}s for {n_published} messages against a "
            "1s/message subscriber -- publish should never block on a slow "
            "subscriber's drain rate"
        )

    @pytest.mark.asyncio
    async def test_slow_consumer_condition_is_logged(
        self, event_bus: EventBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        subject = _load_test_subject()

        async def slow_handler(data: dict) -> None:
            await asyncio.sleep(0.05)

        await event_bus.subscribe(
            subject, slow_handler, pending_msgs_limit=10, pending_bytes_limit=0
        )

        with caplog.at_level(logging.ERROR, logger="activelearning.nats_client"):
            for i in range(200):
                await event_bus.publish(subject, {"seq": i})
            await asyncio.sleep(1.0)

        assert any("SlowConsumerError" in r.message for r in caplog.records), (
            "expected the dropped-message condition to be logged as a "
            "SlowConsumerError via EventBus's error_cb, not silently swallowed"
        )


class TestJetStreamSlowSubscriber:
    """JetStream (safety-critical subjects): server-side backlog, no data loss."""

    @pytest.mark.asyncio
    async def test_slow_consumer_backlog_has_no_data_loss(
        self, event_bus: EventBus, wait_for_message
    ) -> None:
        subject = _decision_subject()
        received: list[dict] = []

        async def slow_handler(data: dict) -> None:
            await asyncio.sleep(0.05)
            received.append(data)

        await event_bus.js_subscribe(
            subject,
            slow_handler,
            durable=f"d-{uuid.uuid4().hex[:8]}",
            max_deliver=3,
            backoff=[5.0, 5.0],
        )

        n_published = 40
        for i in range(n_published):
            await event_bus.publish(subject, {"trace_id": f"t-{i}", "type": "ALLOW"})

        # Publisher finished immediately (JetStream publish is ack'd by the
        # broker, not gated on the consumer); the backlog then drains at the
        # handler's pace. Every message must still arrive -- generous timeout
        # since the handler alone needs n_published * 0.05s minimum.
        await wait_for_message(
            lambda: len(received) == n_published, timeout=n_published * 0.05 + 10.0
        )
        assert len(received) == n_published
        assert {r["trace_id"] for r in received} == {f"t-{i}" for i in range(n_published)}

    @pytest.mark.asyncio
    async def test_slow_consumer_backlog_is_observable_via_lag_snapshot(
        self, event_bus: EventBus, wait_for_message
    ) -> None:
        subject = _decision_subject()
        received: list[dict] = []

        async def slow_handler(data: dict) -> None:
            await asyncio.sleep(0.2)
            received.append(data)

        durable = f"d-{uuid.uuid4().hex[:8]}"
        await event_bus.js_subscribe(
            subject,
            slow_handler,
            durable=durable,
            max_deliver=3,
            backoff=[10.0, 10.0],
        )

        n_published = 20
        for i in range(n_published):
            await event_bus.publish(subject, {"trace_id": f"t-{i}", "type": "ALLOW"})

        # While the handler is still working through the backlog, the lag
        # monitor should see it -- this is the existing issue #224 mechanism
        # this test ties the buffering behavior to.
        async def _backlog_visible() -> bool:
            snapshots = await event_bus.fetch_consumer_lag_snapshots()
            return any(s.consumer == durable and s.lag > 0 for s in snapshots)

        deadline = asyncio.get_event_loop().time() + 5.0
        saw_backlog = False
        while asyncio.get_event_loop().time() < deadline:
            if await _backlog_visible():
                saw_backlog = True
                break
            await asyncio.sleep(0.05)

        assert saw_backlog, (
            "expected fetch_consumer_lag_snapshots() to show positive lag for "
            f"durable={durable} while the slow handler was still draining the "
            "backlog"
        )

        await wait_for_message(lambda: len(received) == n_published, timeout=15.0)
