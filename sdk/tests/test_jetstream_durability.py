"""Durable JetStream consumer reliability tests (E2.3.3).

Proves the safety-critical consumer contract end-to-end against a live broker:
explicit ack-on-success, at-least-once redelivery when a handler crashes before
ack, and a bounded poison/dead-letter path so a repeatedly-failing message stops
redelivering instead of looping forever or being silently dropped.

Uses the embedded ``nats-server`` started by conftest; skips if unavailable.
Each test uses a unique ``decision.<uuid>`` subject (covered by the safety
stream's ``decision.>``) so durable consumers never see other tests' messages.
"""

from __future__ import annotations

import uuid

import pytest

from activelearning.nats_client import EventBus, poison_subject


def _decision_subject() -> str:
    return f"decision.{uuid.uuid4().hex}"


def _decision_payload(marker: str) -> dict:
    # Valid KernelDecisionMessage (trace_id + type required); extra fields allowed.
    return {"trace_id": marker, "type": "ALLOW", "reason": "test"}


@pytest.mark.asyncio
async def test_success_acks_exactly_once(event_bus: EventBus, wait_for_message) -> None:
    subject = _decision_subject()
    calls: list[dict] = []

    async def handler(data: dict) -> None:
        calls.append(data)

    # Short redelivery interval: if the message were NOT acked, a redelivery
    # would arrive within ~0.5s and the count would climb past 1.
    await event_bus.js_subscribe(
        subject,
        handler,
        durable=f"d-{uuid.uuid4().hex[:8]}",
        max_deliver=3,
        backoff=[0.5, 0.5],
    )
    await event_bus.publish(subject, _decision_payload("ok-1"))

    await wait_for_message(lambda: len(calls) == 1, timeout=5.0)
    # Give any erroneous redelivery time to show up, then assert it didn't.
    import asyncio

    await asyncio.sleep(1.5)
    assert len(calls) == 1, "successful message must be acked exactly once"


@pytest.mark.asyncio
async def test_redelivers_after_handler_crash(event_bus: EventBus, wait_for_message) -> None:
    subject = _decision_subject()
    attempts: list[int] = []
    succeeded: list[dict] = []

    async def handler(data: dict) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("simulated crash before ack")
        succeeded.append(data)

    await event_bus.js_subscribe(
        subject,
        handler,
        durable=f"d-{uuid.uuid4().hex[:8]}",
        max_deliver=5,
        backoff=[0.5, 0.5, 0.5, 0.5],
    )
    await event_bus.publish(subject, _decision_payload("retry-1"))

    # First delivery raises (no ack) → broker redelivers → second delivery succeeds.
    await wait_for_message(lambda: len(succeeded) == 1, timeout=8.0)
    assert len(attempts) >= 2, "a crashed (un-acked) message must be redelivered"
    assert succeeded[0]["trace_id"] == "retry-1", "the message must not be lost"


@pytest.mark.asyncio
async def test_poison_after_max_deliver(event_bus: EventBus, wait_for_message) -> None:
    subject = _decision_subject()
    max_deliver = 3
    attempts: list[int] = []
    poisoned: list[dict] = []
    dlq_msgs: list[dict] = []

    async def handler(_data: dict) -> None:
        attempts.append(1)
        raise RuntimeError("always fails")

    async def on_poison(envelope: dict) -> None:
        poisoned.append(envelope)

    # Watch the dead-letter subject too — a poisoned message must be observable.
    await event_bus.subscribe(poison_subject(subject), lambda d: dlq_msgs.append(d) or None)

    await event_bus.js_subscribe(
        subject,
        handler,
        durable=f"d-{uuid.uuid4().hex[:8]}",
        max_deliver=max_deliver,
        backoff=[0.3, 0.3],
        poison_handler=on_poison,
    )
    await event_bus.publish(subject, _decision_payload("poison-1"))

    await wait_for_message(lambda: len(poisoned) == 1, timeout=8.0)

    # Exactly max_deliver attempts, then poisoned — not infinite, not dropped.
    assert len(attempts) == max_deliver
    assert poisoned[0]["original_subject"] == subject
    assert poisoned[0]["num_delivered"] == max_deliver
    assert "poison-1" in poisoned[0]["payload"]

    # No further redelivery after poisoning (term).
    import asyncio

    await asyncio.sleep(1.5)
    assert len(attempts) == max_deliver, "poisoned message must stop redelivering"
    assert len(poisoned) == 1
    # The dead-letter subject received the envelope (observable, not dropped).
    await wait_for_message(lambda: len(dlq_msgs) == 1, timeout=3.0)
    assert dlq_msgs[0]["original_subject"] == subject


@pytest.mark.asyncio
async def test_raising_poison_handler_does_not_crash_consumer(
    event_bus: EventBus, wait_for_message
) -> None:
    """A poison_handler that raises must not kill the JetStream consumer loop.

    Issue #258: ``_route_to_poison`` wraps the handler in try/except — prove it
    with a real raising handler, then confirm a subsequent good message on the
    *same* durable subscription is still delivered and acked.
    """
    import asyncio

    subject = _decision_subject()
    max_deliver = 3
    poison_calls: list[dict] = []
    fail_attempts = 0
    good_msgs: list[dict] = []

    async def flaky_then_ok(data: dict) -> None:
        nonlocal fail_attempts
        # First message always fails until poisoned; later messages succeed.
        if data.get("trace_id") == "poison-boom":
            fail_attempts += 1
            raise RuntimeError("force poison")
        good_msgs.append(data)

    async def boom_poison(envelope: dict) -> None:
        poison_calls.append(envelope)
        raise RuntimeError("poison handler exploded")

    await event_bus.js_subscribe(
        subject,
        flaky_then_ok,
        durable=f"d-{uuid.uuid4().hex[:8]}",
        max_deliver=max_deliver,
        backoff=[0.3, 0.3],
        poison_handler=boom_poison,
    )
    await event_bus.publish(subject, _decision_payload("poison-boom"))

    await wait_for_message(lambda: len(poison_calls) == 1, timeout=8.0)
    assert poison_calls[0]["original_subject"] == subject
    assert fail_attempts == max_deliver

    # Same consumer callback must still be running after the raising poison handler.
    await event_bus.publish(subject, _decision_payload("after-poison"))
    await wait_for_message(lambda: len(good_msgs) == 1, timeout=5.0)
    assert good_msgs[0]["trace_id"] == "after-poison"

    await asyncio.sleep(1.0)
    assert len(good_msgs) == 1
    assert len(poison_calls) == 1
