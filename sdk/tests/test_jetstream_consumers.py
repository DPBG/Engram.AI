"""Integration tests for durable JetStream consumer ack/redelivery (E2.3.3)."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from activelearning.nats_client import (
    JS_MAX_DELIVER,
    poison_subject_for,
    safety_consumer_config,
)
from activelearning.subjects import Subjects
from nats.js.api import AckPolicy

_VALID_PROPOSAL = {
    "trace_id": "js-consumer-test",
    "action": {"type": "move"},
    "provenance": "test",
}

_FAST_BACKOFF = [0.05, 0.05]


class TestSafetyConsumerConfig:
    def test_explicit_ack_policy(self):
        cfg = safety_consumer_config()
        assert cfg.ack_policy == AckPolicy.EXPLICIT
        assert cfg.max_deliver == JS_MAX_DELIVER
        assert cfg.ack_wait is not None
        assert cfg.backoff


@pytest.mark.asyncio
async def test_js_subscribe_acks_once_on_success(event_bus, wait_for_message):
    """Successful handler processing acks exactly once (no redelivery)."""
    attempts: list[dict] = []
    durable = f"test-ack-once-{uuid.uuid4().hex[:8]}"

    async def handler(data: dict) -> None:
        attempts.append(data)

    await event_bus.js_subscribe(
        Subjects.PROPOSAL_NEW,
        handler,
        durable=durable,
        backoff=_FAST_BACKOFF,
    )
    await event_bus.publish(Subjects.PROPOSAL_NEW, _VALID_PROPOSAL)
    await wait_for_message(lambda: len(attempts) == 1, timeout=5.0)
    await asyncio.sleep(0.3)
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_js_subscribe_redelivers_after_handler_crash(event_bus, wait_for_message):
    """A consumer that fails before ack receives redelivery and can succeed."""
    attempts: list[dict] = []
    durable = f"test-redeliver-{uuid.uuid4().hex[:8]}"

    async def handler(data: dict) -> None:
        attempts.append(data)
        if len(attempts) < 2:
            raise RuntimeError("simulated crash before ack")

    await event_bus.js_subscribe(
        Subjects.PROPOSAL_NEW,
        handler,
        durable=durable,
        max_deliver=5,
        backoff=_FAST_BACKOFF,
    )
    await event_bus.publish(Subjects.PROPOSAL_NEW, _VALID_PROPOSAL)
    await wait_for_message(lambda: len(attempts) >= 2, timeout=10.0)
    await asyncio.sleep(0.3)
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_js_subscribe_routes_to_poison_after_max_deliver(event_bus, wait_for_message):
    """Repeated handler failures exhaust max_deliver and land on the poison path."""
    handler_attempts: list[int] = []
    poison_received: list[dict] = []
    durable = f"test-poison-{uuid.uuid4().hex[:8]}"
    poison_durable = f"test-poison-watch-{uuid.uuid4().hex[:8]}"
    poison_subject = poison_subject_for(Subjects.PROPOSAL_NEW)

    async def failing_handler(_data: dict) -> None:
        handler_attempts.append(1)
        raise RuntimeError("persistent handler failure")

    async def poison_handler(data: dict) -> None:
        poison_received.append(data)

    await event_bus.js_subscribe(
        poison_subject,
        poison_handler,
        durable=poison_durable,
        backoff=_FAST_BACKOFF,
    )
    await event_bus.js_subscribe(
        Subjects.PROPOSAL_NEW,
        failing_handler,
        durable=durable,
        max_deliver=3,
        backoff=_FAST_BACKOFF,
    )
    await event_bus.publish(Subjects.PROPOSAL_NEW, _VALID_PROPOSAL)
    await wait_for_message(lambda: len(poison_received) >= 1, timeout=15.0)

    assert len(handler_attempts) == 3
    assert poison_received[0]["original_subject"] == Subjects.PROPOSAL_NEW
    assert poison_received[0]["delivery_count"] == 3
    assert poison_received[0]["payload"]["trace_id"] == _VALID_PROPOSAL["trace_id"]
