"""Tests for validate-on-receive in the EventBus subscribe wrapper.

Uses the shared session fixtures from conftest.py (``nats_url`` starts an
embedded nats-server on an ephemeral port; ``event_bus`` and
``wait_for_message`` build on it). This module previously shadowed those
fixtures with a hardcoded ``nats://localhost:4222`` default — with no broker
on 4222, every test here burned its full per-test timeout in fixture setup.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from nats.aio.msg import Msg

from activelearning.nats_client import EventBus
from activelearning.subjects import Subjects


async def _wait_without_handler_call(
    received: list[dict[str, Any]],
    timeout: float = 1.0,
    interval: float = 0.05,
) -> None:
    """Wait for delivery window to elapse without the handler being invoked."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if received:
            raise AssertionError(f"handler was called unexpectedly: {received}")
        await asyncio.sleep(interval)


@pytest.mark.asyncio
async def test_subscribe_valid_payload_reaches_handler(
    event_bus: EventBus,
    wait_for_message,
) -> None:
    received: list[dict[str, Any]] = []

    async def handler(data: dict[str, Any]) -> None:
        received.append(data)

    await event_bus.subscribe(Subjects.PROPOSAL_NEW, handler)

    payload = {
        "trace_id": "trace-valid",
        "action": {"type": "move"},
        "provenance": "test",
    }
    await event_bus.publish(Subjects.PROPOSAL_NEW, payload)
    await wait_for_message(lambda: len(received) == 1)

    assert received[0]["trace_id"] == "trace-valid"
    assert received[0]["action"]["type"] == "move"


@pytest.mark.asyncio
async def test_subscribe_malformed_payload_rejected(event_bus: EventBus) -> None:
    received: list[dict[str, Any]] = []

    async def handler(data: dict[str, Any]) -> None:
        received.append(data)

    await event_bus.subscribe(Subjects.PROPOSAL_NEW, handler)

    import nats

    nc = await nats.connect(event_bus.nats_url)
    try:
        await nc.publish(
            Subjects.PROPOSAL_NEW,
            b'{"action": {"type": "move"}}',
        )
    finally:
        await nc.close()

    await _wait_without_handler_call(received)

    assert received == []


@pytest.mark.asyncio
async def test_subscribe_unmodeled_subject_passes_through(
    event_bus: EventBus,
    wait_for_message,
) -> None:
    subject = f"custom.unmodeled.{uuid.uuid4().hex[:8]}"
    received: list[dict[str, Any]] = []

    async def handler(data: dict[str, Any]) -> None:
        received.append(data)

    await event_bus.subscribe(subject, handler)

    payload = {"anything": True, "nested": {"x": 1}}
    await event_bus.publish(subject, payload)
    await wait_for_message(lambda: received == [payload])


@pytest.mark.asyncio
async def test_request_handler_validation_error_replies(event_bus: EventBus) -> None:
    handler_called = False

    async def handler(data: dict[str, Any], msg: Msg) -> None:
        nonlocal handler_called
        handler_called = True
        from activelearning.nats_client import serialize_message

        await msg.respond(serialize_message({"ok": True}))

    await event_bus.subscribe(
        Subjects.MEMORY_STORE,
        handler,
        is_request_handler=True,
    )

    import nats

    nc = await nats.connect(event_bus.nats_url)
    try:
        response = await nc.request(
            Subjects.MEMORY_STORE,
            b'{"summary": "missing trace_id"}',
            timeout=2.0,
        )
        body = response.data.decode()
        assert "validation_failed" in body
    finally:
        await nc.close()

    assert handler_called is False
