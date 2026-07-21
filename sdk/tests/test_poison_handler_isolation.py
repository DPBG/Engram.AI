"""Unit tests for EventBus._route_to_poison isolation (issue #258)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from activelearning.nats_client import EventBus


@pytest.mark.asyncio
async def test_route_to_poison_swallows_raising_handler() -> None:
    """``_route_to_poison`` must never propagate a poison_handler exception."""
    bus = EventBus(name="poison-isolation")
    bus._nc = MagicMock()
    bus._nc.publish = AsyncMock()

    called: list[dict] = []

    async def boom(envelope: dict) -> None:
        called.append(envelope)
        raise RuntimeError("handler boom")

    bus._poison_handlers["decision.test"] = boom

    msg = MagicMock()
    msg.data = b'{"trace_id":"x","type":"ALLOW"}'
    msg.metadata.num_delivered = 5

    # Must complete without raising.
    await bus._route_to_poison("decision.test", msg, "max_deliver_exhausted")

    assert len(called) == 1
    assert called[0]["original_subject"] == "decision.test"
    assert called[0]["reason"] == "max_deliver_exhausted"
    bus._nc.publish.assert_awaited()


@pytest.mark.asyncio
async def test_route_to_poison_swallows_dlq_publish_failure() -> None:
    """A failed DLQ publish must also not raise out of ``_route_to_poison``."""
    bus = EventBus(name="poison-dlq-fail")
    bus._nc = MagicMock()
    bus._nc.publish = AsyncMock(side_effect=ConnectionError("broker down"))

    ph = AsyncMock()
    bus._poison_handlers["decision.test"] = ph

    msg = MagicMock()
    msg.data = b"{}"
    msg.metadata.num_delivered = 1

    await bus._route_to_poison("decision.test", msg, "validation_error")
    ph.assert_awaited_once()
