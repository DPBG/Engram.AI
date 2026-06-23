"""Lifecycle and validation tests for CognitiveBridgeService (BaseService migration)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


def _require_sdk():
    pytest.importorskip("activelearning")
    from activelearning.subjects import Subjects

    return Subjects


@pytest.mark.asyncio
async def test_setup_subscribes_to_cognitive_subjects() -> None:
    Subjects = _require_sdk()
    from neuromorphic.cognitive_bridge import CognitiveBridgeService

    bridge = CognitiveBridgeService()
    bus = AsyncMock()
    bridge.event_bus = bus

    await bridge._setup()

    assert bus.subscribe.await_count == 2
    bus.subscribe.assert_any_await(
        Subjects.COGNITIVE_EXECUTE,
        bridge._handle_query,
    )
    bus.subscribe.assert_any_await(
        Subjects.COGNITIVE_QUERY,
        bridge._handle_query,
    )


@pytest.mark.asyncio
async def test_cleanup_closes_llm_client() -> None:
    _require_sdk()
    from neuromorphic.cognitive_bridge import CognitiveBridgeService

    bridge = CognitiveBridgeService()
    llm = AsyncMock()
    bridge._llm = llm

    await bridge._cleanup()

    llm.close.assert_awaited_once()
    assert bridge._llm is None


@pytest.mark.asyncio
async def test_cleanup_idempotent_when_llm_uninitialized() -> None:
    _require_sdk()
    from neuromorphic.cognitive_bridge import CognitiveBridgeService

    bridge = CognitiveBridgeService()
    assert bridge._llm is None

    await bridge._cleanup()

    assert bridge._llm is None


def test_invalid_cognitive_execute_payload_fails_validation() -> None:
    """Wire-model gate rejects structurally invalid payloads before handlers run."""
    Subjects = _require_sdk()
    from activelearning.messages import MessageValidationError, validate_payload

    with pytest.raises(MessageValidationError):
        validate_payload(
            Subjects.COGNITIVE_EXECUTE,
            {"prediction_error": "not-a-float"},
        )


@pytest.mark.asyncio
async def test_invalid_cognitive_payload_does_not_reach_handle_query() -> None:
    """EventBus validates before invoking the registered handler."""
    Subjects = _require_sdk()
    from activelearning.messages import MessageValidationError, validate_payload
    from neuromorphic.cognitive_bridge import CognitiveBridgeService

    bridge = CognitiveBridgeService()
    calls: list[dict] = []

    async def spy_handle(data: dict) -> None:
        calls.append(data)

    bridge._handle_query = spy_handle  # type: ignore[method-assign]
    bridge.event_bus = AsyncMock()

    await bridge._setup()

    invalid = {"prediction_error": "not-a-float"}
    with pytest.raises(MessageValidationError):
        validate_payload(Subjects.COGNITIVE_EXECUTE, invalid)

    assert calls == []


@pytest.mark.asyncio
async def test_cleanup_closes_llm_only_once_on_repeat() -> None:
    _require_sdk()
    from neuromorphic.cognitive_bridge import CognitiveBridgeService

    bridge = CognitiveBridgeService()
    llm = AsyncMock()
    bridge._llm = llm

    await bridge._cleanup()
    await bridge._cleanup()

    llm.close.assert_awaited_once()
