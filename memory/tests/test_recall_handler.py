"""Tests for memory recall NATS reply publishing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.models import MemoryResult
from memory.service import MemoryService


@pytest.mark.asyncio
async def test_handle_recall_publishes_semantic_results():
    service = MemoryService.__new__(MemoryService)
    service.logger = MagicMock()
    service.event_bus = MagicMock()
    service.event_bus.publish = AsyncMock()

    sample = [
        MemoryResult(
            episode_id="ep-1",
            score=0.91,
            trace_id="trace-1",
            timestamp=1000,
            tags=["test"],
            summary="hello",
        )
    ]
    service.recall_by_similarity = AsyncMock(return_value=sample)

    await service._handle_recall(
        {
            "query_id": "query-123",
            "query_type": "semantic",
            "query_text": "hello",
            "limit": 5,
        }
    )

    service.event_bus.publish.assert_called_once()
    subject, payload = service.event_bus.publish.call_args[0]
    assert subject == "memory.recall.result.query-123"
    assert payload["query_id"] == "query-123"
    assert payload["count"] == 1
    assert payload["results"][0]["episode_id"] == "ep-1"
    service.recall_by_similarity.assert_awaited_once_with("hello", 5, 0.5)


@pytest.mark.asyncio
async def test_handle_recall_accepts_query_alias():
    service = MemoryService.__new__(MemoryService)
    service.logger = MagicMock()
    service.event_bus = MagicMock()
    service.event_bus.publish = AsyncMock()
    service.recall_by_similarity = AsyncMock(return_value=[])

    await service._handle_recall(
        {
            "query_id": "query-456",
            "type": "similarity",
            "query": "legacy field",
        }
    )

    service.recall_by_similarity.assert_awaited_once_with("legacy field", 10, 0.5)


@pytest.mark.asyncio
async def test_handle_recall_tags_invalid_payload_publishes_error():
    service = MemoryService.__new__(MemoryService)
    service.logger = MagicMock()
    service.event_bus = MagicMock()
    service.event_bus.publish = AsyncMock()

    await service._handle_recall(
        {
            "query_id": "query-bad-tags",
            "query_type": "tags",
            "tags": "safe",
        }
    )

    service.event_bus.publish.assert_called_once()
    subject, payload = service.event_bus.publish.call_args[0]
    assert subject == "memory.recall.result.query-bad-tags"
    assert payload["count"] == 0
    assert "tags must be a list of strings" in payload["error"]


@pytest.mark.asyncio
async def test_handle_recall_publishes_error_payload():
    service = MemoryService.__new__(MemoryService)
    service.logger = MagicMock()
    service.event_bus = MagicMock()
    service.event_bus.publish = AsyncMock()
    service.recall_by_similarity = AsyncMock(side_effect=RuntimeError("boom"))

    await service._handle_recall(
        {
            "query_id": "query-err",
            "query_text": "fail",
        }
    )

    service.event_bus.publish.assert_called_once()
    subject, payload = service.event_bus.publish.call_args[0]
    assert subject == "memory.recall.result.query-err"
    assert payload["count"] == 0
    assert payload["error"] == "boom"
