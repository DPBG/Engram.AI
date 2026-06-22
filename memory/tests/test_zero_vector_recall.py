"""Tests for memory recall when embeddings fail with a zero-vector sentinel."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from activelearning.embeddings import zero_vector
from memory.models import Episode
from memory.service import EMBEDDING_DIMENSIONS, MemoryService


@pytest.mark.asyncio
async def test_recall_by_similarity_returns_empty_on_zero_vector():
    service = MemoryService.__new__(MemoryService)
    service.logger = MagicMock()
    service._embed_text = AsyncMock(return_value=zero_vector(EMBEDDING_DIMENSIONS))
    service._qdrant = MagicMock()
    service._qdrant.search = AsyncMock()

    results = await service.recall_by_similarity("unavailable embedding")

    assert results == []
    service._qdrant.search.assert_not_called()
    service.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_store_episode_skips_qdrant_on_zero_vector():
    service = MemoryService.__new__(MemoryService)
    service.logger = MagicMock()
    service.database = MagicMock()
    service.database.insert = AsyncMock()
    service._embed_text = AsyncMock(return_value=zero_vector(EMBEDDING_DIMENSIONS))
    service._qdrant = MagicMock()
    service._qdrant.upsert = AsyncMock()

    episode = Episode(trace_id="trace-1", summary="embedding backend down")
    episode_id = await service.store_episode(episode)

    assert episode_id == episode.id
    service.database.insert.assert_awaited_once()
    inserted = service.database.insert.await_args.args[1]
    assert inserted["embedding_ref"] is None
    service._qdrant.upsert.assert_not_called()
    service.logger.warning.assert_called_once()
