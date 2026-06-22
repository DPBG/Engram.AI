"""Tests for memory recall when embeddings fail with a zero-vector sentinel."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from activelearning.embeddings import zero_vector
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
