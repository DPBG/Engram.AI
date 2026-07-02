"""Tests for embedding failure handling and zero-vector sentinels."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from activelearning.embeddings import (
    EmbeddingService,
    is_zero_vector,
    zero_vector,
)


class TestZeroVectorHelpers:
    def test_zero_vector_dimensions(self):
        assert zero_vector(4) == [0.0, 0.0, 0.0, 0.0]

    def test_is_zero_vector_true(self):
        assert is_zero_vector([0.0, 0.0]) is True

    def test_is_zero_vector_false_for_nonempty(self):
        assert is_zero_vector([0.0, 0.1]) is False

    def test_is_zero_vector_false_for_empty(self):
        assert is_zero_vector([]) is False

    def test_zero_vector_rejects_non_positive_dimensions(self):
        with pytest.raises(ValueError, match="positive integer"):
            zero_vector(0)
        with pytest.raises(ValueError, match="positive integer"):
            zero_vector(-1)


@pytest.mark.asyncio
async def test_embed_text_returns_zero_vector_on_http_error():
    service = EmbeddingService(dimensions=8)

    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="server error")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    service._get_session = AsyncMock(return_value=mock_session)

    result = await service.embed_text("hello world")

    assert result == zero_vector(8)
    assert is_zero_vector(result)


@pytest.mark.asyncio
async def test_embed_text_returns_zero_vector_on_network_error():
    service = EmbeddingService(dimensions=16)
    service._get_session = AsyncMock(side_effect=ConnectionError("refused"))

    result = await service.embed_text("offline query")

    assert result == zero_vector(16)


@pytest.mark.asyncio
async def test_embed_text_returns_zero_vector_on_dimension_mismatch():
    service = EmbeddingService(dimensions=4)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"embedding": [0.1, 0.2]})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    service._get_session = AsyncMock(return_value=mock_session)

    result = await service.embed_text("wrong size")

    assert result == zero_vector(4)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"embedding": "not-a-list"}])
async def test_embed_text_returns_zero_vector_on_invalid_payload_type(payload):
    service = EmbeddingService(dimensions=4)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=payload)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    service._get_session = AsyncMock(return_value=mock_session)

    result = await service.embed_text("bad payload")

    assert result == zero_vector(4)
    assert is_zero_vector(result)


@pytest.mark.asyncio
async def test_embed_text_returns_zero_vector_on_non_numeric_components():
    service = EmbeddingService(dimensions=3)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"embedding": [0.1, "bad", 0.3]})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    service._get_session = AsyncMock(return_value=mock_session)

    result = await service.embed_text("non-numeric")

    assert result == zero_vector(3)


@pytest.mark.asyncio
async def test_embed_text_does_not_cache_zero_vector_on_failure():
    service = EmbeddingService(dimensions=4)
    service._get_session = AsyncMock(side_effect=RuntimeError("boom"))

    await service.embed_text("fail once")
    await service.embed_text("fail once")

    service._get_session.assert_awaited()
    assert service._get_session.await_count == 2


@pytest.mark.asyncio
async def test_embed_text_caches_successful_embedding():
    service = EmbeddingService(dimensions=3)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"embedding": [0.1, 0.2, 0.3]})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    service._get_session = AsyncMock(return_value=mock_session)

    first = await service.embed_text("cache me")
    second = await service.embed_text("cache me")

    assert first == [0.1, 0.2, 0.3]
    assert second == first
    assert service._get_session.await_count == 1


@pytest.mark.asyncio
async def test_embed_batch_empty_list():
    service = EmbeddingService(dimensions=4)
    assert await service.embed_batch([]) == []


@pytest.mark.asyncio
async def test_embed_batch_uses_cache_without_http():
    service = EmbeddingService(dimensions=3)
    service._cache[service._cache_key("cached")] = [0.1, 0.2, 0.3]
    service._get_session = AsyncMock()

    result = await service.embed_batch(["cached", "cached"])

    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    service._get_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_batch_fetches_uncached_texts_in_single_request():
    service = EmbeddingService(dimensions=3)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    service._get_session = AsyncMock(return_value=mock_session)

    result = await service.embed_batch(["alpha", "beta"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_session.post.assert_called_once()
    assert mock_session.post.call_args.kwargs["json"]["input"] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_embed_batch_preserves_order_with_mixed_cache_hits():
    service = EmbeddingService(dimensions=2)
    service._cache[service._cache_key("cached")] = [1.0, 2.0]

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"embeddings": [[0.3, 0.4]]})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    service._get_session = AsyncMock(return_value=mock_session)

    result = await service.embed_batch(["cached", "fresh"])

    assert result == [[1.0, 2.0], [0.3, 0.4]]
    assert mock_session.post.call_args.kwargs["json"]["input"] == ["fresh"]


@pytest.mark.asyncio
async def test_embed_batch_falls_back_to_sequential_on_batch_failure():
    service = EmbeddingService(dimensions=2)

    single_response = MagicMock()
    single_response.status = 200
    single_response.json = AsyncMock(
        side_effect=[
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
    )
    single_response.__aenter__ = AsyncMock(return_value=single_response)
    single_response.__aexit__ = AsyncMock(return_value=False)

    batch_response = MagicMock()
    batch_response.status = 500
    batch_response.text = AsyncMock(return_value="batch failed")
    batch_response.__aenter__ = AsyncMock(return_value=batch_response)
    batch_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=[batch_response, single_response, single_response])
    service._get_session = AsyncMock(return_value=mock_session)

    result = await service.embed_batch(["one", "two"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert mock_session.post.call_count == 3
