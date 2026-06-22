"""Tests for the shared QdrantStore helper.

The store is exercised against a mocked AsyncQdrantClient so the tests don't
require a running Qdrant. They assert the helper speaks the official client API
(query_points/upsert/delete/create_collection) and normalizes results into
plain QdrantHit value objects.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client.models import Distance, PointStruct

from activelearning import QdrantHit, QdrantPoint, QdrantStore


def _make_store():
    """Return a QdrantStore wrapping a fully-mocked async client."""
    client = AsyncMock()
    store = QdrantStore(client=client)
    return store, client


@pytest.mark.asyncio
async def test_search_normalizes_query_points_response():
    store, client = _make_store()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(id="a", score=0.9, payload={"task_id": "a"}),
            SimpleNamespace(id="b", score=0.4, payload=None),
        ]
    )

    hits = await store.search("tasks", [0.1, 0.2, 0.3], limit=2, score_threshold=0.3)

    assert hits == [
        QdrantHit(id="a", score=0.9, payload={"task_id": "a"}),
        QdrantHit(id="b", score=0.4, payload={}),  # None payload becomes {}
    ]
    # Uses the modern query_points API (not the removed .search()).
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "tasks"
    assert kwargs["query"] == [0.1, 0.2, 0.3]
    assert kwargs["limit"] == 2
    assert kwargs["with_payload"] is True
    assert kwargs["score_threshold"] == 0.3


@pytest.mark.asyncio
async def test_search_empty_results():
    store, client = _make_store()
    client.query_points.return_value = SimpleNamespace(points=[])
    assert await store.search("tasks", [0.0, 1.0]) == []


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing():
    store, client = _make_store()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="other")]
    )

    await store.ensure_collection("tasks")

    client.create_collection.assert_awaited_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "tasks"
    assert kwargs["vectors_config"].size == 768
    assert kwargs["vectors_config"].distance == Distance.COSINE


@pytest.mark.asyncio
async def test_ensure_collection_is_idempotent_when_present():
    store, client = _make_store()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="tasks")]
    )

    await store.ensure_collection("tasks")

    client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_collection_honors_dimension_override():
    store, client = _make_store()
    client.get_collections.return_value = SimpleNamespace(collections=[])

    await store.ensure_collection("tasks", dimensions=1024)

    assert client.create_collection.call_args.kwargs["vectors_config"].size == 1024


@pytest.mark.asyncio
async def test_upsert_builds_point_structs():
    store, client = _make_store()

    await store.upsert(
        "tasks",
        [QdrantPoint(id="t1", vector=[0.1, 0.2], payload={"task_id": "t1"})],
    )

    kwargs = client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "tasks"
    points = kwargs["points"]
    assert len(points) == 1
    assert isinstance(points[0], PointStruct)
    assert points[0].id == "t1"
    assert points[0].vector == [0.1, 0.2]
    assert points[0].payload == {"task_id": "t1"}


@pytest.mark.asyncio
async def test_delete_passes_id_list():
    store, client = _make_store()

    await store.delete("tasks", ["h1", "h2"])

    kwargs = client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "tasks"
    assert kwargs["points_selector"] == ["h1", "h2"]


@pytest.mark.asyncio
async def test_close_closes_client():
    store, client = _make_store()
    await store.close()
    client.close.assert_awaited_once()
