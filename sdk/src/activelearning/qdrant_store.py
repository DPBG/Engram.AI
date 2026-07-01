"""
Shared Qdrant vector-store helper.

Wraps the official ``AsyncQdrantClient`` so services don't hand-write REST
payloads (and don't each spin up a fresh ``aiohttp`` session per operation).
It centralizes the small set of operations services actually need:

- ``ensure_collection`` — create a collection on first use if it's missing
- ``search`` — nearest-neighbour query, returning normalized :class:`QdrantHit`
- ``upsert`` — write points described by :class:`QdrantPoint`
- ``delete`` — remove points by id

Returning plain value objects (:class:`QdrantHit`) keeps callers decoupled from
``qdrant_client``'s wire types, which makes them trivial to unit-test.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# nomic-embed-text — the embedding model shared across every service.
DEFAULT_DIMENSIONS = 768
DEFAULT_QDRANT_URL = "http://localhost:6333"


@dataclass
class QdrantHit:
    """A single scored search result, decoupled from the client's wire type."""

    id: Any
    score: float
    payload: dict[str, Any]


@dataclass
class QdrantPoint:
    """A point to upsert: an id, its embedding vector, and a payload."""

    id: Any
    vector: list[float]
    payload: dict[str, Any]


class QdrantStore:
    """Thin async wrapper over :class:`AsyncQdrantClient`.

    A single store instance manages one Qdrant connection and may serve any
    number of collections. Construct it with the server URL (or inject a
    pre-built ``client`` for testing); collection geometry defaults to the
    shared embedding model and cosine distance.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        dimensions: int = DEFAULT_DIMENSIONS,
        distance: Distance = Distance.COSINE,
        client: Optional[AsyncQdrantClient] = None,
    ):
        """Create a store.

        Args:
            url: Qdrant server URL. Defaults to ``QDRANT_URL`` env var, then
                ``http://localhost:6333``. Ignored when ``client`` is given.
            dimensions: Default vector size used by :meth:`ensure_collection`.
            distance: Default distance metric for new collections.
            client: Pre-built client to wrap (primarily for testing).
        """
        self._client = client or AsyncQdrantClient(
            url=url or os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
        )
        self._dimensions = dimensions
        self._distance = distance

    async def ensure_collection(self, collection: str, *, dimensions: Optional[int] = None) -> None:
        """Create ``collection`` if it does not already exist (idempotent)."""
        existing = await self._client.get_collections()
        if collection not in {c.name for c in existing.collections}:
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dimensions or self._dimensions,
                    distance=self._distance,
                ),
            )

    async def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> list[QdrantHit]:
        """Return the ``limit`` nearest points to ``vector`` as :class:`QdrantHit`."""
        response = await self._client.query_points(
            collection_name=collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return [
            QdrantHit(id=point.id, score=point.score, payload=point.payload or {})
            for point in response.points
        ]

    async def upsert(self, collection: str, points: Sequence[QdrantPoint]) -> None:
        """Insert or replace ``points`` in ``collection``."""
        await self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
        )

    async def delete(self, collection: str, ids: Sequence[Any]) -> None:
        """Delete points with the given ``ids`` from ``collection``."""
        await self._client.delete(
            collection_name=collection,
            points_selector=list(ids),
        )

    async def close(self) -> None:
        """Close the underlying client connection."""
        await self._client.close()
