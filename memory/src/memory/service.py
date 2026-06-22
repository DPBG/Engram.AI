"""
Memory Service - Episodic memory with vector embeddings.

This service manages the system's memory, storing observations
and experiences with vector embeddings for semantic retrieval.
"""

import asyncio
import json
from dataclasses import asdict
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from activelearning import BaseService, get_embedding_service
from activelearning.subjects import Subjects

from memory.models import Episode, MemoryQuery, MemoryResult

# Constants
COLLECTION_NAME = "memory_episodes"
EMBEDDING_DIMENSIONS = 768  # nomic-embed-text


class MemoryService(BaseService):
    """
    Episodic memory service with vector storage.

    Stores memories in SQLite with vector embeddings in Qdrant
    for semantic retrieval.
    """

    def __init__(self):
        super().__init__("memory", use_database=True, use_event_bus=True)
        self._qdrant: Optional[AsyncQdrantClient] = None
        self._embedding_service = get_embedding_service()

    async def _setup(self) -> None:
        """Service-specific setup."""
        # Connect to Qdrant
        self.logger.info(f"Connecting to Qdrant at {self.config.qdrant_url}")
        self._qdrant = AsyncQdrantClient(url=self.config.qdrant_url)

        # Ensure collection exists
        await self._ensure_collection()

        # Subscribe to memory events using EventBus
        await self.event_bus.subscribe(Subjects.MEMORY_STORE, self._handle_store)
        await self.event_bus.subscribe(Subjects.MEMORY_QUERY, self._handle_query)
        await self.event_bus.subscribe(Subjects.MEMORY_RECALL, self._handle_recall)

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        if self._qdrant:
            await self._qdrant.close()

    async def _ensure_collection(self) -> None:
        """Ensure Qdrant collection exists."""
        collections = await self._qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]

        if COLLECTION_NAME not in collection_names:
            self.logger.info(f"Creating Qdrant collection: {COLLECTION_NAME}")
            await self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )

    async def store_episode(self, episode: Episode) -> str:
        """
        Store an episode in memory.

        Args:
            episode: The episode to store

        Returns:
            Episode ID
        """
        # Generate embedding
        embedding = await self._embed_text(episode.summary)

        # Store in SQLite using Database helper
        await self.database.insert(
            "memory_episodes",
            {
                "id": episode.id,
                "trace_id": episode.trace_id,
                "timestamp": episode.timestamp,
                "embedding_ref": episode.id,  # Points to Qdrant point ID
                "semantic_tags": json.dumps(episode.tags),
                "utility_score": episode.utility_score,
                "data": json.dumps(episode.data),
            },
        )

        # Store embedding in Qdrant
        await self._qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=episode.id,
                    vector=embedding,
                    payload={
                        "trace_id": episode.trace_id,
                        "timestamp": episode.timestamp,
                        "tags": episode.tags,
                        "summary": episode.summary,
                    },
                )
            ],
        )

        self.logger.debug(f"Stored episode: {episode.id}")
        return episode.id

    async def recall_by_similarity(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
    ) -> list[MemoryResult]:
        """
        Recall memories by semantic similarity.

        Args:
            query: Search query
            limit: Max results
            min_score: Minimum similarity score

        Returns:
            List of matching memories
        """
        # Generate query embedding
        embedding = await self._embed_text(query)

        # Search Qdrant
        results = await self._qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
            score_threshold=min_score,
        )

        memories = []
        for result in results:
            memories.append(
                MemoryResult(
                    episode_id=str(result.id),
                    score=result.score,
                    trace_id=result.payload.get("trace_id", ""),
                    timestamp=result.payload.get("timestamp", 0),
                    tags=result.payload.get("tags", []),
                    summary=result.payload.get("summary", ""),
                )
            )

        return memories

    async def recall_by_time_window(
        self,
        start_time: int,
        end_time: int,
        limit: int = 100,
    ) -> list[MemoryResult]:
        """
        Recall memories within a time window.

        Args:
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)
            limit: Max results

        Returns:
            List of matching memories
        """
        rows = await self.database.fetchall(
            """
            SELECT id, trace_id, timestamp, semantic_tags, utility_score, data
            FROM memory_episodes
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (start_time, end_time, limit),
        )

        memories = []
        for row in rows:
            memories.append(
                MemoryResult(
                    episode_id=row["id"],
                    score=row["utility_score"],
                    trace_id=row["trace_id"],
                    timestamp=row["timestamp"],
                    tags=json.loads(row["semantic_tags"] or "[]"),
                    summary="",  # Would need to fetch from data
                )
            )

        return memories

    async def recall_by_tags(
        self,
        tags: list[str],
        limit: int = 100,
    ) -> list[MemoryResult]:
        """
        Recall memories by semantic tags.

        Args:
            tags: Tags to search for
            limit: Max results

        Returns:
            List of matching memories
        """
        # Build tag matching query
        tag_conditions = " OR ".join(
            f"semantic_tags LIKE '%\"{tag}\"%'" for tag in tags
        )

        rows = await self.database.fetchall(
            f"""
            SELECT id, trace_id, timestamp, semantic_tags, utility_score
            FROM memory_episodes
            WHERE {tag_conditions}
            ORDER BY utility_score DESC, timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

        memories = []
        for row in rows:
            memories.append(
                MemoryResult(
                    episode_id=row["id"],
                    score=row["utility_score"],
                    trace_id=row["trace_id"],
                    timestamp=row["timestamp"],
                    tags=json.loads(row["semantic_tags"] or "[]"),
                    summary="",
                )
            )

        return memories

    async def _embed_text(self, text: str) -> list[float]:
        """Generate embedding for text using SDK EmbeddingService."""
        return await self._embedding_service.embed_text(text)

    async def _handle_store(self, data: dict) -> None:
        """Handle memory store requests."""
        try:
            episode = Episode(**data)
            episode_id = await self.store_episode(episode)
            # EventBus handles serialization automatically
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")

    async def _handle_query(self, data: dict) -> None:
        """Handle memory query requests."""
        try:
            query = MemoryQuery(**data)
            results = await self.recall_by_similarity(
                query.query,
                limit=query.limit,
                min_score=query.min_score,
            )
        except Exception as e:
            self.logger.error(f"Error querying memory: {e}")

    async def _handle_recall(self, data: dict) -> None:
        """Handle memory recall requests."""
        query_id = data.get("query_id")
        recall_type = data.get("query_type") or data.get("type", "similarity")

        try:
            if recall_type == "time_window":
                results = await self.recall_by_time_window(
                    data["start_time"],
                    data["end_time"],
                    data.get("limit", 100),
                )
            elif recall_type == "tags":
                results = await self.recall_by_tags(
                    data["tags"],
                    data.get("limit", 100),
                )
            else:
                query_text = data.get("query_text") or data.get("query", "")
                results = await self.recall_by_similarity(
                    query_text,
                    data.get("limit", 10),
                    data.get("min_score", 0.5),
                )
        except Exception as e:
            self.logger.error(f"Error recalling memory: {e}")
            if query_id:
                await self.event_bus.publish(
                    f"memory.recall.result.{query_id}",
                    {
                        "query_id": query_id,
                        "results": [],
                        "count": 0,
                        "error": str(e),
                    },
                )
            return

        if query_id:
            await self.event_bus.publish(
                f"memory.recall.result.{query_id}",
                {
                    "query_id": query_id,
                    "results": [asdict(result) for result in results],
                    "count": len(results),
                },
            )


async def main() -> None:
    """Main entry point."""
    service = MemoryService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
