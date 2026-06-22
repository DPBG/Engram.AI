"""
LLM Cache - Semantic caching for LLM responses.

Uses vector similarity to find cached responses for similar prompts.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
import aiohttp

from activelearning import EmbeddingService

logger = logging.getLogger(__name__)


class LLMCache:
    """
    LLM response cache with semantic similarity matching.

    Uses Qdrant vector DB to find cached responses for semantically
    similar prompts.
    """

    def __init__(
        self,
        qdrant_url: str,
        ollama_url: str,
        db: Any,
        hit_threshold: float = 0.95,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.db = db
        self.hit_threshold = hit_threshold

        self.collection_name = "llm_cache"

        # Embeddings go through the shared SDK service (reused session + LRU
        # cache) rather than a hand-rolled per-call Ollama request.
        self._embeddings = embedding_service or EmbeddingService(ollama_host=ollama_url)

        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0

    async def get(self, prompt: str, model: str = "deepseek-coder:6.7b") -> Optional[Dict]:
        """
        Get cached response for a prompt.

        Args:
            prompt: The LLM prompt
            model: Model name

        Returns:
            Cached response dict if found with high confidence, else None
        """
        try:
            # Get prompt embedding
            embedding = await self._get_embedding(prompt)

            # Search for similar cached prompts
            results = await self._search_cache(embedding)

            if not results:
                self._cache_misses += 1
                logger.debug(f"Cache miss: {prompt[:50]}...")
                return None

            # Check best match confidence
            best_match = results[0]
            confidence = best_match["score"]

            if confidence >= self.hit_threshold:
                # Cache hit!
                self._cache_hits += 1
                cached_data = best_match["payload"]
                cache_id = cached_data["id"]

                logger.info(f"Cache hit (confidence: {confidence:.3f}): {prompt[:50]}...")

                # Update hit count and timestamp
                await self._update_hit_stats(cache_id)

                return {
                    "response": cached_data["response"],
                    "model": cached_data["model"],
                    "cached_at": cached_data["cached_at"],
                    "hit_count": cached_data["hit_count"] + 1,
                    "confidence": confidence,
                }
            else:
                self._cache_misses += 1
                logger.debug(f"Cache miss (best match: {confidence:.3f}): {prompt[:50]}...")
                return None

        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            self._cache_misses += 1
            return None

    async def set(
        self,
        prompt: str,
        response: str,
        model: str = "deepseek-coder:6.7b",
    ) -> bool:
        """
        Cache an LLM response.

        Args:
            prompt: The LLM prompt
            response: The LLM response
            model: Model name

        Returns:
            bool indicating success
        """
        try:
            # Generate prompt hash for deduplication
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

            # Get prompt embedding
            embedding = await self._get_embedding(prompt)

            # Create cache entry
            import time
            cache_entry = {
                "id": prompt_hash,
                "prompt": prompt,
                "response": response,
                "model": model,
                "cached_at": int(time.time() * 1000),
                "hit_count": 0,
                "last_hit_at": None,
            }

            # Store in Qdrant
            async with aiohttp.ClientSession() as session:
                payload = {
                    "points": [
                        {
                            "id": prompt_hash,
                            "vector": embedding,
                            "payload": cache_entry,
                        }
                    ]
                }

                async with session.put(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as http_response:
                    if http_response.status in [200, 201]:
                        logger.debug(f"Cached response: {prompt[:50]}...")

                        # Also store in SQLite
                        await self._store_in_db(cache_entry)

                        return True
                    else:
                        logger.error(f"Failed to cache: {http_response.status}")
                        return False

        except Exception as e:
            logger.error(f"Cache store error: {e}", exc_info=True)
            return False

    async def _get_embedding(self, text: str) -> List[float]:
        """Get text embedding via the shared SDK embedding service."""
        try:
            return await self._embeddings.embed_text(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * 768

    async def close(self) -> None:
        """Release the shared embedding HTTP session."""
        await self._embeddings.close()

    async def _search_cache(self, embedding: List[float]) -> List[Dict]:
        """Search cache for similar prompts."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "vector": embedding,
                    "limit": 1,  # Only need best match
                    "with_payload": True,
                }

                async with session.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    return data.get("result", [])

        except Exception as e:
            logger.error(f"Cache search error: {e}")
            return []

    async def _store_in_db(self, cache_entry: Dict) -> None:
        """Store cache entry in SQLite."""
        try:
            await self.db.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (prompt_hash, prompt, response, model, cached_at, hit_count, last_hit_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_entry["id"],
                    cache_entry["prompt"],
                    cache_entry["response"],
                    cache_entry["model"],
                    cache_entry["cached_at"],
                    cache_entry["hit_count"],
                    cache_entry["last_hit_at"],
                ),
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error storing in DB: {e}")

    async def _update_hit_stats(self, cache_id: str) -> None:
        """Update cache hit statistics."""
        try:
            import time
            now = int(time.time() * 1000)

            await self.db.execute(
                """
                UPDATE llm_cache
                SET hit_count = hit_count + 1,
                    last_hit_at = ?
                WHERE prompt_hash = ?
                """,
                (now, cache_id),
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error updating hit stats: {e}")

    def get_metrics(self) -> Dict:
        """Get cache metrics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    async def invalidate(self, prompt_hash: str) -> bool:
        """Invalidate a specific cache entry."""
        try:
            # Delete from Qdrant
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/delete",
                    json={"points": [prompt_hash]},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        return False

            # Delete from SQLite
            await self.db.execute(
                "DELETE FROM llm_cache WHERE prompt_hash = ?",
                (prompt_hash,),
            )
            await self.db.commit()

            logger.info(f"Invalidated cache: {prompt_hash}")
            return True

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False
