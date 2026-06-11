"""
Embedding service for vector operations.

Uses Ollama's embedding models to generate text embeddings for:
- Task lookup
- LLM cache
- Override search
- Memory episodes
"""

import hashlib
import logging
import os
from collections import OrderedDict
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates text embeddings using Ollama's embedding model.
    Caches embeddings in memory to avoid regeneration.
    """

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
    ):
        """
        Initialize the embedding service.

        Args:
            ollama_host: Ollama server URL (defaults to OLLAMA_URL env var)
            model: Embedding model name (default: nomic-embed-text)
            dimensions: Expected embedding dimensions
        """
        self.ollama_host = ollama_host or os.environ.get(
            "OLLAMA_URL", "http://localhost:11434"
        )
        self.model = model
        self.dimensions = dimensions
        # Insertion-ordered map used as a true LRU: a cache hit promotes the
        # key to the most-recently-used end, and eviction drops the LRU front.
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_max_size = 10000  # Max cached embeddings

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Check memory cache first
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            logger.debug(f"Embedding cache hit for key {cache_key[:8]}...")
            self._cache.move_to_end(cache_key)  # mark most-recently-used
            return self._cache[cache_key]

        # Call Ollama embedding endpoint
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama embedding error: {error_text}")

                    result = await response.json()
                    embedding = result["embedding"]

        except aiohttp.ClientError as e:
            logger.error(f"Failed to get embedding: {e}")
            raise RuntimeError(f"Embedding service unavailable: {e}") from e

        # Cache the result
        self._add_to_cache(cache_key, embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # TODO: Ollama may support batch embeddings in the future
        # For now, we process sequentially with caching
        return [await self.embed_text(t) for t in texts]

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _add_to_cache(self, key: str, embedding: list[float]) -> None:
        """Add an embedding to the cache with true LRU eviction."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = embedding
        if len(self._cache) > self._cache_max_size:
            # Evict the least-recently-used entry (front of the OrderedDict).
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    async def is_available(self) -> bool:
        """Check if the embedding service is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        return self.model in models or any(
                            self.model in m for m in models
                        )
                    return False
        except Exception as e:
            logger.warning(f"Embedding service not available: {e}")
            return False


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


async def embed_text(text: str) -> list[float]:
    """Convenience function for embedding text."""
    service = get_embedding_service()
    return await service.embed_text(text)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convenience function for batch embedding."""
    service = get_embedding_service()
    return await service.embed_batch(texts)
