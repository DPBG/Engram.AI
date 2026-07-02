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
from collections import OrderedDict

import aiohttp

from activelearning.llm import _OllamaSession

logger = logging.getLogger(__name__)


def zero_vector(dimensions: int) -> list[float]:
    """Return a zero embedding of the requested dimensionality."""
    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be a positive integer")
    return [0.0] * dimensions


def is_zero_vector(vector: list[float]) -> bool:
    """True when every component is zero (embedding failure sentinel)."""
    return bool(vector) and all(v == 0.0 for v in vector)


class EmbeddingService(_OllamaSession):
    """
    Generates text embeddings using Ollama's embedding model.
    Caches embeddings in memory to avoid regeneration.

    Reuses the shared :class:`~activelearning.llm._OllamaSession` for session
    lifecycle (``_get_session``/``close``), so the HTTP plumbing lives in one
    place alongside :class:`~activelearning.llm.LLMClient`.
    """

    def __init__(
        self,
        ollama_host: str | None = None,
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
        super().__init__(ollama_host)
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
            session = await self._get_session()
            async with session.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        "Embedding backend returned HTTP %s: %s",
                        response.status,
                        error_text,
                        extra={"cache_key": cache_key[:8], "model": self.model},
                    )
                    return self._zero_vector()

                result = await response.json()
                embedding = result.get("embedding")
                if not isinstance(embedding, list):
                    logger.error(
                        "Embedding backend returned invalid payload",
                        extra={"cache_key": cache_key[:8], "model": self.model},
                    )
                    return self._zero_vector()
                if len(embedding) != self.dimensions:
                    logger.error(
                        "Embedding dimension mismatch: expected %d, got %d",
                        self.dimensions,
                        len(embedding),
                        extra={"cache_key": cache_key[:8], "model": self.model},
                    )
                    return self._zero_vector()
                if not all(isinstance(v, (int, float)) for v in embedding):
                    logger.error(
                        "Embedding payload contains non-numeric components",
                        extra={"cache_key": cache_key[:8], "model": self.model},
                    )
                    return self._zero_vector()
                embedding = [float(v) for v in embedding]

        except Exception as e:
            logger.error(
                "Failed to get embedding: %s",
                e,
                extra={"cache_key": cache_key[:8], "model": self.model},
            )
            return self._zero_vector()

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

    def _zero_vector(self) -> list[float]:
        """Sentinel vector returned when embedding generation fails."""
        return zero_vector(self.dimensions)

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
            session = await self._get_session()
            async with session.get(
                f"{self.ollama_host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return self.model in models or any(self.model in m for m in models)
                return False
        except Exception as e:
            logger.warning(f"Embedding service not available: {e}")
            return False


# Global embedding service instance
_embedding_service: EmbeddingService | None = None


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
