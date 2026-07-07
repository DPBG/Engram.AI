"""Knowledge conflict detection for external API responses."""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.75
DEFAULT_WORD_OVERLAP_THRESHOLD = 0.3


class EmbeddingService(Protocol):
    async def embed_text(self, text: str) -> list[float]: ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors in [-1, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _is_zero_vector(vector: list[float]) -> bool:
    return bool(vector) and all(v == 0.0 for v in vector)


def word_overlap_conflict(
    local_text: str,
    external_text: str,
    *,
    threshold: float = DEFAULT_WORD_OVERLAP_THRESHOLD,
) -> bool:
    """Flag a conflict when vocabulary overlap falls below *threshold*."""
    if not local_text or not external_text:
        return False
    local_words = set(local_text.lower().split())
    external_words = set(external_text.lower().split())
    overlap = local_words & external_words
    min_size = min(len(local_words), len(external_words))
    if min_size == 0:
        return False
    return len(overlap) < min_size * threshold


async def detect_knowledge_conflict(
    external_response: str,
    local_knowledge: dict[str, Any],
    *,
    embedding_service: EmbeddingService | None = None,
    semantic_threshold: float = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
    word_overlap_threshold: float = DEFAULT_WORD_OVERLAP_THRESHOLD,
) -> bool:
    """Return True when external text likely contradicts local knowledge."""
    local_text = str(local_knowledge.get("description", ""))
    if not local_text or not external_response:
        return False

    if embedding_service is not None:
        try:
            local_vec = await embedding_service.embed_text(local_text)
            external_vec = await embedding_service.embed_text(external_response)
            if not _is_zero_vector(local_vec) and not _is_zero_vector(external_vec):
                similarity = cosine_similarity(local_vec, external_vec)
                return similarity < semantic_threshold
        except Exception as exc:
            logger.warning("Semantic conflict check failed, using word overlap: %s", exc)

    return word_overlap_conflict(
        local_text,
        external_response,
        threshold=word_overlap_threshold,
    )
