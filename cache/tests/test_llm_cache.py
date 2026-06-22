"""Tests for LLMCache's embedding + vector-store integration.

Loads llm_cache.py directly so the cache package isn't imported, and injects a
mocked Qdrant store, embedding service, and DB. Uses asyncio.run (no
pytest-asyncio) so it runs under the bare-pytest governance CI lane.
"""

import asyncio
import importlib.util
import os
import sys
from unittest.mock import AsyncMock, MagicMock

from activelearning import QdrantHit, QdrantPoint

_LC_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "cache", "llm_cache.py")
_spec = importlib.util.spec_from_file_location("cache_llm_cache", _LC_PATH)
lc = importlib.util.module_from_spec(_spec)
sys.modules["cache_llm_cache"] = lc
_spec.loader.exec_module(lc)

LLMCache = lc.LLMCache


def _make(*, search_result=None, embed=(0.1, 0.2), embed_error=None, hit_threshold=0.95):
    store = MagicMock()
    store.search = AsyncMock(return_value=list(search_result or []))
    store.upsert = AsyncMock()
    store.delete = AsyncMock()
    store.ensure_collection = AsyncMock()
    store.close = AsyncMock()

    embeddings = MagicMock()
    if embed_error is not None:
        embeddings.embed_text = AsyncMock(side_effect=embed_error)
    else:
        embeddings.embed_text = AsyncMock(return_value=list(embed))

    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()

    cache = LLMCache(
        qdrant_url="http://qdrant",
        db=db,
        hit_threshold=hit_threshold,
        store=store,
        embedding_service=embeddings,
    )
    return cache, store, embeddings, db


def _hit(score, **payload):
    payload.setdefault("id", "h1")
    payload.setdefault("response", "answer")
    payload.setdefault("model", "deepseek-coder:6.7b")
    payload.setdefault("cached_at", 123)
    payload.setdefault("hit_count", 0)
    return QdrantHit(id=payload["id"], score=score, payload=payload)


# ── get ─────────────────────────────────────────────────────────────────────


def test_get_returns_hit_above_threshold():
    cache, store, embeddings, db = _make(search_result=[_hit(0.97)])

    result = asyncio.run(cache.get("prompt"))

    assert result["response"] == "answer"
    assert result["confidence"] == 0.97
    assert result["hit_count"] == 1  # incremented over the stored 0
    embeddings.embed_text.assert_awaited_once_with("prompt")
    store.search.assert_awaited_once_with("llm_cache", [0.1, 0.2], limit=1)
    db.execute.assert_awaited()  # hit stats updated
    assert cache.get_metrics()["cache_hits"] == 1


def test_get_miss_below_threshold():
    cache, _, _, _ = _make(search_result=[_hit(0.5)])
    assert asyncio.run(cache.get("prompt")) is None
    assert cache.get_metrics()["cache_misses"] == 1


def test_get_miss_no_results():
    cache, _, _, _ = _make(search_result=[])
    assert asyncio.run(cache.get("prompt")) is None
    assert cache.get_metrics()["cache_misses"] == 1


def test_get_embedding_failure_is_clean_miss_without_search():
    """A failed embedding becomes a miss, never a zero-vector search."""
    cache, store, _, _ = _make(embed_error=RuntimeError("Ollama down"))

    assert asyncio.run(cache.get("prompt")) is None
    store.search.assert_not_called()
    assert cache.get_metrics()["cache_misses"] == 1


# ── set ─────────────────────────────────────────────────────────────────────


def test_set_upserts_point_and_mirrors_to_db():
    cache, store, embeddings, db = _make(embed=(0.3, 0.4))

    ok = asyncio.run(cache.set("prompt", "the response", "deepseek-coder:6.7b"))

    assert ok is True
    embeddings.embed_text.assert_awaited_once_with("prompt")
    store.upsert.assert_awaited_once()
    collection, points = store.upsert.call_args.args
    assert collection == "llm_cache"
    point = points[0]
    assert isinstance(point, QdrantPoint)
    assert point.vector == [0.3, 0.4]
    assert point.payload["response"] == "the response"
    assert point.id == point.payload["id"]  # both the prompt hash
    db.execute.assert_awaited()  # mirrored to SQLite


def test_set_embedding_failure_returns_false_without_upsert():
    cache, store, _, db = _make(embed_error=RuntimeError("down"))

    ok = asyncio.run(cache.set("prompt", "resp"))

    assert ok is False
    store.upsert.assert_not_called()  # no zero-vector entry written
    db.execute.assert_not_called()


# ── invalidate ──────────────────────────────────────────────────────────────


def test_invalidate_deletes_from_store_and_db():
    cache, store, _, db = _make()

    ok = asyncio.run(cache.invalidate("hash-1"))

    assert ok is True
    store.delete.assert_awaited_once_with("llm_cache", ["hash-1"])
    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()


# ── setup ───────────────────────────────────────────────────────────────────


def test_setup_ensures_collection():
    cache, store, _, _ = _make()
    asyncio.run(cache.setup())
    store.ensure_collection.assert_awaited_once_with("llm_cache")
