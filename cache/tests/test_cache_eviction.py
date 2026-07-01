"""Eviction tests against the *real* shared SQLite schema.

The unit tests in ``test_llm_cache.py`` mock the DB, so they never exercised
the ``llm_cache`` table and missed the schema mismatch that silently disabled
all eviction (issue #120). These tests wire the actual SDK ``Database`` (with
``schema.sql`` applied) into ``LLMCache``/``CacheInvalidator`` so the SQLite
mirror, the age sweep, and tag invalidation are exercised end to end.

Qdrant and embeddings are mocked; Uses ``asyncio.run`` (no pytest-asyncio) so it
runs under the bare-pytest governance CI lane.
"""

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

from activelearning import Database, current_timestamp


def _load(name, rel_path):
    path = os.path.join(os.path.dirname(__file__), "..", "src", "cache", rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LLMCache = _load("cache_llm_cache", "llm_cache.py").LLMCache
CacheInvalidator = _load("cache_invalidator", "invalidator.py").CacheInvalidator


async def _make_db():
    db = Database(":memory:")
    await db.initialize()
    return db


def _make_cache(db):
    store = MagicMock()
    store.upsert = AsyncMock()
    store.delete = AsyncMock()

    embeddings = MagicMock()
    embeddings.embed_text = AsyncMock(return_value=[0.1, 0.2])

    cache = LLMCache(
        qdrant_url="http://qdrant",
        db=db,
        store=store,
        embedding_service=embeddings,
    )
    return cache, store


async def _row(db, prompt_hash):
    return await db.fetchone(
        "SELECT prompt_hash, response, tags, created_at FROM llm_cache WHERE prompt_hash = ?",
        (prompt_hash,),
    )


# ── SQLite mirror matches the schema (was: every write silently failed) ───────


def test_set_mirrors_a_row_matching_the_real_schema():
    async def run():
        db = await _make_db()
        try:
            cache, _ = _make_cache(db)

            ok = await cache.set("prompt", "the answer", tags=["code_generation"])
            assert ok is True

            phash = hashlib.sha256(b"prompt").hexdigest()
            row = await _row(db, phash)
            assert row is not None, "mirror write must land in llm_cache"
            assert row["response"] == "the answer"
            assert json.loads(row["tags"]) == ["code_generation"]
            assert row["created_at"] is not None  # populates the NOT NULL column
        finally:
            await db.close()

    asyncio.run(run())


# ── tag invalidation ──────────────────────────────────────────────────────────


def test_invalidate_by_tag_evicts_only_matching_entries():
    async def run():
        db = await _make_db()
        try:
            cache, store = _make_cache(db)

            await cache.set("code prompt", "code resp", tags=["code_generation"])
            await cache.set("cfg prompt", "cfg resp", tags=["configuration"])

            deleted = await cache.invalidate_by_tag("code_generation")
            assert deleted == 1

            code_hash = hashlib.sha256(b"code prompt").hexdigest()
            cfg_hash = hashlib.sha256(b"cfg prompt").hexdigest()
            assert await _row(db, code_hash) is None  # matched tag → gone
            assert await _row(db, cfg_hash) is not None  # other tag → kept
            store.delete.assert_awaited_once_with("llm_cache", [code_hash])
        finally:
            await db.close()

    asyncio.run(run())


def test_invalidate_by_tag_no_match_is_a_noop():
    async def run():
        db = await _make_db()
        try:
            cache, store = _make_cache(db)
            await cache.set("p", "r", tags=["configuration"])

            assert await cache.invalidate_by_tag("code_generation") == 0
            store.delete.assert_not_called()
        finally:
            await db.close()

    asyncio.run(run())


# ── age sweep (was: SELECT cached_at raised, disabling the sweep) ──────────────


def test_delete_expired_removes_aged_entries_via_created_at():
    async def run():
        db = await _make_db()
        try:
            cache, store = _make_cache(db)
            await cache.set("old", "stale")

            old_hash = hashlib.sha256(b"old").hexdigest()
            # Backdate the entry well past max_age_days (7d).
            await db.execute(
                "UPDATE llm_cache SET created_at = ? WHERE prompt_hash = ?",
                (current_timestamp() - 30 * 24 * 60 * 60 * 1000, old_hash),
            )
            await db.commit()

            invalidator = CacheInvalidator(event_bus=MagicMock(), llm_cache=cache, db=db)
            deleted = await invalidator._delete_expired()

            assert deleted == 1
            assert await _row(db, old_hash) is None
            store.delete.assert_awaited_once_with("llm_cache", [old_hash])
        finally:
            await db.close()

    asyncio.run(run())


def test_delete_expired_keeps_fresh_entries():
    async def run():
        db = await _make_db()
        try:
            cache, _ = _make_cache(db)
            await cache.set("fresh", "current")

            invalidator = CacheInvalidator(event_bus=MagicMock(), llm_cache=cache, db=db)
            assert await invalidator._delete_expired() == 0
        finally:
            await db.close()

    asyncio.run(run())


# ── invalidator delegates event handlers to the tag index ─────────────────────


def test_event_handlers_invalidate_by_tag():
    async def run():
        cache = MagicMock()
        cache.invalidate_by_tag = AsyncMock(return_value=2)
        invalidator = CacheInvalidator(event_bus=MagicMock(), llm_cache=cache, db=MagicMock())

        await invalidator._on_code_deployed({})
        await invalidator._on_override_applied({"parameter": "x"})
        await invalidator._on_task_saved({"task_id": "t1"})

        cache.invalidate_by_tag.assert_any_await("code_generation")
        cache.invalidate_by_tag.assert_any_await("configuration")
        cache.invalidate_by_tag.assert_any_await("task_query")

    asyncio.run(run())
