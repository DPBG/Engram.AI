"""End-to-end tests for CacheInvalidator's tag- and age-based eviction.

These run against a real in-memory SQLite database (the production schema, via
the SDK's ``Database``) and a real ``LLMCache`` with a mocked Qdrant store and
embedding service. That exercises the actual ``json_each`` tag query *and* the
schema/code alignment that makes the SQLite mirror writable — the silent
mismatch that previously made both the SQLite store and every invalidation path
a no-op. Uses ``asyncio.run`` (no pytest-asyncio) so it runs under the
bare-pytest governance CI lane.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from activelearning import Database

from cache.invalidator import CacheInvalidator
from cache.llm_cache import CacheTag, LLMCache


def _run(body):
    """Run an async test ``body(inv, cache, db, store)`` against a fresh
    invalidator + LLMCache backed by one real in-memory database, closing the
    connection afterwards so aiosqlite's worker thread shuts down cleanly."""

    async def runner():
        db = Database(":memory:")
        await db.initialize()

        store = MagicMock()
        store.upsert = AsyncMock()
        store.delete = AsyncMock()
        store.ensure_collection = AsyncMock()
        store.close = AsyncMock()

        embeddings = MagicMock()
        embeddings.embed_text = AsyncMock(return_value=[0.1, 0.2])

        cache = LLMCache(
            qdrant_url="http://qdrant",
            db=db,
            store=store,
            embedding_service=embeddings,
        )
        invalidator = CacheInvalidator(event_bus=MagicMock(), llm_cache=cache, db=db)
        try:
            await body(invalidator, cache, db, store)
        finally:
            await db.close()

    asyncio.run(runner())


async def _responses(db):
    """Responses still present in the cache, sorted — entries are identified by
    their (test-controlled) response text rather than opaque prompt hashes."""
    rows = await db.fetchall("SELECT response FROM llm_cache")
    return sorted(row[0] for row in rows)


# ── tag invalidation ─────────────────────────────────────────────────────────


def test_event_handlers_evict_only_their_tag():
    async def body(inv, cache, db, _store):
        await cache.set("p1", "CODE", tags=[CacheTag.CODE_GENERATION])
        await cache.set("p2", "CFG", tags=[CacheTag.CONFIGURATION])
        await cache.set("p3", "TASK", tags=[CacheTag.TASK_QUERY])
        await cache.set("p4", "PLAIN")  # untagged
        assert await _responses(db) == ["CFG", "CODE", "PLAIN", "TASK"]

        await inv._on_code_deployed({"path": "x"})
        assert await _responses(db) == ["CFG", "PLAIN", "TASK"]

        await inv._on_override_applied({"parameter": "thr"})
        assert await _responses(db) == ["PLAIN", "TASK"]

        await inv._on_task_saved({"task_id": "t"})
        assert await _responses(db) == ["PLAIN"]  # only the untagged entry left

    _run(body)


def test_invalidate_by_tag_forwards_to_qdrant_and_counts():
    async def body(inv, cache, db, store):
        await cache.set("p1", "CODE", tags=[CacheTag.CODE_GENERATION])
        await cache.set("p2", "PLAIN")

        evicted = await inv._invalidate_by_tag(CacheTag.CODE_GENERATION)

        assert evicted == 1
        assert await _responses(db) == ["PLAIN"]
        store.delete.assert_awaited_once()  # also removed from the vector store

    _run(body)


def test_tag_match_is_exact_not_substring():
    """``task_query`` must not match ``task_query_extended`` (no LIKE)."""

    async def body(inv, cache, db, _store):
        await cache.set("p1", "EXTENDED", tags=["task_query_extended"])
        await cache.set("p2", "TASK", tags=[CacheTag.TASK_QUERY])

        evicted = await inv._invalidate_by_tag(CacheTag.TASK_QUERY)

        assert evicted == 1
        assert await _responses(db) == ["EXTENDED"]

    _run(body)


def test_entry_with_multiple_tags_evicted_by_any_of_them():
    async def body(inv, cache, db, _store):
        await cache.set("p1", "MULTI", tags=[CacheTag.CODE_GENERATION, CacheTag.CONFIGURATION])

        evicted = await inv._invalidate_by_tag(CacheTag.CONFIGURATION)

        assert evicted == 1
        assert await _responses(db) == []

    _run(body)


def test_invalidate_by_tag_with_no_matches_is_noop():
    async def body(inv, cache, db, store):
        await cache.set("p1", "PLAIN")  # untagged
        await cache.set("p2", "CFG", tags=[CacheTag.CONFIGURATION])

        evicted = await inv._invalidate_by_tag(CacheTag.CODE_GENERATION)

        assert evicted == 0
        assert await _responses(db) == ["CFG", "PLAIN"]
        store.delete.assert_not_called()

    _run(body)


# ── persistence + age-based eviction (regressions for the silent mismatch) ────


def test_set_persists_row_to_sqlite():
    """Regression: the mirror INSERT used columns the schema lacked, so it
    failed silently and nothing was ever stored — leaving invalidation no rows
    to act on."""

    async def body(_inv, cache, db, _store):
        await cache.set("hello world", "hi there", "model-x", tags=[CacheTag.TASK_QUERY])

        row = await db.fetchone(
            "SELECT prompt, response, model, tags, cached_at, hit_count FROM llm_cache"
        )
        assert row is not None
        assert row[0] == "hello world"
        assert row[1] == "hi there"
        assert row[2] == "model-x"
        assert row[3] == json.dumps([CacheTag.TASK_QUERY])
        assert row[4] > 0  # cached_at populated
        assert row[5] == 0  # hit_count default

    _run(body)


def test_delete_hashes_counts_only_successful_invalidations():
    async def run():
        cache = MagicMock()
        cache.invalidate = AsyncMock(side_effect=[True, False, True])
        inv = CacheInvalidator(event_bus=MagicMock(), llm_cache=cache, db=MagicMock())

        assert await inv._delete_hashes(["a", "b", "c"]) == 2

    asyncio.run(run())


def test_evict_expired_with_nothing_to_evict():
    async def body(inv, cache, db, _store):
        await cache.set("fresh", "FRESH")  # within the age window

        assert await inv._evict_expired() == 0
        assert await _responses(db) == ["FRESH"]

    _run(body)


def test_evict_expired_drops_old_entries_and_keeps_fresh():
    """Regression: the cleanup query referenced ``cached_at`` which the old
    schema spelled ``created_at``, so age-based eviction always errored out."""

    async def body(inv, cache, db, _store):
        await cache.set("fresh", "FRESH", tags=[CacheTag.TASK_QUERY])
        await cache.set("stale", "STALE")
        # Backdate the stale entry well past max_age_days.
        await db.execute(
            "UPDATE llm_cache SET cached_at = 1, last_hit_at = NULL WHERE response = ?",
            ("STALE",),
        )
        await db.commit()

        evicted = await inv._evict_expired()

        assert evicted == 1
        assert await _responses(db) == ["FRESH"]

    _run(body)


def test_evict_expired_drops_never_hit_entry_past_unused_age():
    """A never-hit entry idle past unused_age_days is evicted even while it is
    younger than max_age_days (its cached_at stands in for last_hit_at)."""

    async def body(inv, cache, db, _store):
        await cache.set("idle", "IDLE")  # never hit
        four_days_ms = 4 * 24 * 60 * 60 * 1000  # > unused_age (3d), < max_age (7d)
        await db.execute(
            "UPDATE llm_cache SET cached_at = cached_at - ?, last_hit_at = NULL WHERE response = ?",
            (four_days_ms, "IDLE"),
        )
        await db.commit()

        assert await inv._evict_expired() == 1
        assert await _responses(db) == []

    _run(body)


def test_invalidate_all_clears_every_entry():
    async def body(inv, cache, db, _store):
        await cache.set("p1", "A", tags=[CacheTag.CODE_GENERATION])
        await cache.set("p2", "B")

        deleted = await inv.invalidate_all()

        assert deleted == 2
        assert await _responses(db) == []

    _run(body)


# ── error handling (a failing query must not crash the caller) ────────────────


def _failing_invalidator():
    """An invalidator whose every DB query raises, with a cache that records
    whether any eviction was attempted."""
    db = MagicMock()
    db.execute = AsyncMock(side_effect=RuntimeError("db down"))
    cache = MagicMock()
    cache.invalidate = AsyncMock(return_value=True)
    inv = CacheInvalidator(event_bus=MagicMock(), llm_cache=cache, db=db)
    return inv, cache


def test_invalidate_by_tag_swallows_db_error():
    inv, cache = _failing_invalidator()

    assert asyncio.run(inv._invalidate_by_tag(CacheTag.CONFIGURATION)) == 0
    cache.invalidate.assert_not_called()


def test_invalidate_all_swallows_db_error():
    inv, cache = _failing_invalidator()

    assert asyncio.run(inv.invalidate_all()) == 0
    cache.invalidate.assert_not_called()


# ── periodic loop ─────────────────────────────────────────────────────────────


def test_periodic_cleanup_evicts_then_exits_on_cancel():
    """One pass evicts, the next sleep is cancelled (clean shutdown)."""

    async def body(inv, cache, db, _store):
        await cache.set("stale", "STALE")
        await db.execute(
            "UPDATE llm_cache SET cached_at = 1, last_hit_at = NULL WHERE response = ?",
            ("STALE",),
        )
        await db.commit()
        inv._running = True

        calls = {"n": 0}

        async def fake_sleep(_seconds):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", fake_sleep):
            await inv._periodic_cleanup()

        assert await _responses(db) == []  # evicted on the first pass

    _run(body)


def test_periodic_cleanup_keeps_running_after_an_error():
    """A failed pass is logged and swallowed; the loop exits once stopped."""

    async def body(inv, _cache, _db, _store):
        inv._running = True

        async def boom():
            inv._running = False  # let the loop exit after this pass
            raise RuntimeError("boom")

        inv._evict_expired = boom
        with patch("asyncio.sleep", AsyncMock()):
            await inv._periodic_cleanup()  # returns without raising

    _run(body)
