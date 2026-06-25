"""Tests for parameterized tag recall (SQL injection safe)."""

import json

import pytest

from activelearning.database import Database
from memory.service import MemoryService


async def _make_service(db: Database) -> MemoryService:
    service = MemoryService.__new__(MemoryService)
    service.database = db
    return service


@pytest.fixture
async def tag_db(tmp_path):
    db = Database(str(tmp_path / "memory-tags.db"))
    await db.initialize()
    await db.execute("DROP TABLE IF EXISTS memory_episodes")
    await db.execute(
        """
        CREATE TABLE memory_episodes (
            id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            semantic_tags TEXT,
            utility_score REAL DEFAULT 0.0,
            data TEXT
        )
        """
    )
    await db.commit()
    return db


@pytest.mark.asyncio
async def test_recall_by_tags_finds_matching_episodes(tag_db):
    await tag_db.execute(
        """
        INSERT INTO memory_episodes
            (id, trace_id, timestamp, semantic_tags, utility_score, data)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("ep-1", "trace-1", 1000, json.dumps(["kitchen", "demo"]), 0.9, "{}"),
    )
    await tag_db.execute(
        """
        INSERT INTO memory_episodes
            (id, trace_id, timestamp, semantic_tags, utility_score, data)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("ep-2", "trace-2", 2000, json.dumps(["bedroom"]), 0.5, "{}"),
    )
    await tag_db.commit()

    service = await _make_service(tag_db)
    results = await service.recall_by_tags(["kitchen"], limit=10)

    assert len(results) == 1
    assert results[0].episode_id == "ep-1"
    assert "kitchen" in results[0].tags


@pytest.mark.asyncio
async def test_recall_by_tags_rejects_sql_injection(tag_db):
    await tag_db.execute(
        """
        INSERT INTO memory_episodes
            (id, trace_id, timestamp, semantic_tags, utility_score, data)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("ep-1", "trace-1", 1000, json.dumps(["safe"]), 0.9, "{}"),
    )
    await tag_db.execute(
        """
        INSERT INTO memory_episodes
            (id, trace_id, timestamp, semantic_tags, utility_score, data)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("ep-2", "trace-2", 2000, json.dumps(["other"]), 0.5, "{}"),
    )
    await tag_db.commit()

    service = await _make_service(tag_db)
    results = await service.recall_by_tags(["' OR 1=1 --"], limit=10)

    assert results == []


@pytest.mark.asyncio
async def test_recall_by_tags_empty_list_returns_empty(tag_db):
    service = await _make_service(tag_db)
    assert await service.recall_by_tags([]) == []
