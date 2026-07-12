"""Tests for OverrideProcessor persistence against the shared schema."""

from __future__ import annotations

import asyncio
import json
import sqlite3

from overrides.processor import OverrideProcessor

_SCHEMA = """
CREATE TABLE IF NOT EXISTS human_overrides (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    override_type TEXT NOT NULL CHECK(override_type IN ('operational', 'knowledge')),
    prompt TEXT NOT NULL,
    parameter_path TEXT,
    value TEXT,
    embedding_ref TEXT,
    verification_method TEXT CHECK(verification_method IN ('camera', 'microphone', 'button', 'none')),
    verification_confidence REAL,
    applied_at INTEGER,
    created_at INTEGER NOT NULL,
    metadata TEXT
);
"""


class _FakeDb:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(_SCHEMA)
        self.executed: list[tuple[str, tuple]] = []

    async def execute(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))
        self.conn.execute(sql, params)

    async def commit(self) -> None:
        self.conn.commit()


class _FakeBus:
    async def publish(self, subject: str, data: dict) -> None:
        pass


def _run(coro):
    return asyncio.run(coro)


def test_apply_override_inserts_matching_schema_columns():
    db = _FakeDb()
    processor = OverrideProcessor(_FakeBus(), db)

    result = _run(
        processor.apply_override(
            trace_id="trace-1",
            parameter="planner.mode",
            value="LEARNING",
            verified_by="camera",
            prompt="Switch to learning mode",
            verification_confidence=0.95,
        )
    )

    assert result["success"] is True
    assert len(db.executed) == 1
    _, params = db.executed[0]
    assert len(params) == 10

    row = db.conn.execute(
        "SELECT trace_id, override_type, prompt, parameter_path, value, "
        "verification_method, verification_confidence, applied_at, created_at "
        "FROM human_overrides"
    ).fetchone()
    assert row is not None
    assert row[0] == "trace-1"
    assert row[1] == "operational"
    assert row[2] == "Switch to learning mode"
    assert row[3] == "planner.mode"
    assert json.loads(row[4]) == "LEARNING"
    assert row[5] == "camera"
    assert row[6] == 0.95
    assert row[7] == row[8]  # applied_at == created_at
