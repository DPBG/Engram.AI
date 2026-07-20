"""Regression tests for Database.insert() row-ID return value (issue #245).

``insert()`` is documented as returning the ID of the inserted row, but used
to return ``data.get("id", "")`` — the caller's own input. Callers that rely
on AUTOINCREMENT and omit ``id`` therefore got ``""`` back.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from activelearning.database import Database


def _run(coro):
    return asyncio.run(coro)


async def _with_db(tmp_path: Path):
    db = Database(str(tmp_path / "insert-rowid.db"))
    await db.initialize()
    return db


def test_insert_returns_autoincrement_lastrowid() -> None:
    """When the caller omits ``id``, return SQLite's assigned lastrowid."""

    async def _body() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = await _with_db(Path(tmp))
            try:
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS auto_rows ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  payload TEXT NOT NULL"
                    ")"
                )
                await db.commit()

                first = await db.insert("auto_rows", {"payload": "alpha"})
                second = await db.insert("auto_rows", {"payload": "beta"})

                assert first == "1"
                assert second == "2"
                assert first != ""
                assert second != ""

                row = await db.fetchone("SELECT id, payload FROM auto_rows WHERE id = ?", (1,))
                assert row is not None
                assert row["payload"] == "alpha"
            finally:
                await db.close()

    _run(_body())


def test_insert_returns_caller_supplied_text_id() -> None:
    """When the caller supplies ``id``, echo that value (existing callers)."""

    async def _body() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = await _with_db(Path(tmp))
            try:
                returned = await db.insert(
                    "audit_entries",
                    {
                        "id": "audit-explicit-1",
                        "trace_id": "trace-1",
                        "timestamp": 1_700_000_000_000,
                        "component": "test",
                        "action": "insert",
                        "details": "{}",
                    },
                )
                assert returned == "audit-explicit-1"

                row = await db.fetchone(
                    "SELECT id FROM audit_entries WHERE id = ?",
                    ("audit-explicit-1",),
                )
                assert row is not None
                assert row["id"] == "audit-explicit-1"
            finally:
                await db.close()

    _run(_body())


def test_insert_omitted_id_returns_nonzero_lastrowid() -> None:
    """Omitting ``id`` must never return the empty string on AUTOINCREMENT tables."""

    async def _body() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = await _with_db(Path(tmp))
            try:
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS auto_rows ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  payload TEXT NOT NULL"
                    ")"
                )
                await db.commit()

                returned = await db.insert("auto_rows", {"payload": "gamma"})
                assert returned != ""
                assert returned.isdigit()
                assert int(returned) >= 1
            finally:
                await db.close()

    _run(_body())
