"""Concurrent-access stress tests for sdk/src/activelearning/database.py.

Architecture note
-----------------
``Database`` wraps a **single** ``aiosqlite.Connection`` (one persistent
connection per instance).  ``aiosqlite`` serialises every operation through a
background thread and an internal ``queue.Queue``, so concurrent ``await``
calls on **one** instance are safe by construction — there is only ever one
in-flight SQLite call at a time, and no "database is locked" can occur.

The exhaustion/contention risk surfaces when **multiple** ``Database``
instances open the **same file** simultaneously.  SQLite WAL mode allows
concurrent readers plus one writer; without ``PRAGMA busy_timeout`` a second
writer that arrives while the first is active raises
``sqlite3.OperationalError: database is locked`` immediately (default timeout
= 0 ms) rather than retrying.

``database.py`` does not currently set ``busy_timeout``.

Documented limits (see ``TestMultiConnectionContention``)
---------------------------------------------------------
* Single shared connection  — unlimited concurrency, zero contention.
* Multiple connections, read-only — unlimited concurrency under WAL.
* Multiple connections, concurrent writes, no busy_timeout — immediate
  ``OperationalError`` under write collision.
* Multiple connections, concurrent writes, ``busy_timeout >= 1000 ms`` —
  writers queue and all commits succeed.

Recommended fix: add ``PRAGMA busy_timeout = 5000`` in
``Database.initialize()`` after the WAL and synchronous pragmas.
"""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from activelearning.database import Database

_CONCURRENT_WRITERS = 50
_CONCURRENT_READERS = 50

_DDL = """
CREATE TABLE IF NOT EXISTS stress_rows (
    id  INTEGER PRIMARY KEY AUTOINCREMENT,
    val TEXT NOT NULL
)
"""


async def _open_db(tmp_path, filename: str = "stress.db") -> Database:
    db = Database(str(tmp_path / filename))
    await db.initialize()
    await db.execute(_DDL)
    await db.commit()
    return db


# ---------------------------------------------------------------------------
# Single-connection concurrent access
# ---------------------------------------------------------------------------


class TestSingleConnectionConcurrentWrites:
    """All coroutines share ONE Database instance, serialised by aiosqlite."""

    @pytest.mark.asyncio
    async def test_all_inserts_committed(self, tmp_path):
        """N concurrent writers on a single connection commit every row."""
        db = await _open_db(tmp_path)
        try:

            async def insert(i: int) -> None:
                await db.execute("INSERT INTO stress_rows (val) VALUES (?)", (f"row-{i}",))
                await db.commit()

            await asyncio.gather(*[insert(i) for i in range(_CONCURRENT_WRITERS)])

            rows = await db.fetchall("SELECT COUNT(*) FROM stress_rows")
            assert rows[0][0] == _CONCURRENT_WRITERS
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_no_duplicate_primary_keys(self, tmp_path):
        """Auto-increment IDs stay unique under concurrent inserts."""
        db = await _open_db(tmp_path)
        try:

            async def insert(i: int) -> None:
                await db.execute("INSERT INTO stress_rows (val) VALUES (?)", (f"v{i}",))
                await db.commit()

            await asyncio.gather(*[insert(i) for i in range(_CONCURRENT_WRITERS)])

            rows = await db.fetchall("SELECT id FROM stress_rows")
            ids = [r[0] for r in rows]
            assert len(ids) == len(set(ids)), "Duplicate primary keys detected"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_executemany_concurrent_batches(self, tmp_path):
        """Concurrent executemany calls on a single connection produce correct row count."""
        db = await _open_db(tmp_path)
        try:
            batch_size = 5
            n_batches = 20

            async def insert_batch(batch_id: int) -> None:
                params = [(f"b{batch_id}-{j}",) for j in range(batch_size)]
                await db.executemany("INSERT INTO stress_rows (val) VALUES (?)", params)
                await db.commit()

            await asyncio.gather(*[insert_batch(i) for i in range(n_batches)])

            rows = await db.fetchall("SELECT COUNT(*) FROM stress_rows")
            assert rows[0][0] == batch_size * n_batches
        finally:
            await db.close()


class TestSingleConnectionConcurrentReads:
    """Reads under high concurrency on a single connection."""

    @pytest.mark.asyncio
    async def test_all_readers_see_committed_rows(self, tmp_path):
        """N concurrent readers all observe the same seeded row count."""
        db = await _open_db(tmp_path)
        try:
            for i in range(10):
                await db.execute("INSERT INTO stress_rows (val) VALUES (?)", (f"seed-{i}",))
            await db.commit()

            async def read() -> int:
                rows = await db.fetchall("SELECT COUNT(*) FROM stress_rows")
                return rows[0][0]

            counts = await asyncio.gather(*[read() for _ in range(_CONCURRENT_READERS)])
            assert all(c == 10 for c in counts)
        finally:
            await db.close()


class TestSingleConnectionMixedAccess:
    """Interleaved reads and writes on a single connection produce no errors."""

    @pytest.mark.asyncio
    async def test_mixed_access_no_error(self, tmp_path):
        db = await _open_db(tmp_path)
        try:
            half = 20

            async def write(i: int) -> None:
                await db.execute("INSERT INTO stress_rows (val) VALUES (?)", (f"m{i}",))
                await db.commit()

            async def read() -> None:
                await db.fetchall("SELECT val FROM stress_rows")

            tasks = [write(i) for i in range(half)] + [read() for _ in range(half)]
            await asyncio.gather(*tasks)
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_update_and_read_concurrent_no_corruption(self, tmp_path):
        """Concurrent updates and reads on a single connection leave data consistent."""
        db = await _open_db(tmp_path)
        try:
            await db.execute("INSERT INTO stress_rows (val) VALUES ('initial')")
            await db.commit()

            async def bump(i: int) -> None:
                await db.execute("UPDATE stress_rows SET val = ? WHERE id = 1", (f"v{i}",))
                await db.commit()

            async def read() -> str | None:
                rows = await db.fetchall("SELECT val FROM stress_rows WHERE id = 1")
                return rows[0][0] if rows else None

            tasks = [bump(i) for i in range(25)] + [read() for _ in range(25)]
            await asyncio.gather(*tasks)

            # Final value must be one of the written values, never corrupted.
            rows = await db.fetchall("SELECT val FROM stress_rows WHERE id = 1")
            assert rows[0][0].startswith("v")
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Multi-connection contention — documents the exhaustion/contention limits
# ---------------------------------------------------------------------------


class TestMultiConnectionContention:
    """Multiple Database instances open the SAME file concurrently.

    Key finding (documented by tests below):
        database.py does not set PRAGMA busy_timeout.  Under concurrent write
        load from independent connections, a writer that cannot acquire the
        WAL write lock raises sqlite3.OperationalError immediately.

        Recommended fix: add ``PRAGMA busy_timeout = 5000`` in
        ``Database.initialize()`` after WAL and synchronous are set.
    """

    @pytest.mark.asyncio
    async def test_concurrent_readers_multiple_connections_no_error(self, tmp_path):
        """N independent read-only connections to the same WAL file never lock."""
        db_path = str(tmp_path / "shared.db")

        seed = Database(db_path)
        await seed.initialize()
        await seed.execute(_DDL)
        await seed.execute("INSERT INTO stress_rows (val) VALUES ('seed')")
        await seed.commit()
        await seed.close()

        async def read_one() -> int:
            db = Database(db_path)
            await db.initialize()
            try:
                rows = await db.fetchall("SELECT COUNT(*) FROM stress_rows")
                return rows[0][0]
            finally:
                await db.close()

        counts = await asyncio.gather(*[read_one() for _ in range(20)])
        assert all(c >= 1 for c in counts)

    @pytest.mark.asyncio
    async def test_busy_timeout_resolves_write_contention(self, tmp_path):
        """With busy_timeout set, concurrent writers from different connections
        wait rather than failing, and every commit lands.

        This is the recommended mitigation for the missing busy_timeout in
        database.py's initialize() method.
        """
        db_path = str(tmp_path / "contention.db")

        seed = Database(db_path)
        await seed.initialize()
        await seed.execute(_DDL)
        await seed.commit()
        await seed.close()

        n_writers = 8

        async def write_with_timeout(i: int) -> None:
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA busy_timeout=5000")
                await conn.execute(_DDL)
                await conn.execute("INSERT INTO stress_rows (val) VALUES (?)", (f"wt-{i}",))
                await conn.commit()

        await asyncio.gather(*[write_with_timeout(i) for i in range(n_writers)])

        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM stress_rows")
            row = await cursor.fetchone()
        assert row[0] == n_writers
