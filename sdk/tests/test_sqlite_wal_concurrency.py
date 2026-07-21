"""SQLite WAL-mode behavior under concurrent service writes (issue #231).

Several services share one SQLite file in production (CLAUDE.md: "persist to
SQLite"; docker-compose.yml points kernel, safety-supervisor, beliefs,
planner, external-api, memory, cache, and coordinator at the same
``unified.db``). These tests exercise that scenario directly against the real
``Database`` class (no mocking of aiosqlite/sqlite3) and pin down the two
regimes that matter:

1. Concurrent *first-time* ``initialize()`` -- multiple services starting up
   at once against a brand-new file. See docs/sqlite-wal-concurrency.md for
   the measured failure rate this uncovered and the retry-with-backoff fix
   in ``Database.initialize()``.
2. Concurrent *steady-state* writes -- once the schema/WAL mode already
   exist, many services inserting concurrently.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid

import aiosqlite
import pytest

from activelearning.database import Database


def _row(component: str, i: int) -> dict:
    return {
        "id": f"{component}-{i}-{uuid.uuid4().hex[:8]}",
        "trace_id": f"trace-{component}",
        "timestamp": int(time.time() * 1000),
        "component": component,
        "action": "concurrency_test",
        "details": "{}",
    }


async def _write_n_rows(db: Database, component: str, n_rows: int) -> int:
    """Insert n_rows sequentially on one Database instance; return count that succeeded."""
    succeeded = 0
    for i in range(n_rows):
        await db.insert("audit_entries", _row(component, i))
        succeeded += 1
    return succeeded


class TestConcurrentFirstTimeInitialize:
    """Multiple services racing to create the schema on a brand-new shared file."""

    @pytest.mark.asyncio
    async def test_many_services_initializing_concurrently_all_succeed(self, tmp_path):
        db_path = str(tmp_path / "shared.db")
        n_services = 8

        dbs = [Database(db_path=db_path) for _ in range(n_services)]
        # Every real deployment scenario this models -- N service processes
        # each calling get_database() around the same time at startup --
        # relies on initialize() itself handling the race; nothing here
        # should need to catch an exception if the fix works.
        results = await asyncio.gather(
            *(asyncio.wait_for(db.initialize(), timeout=10) for db in dbs),
            return_exceptions=True,
        )

        failures = [r for r in results if isinstance(r, BaseException)]
        assert not failures, (
            f"{len(failures)}/{n_services} concurrent initialize() calls raised: "
            f"{[repr(f) for f in failures]}"
        )

        for db in dbs:
            await db.close()

    @pytest.mark.asyncio
    async def test_repeated_concurrent_initialize_trials_are_reliable(self, tmp_path):
        """Regression guard for the race this issue found: run several
        independent trials (fresh file each time) and require zero failures
        across all of them, not just a single lucky run."""
        n_trials = 15
        n_services = 4

        for trial in range(n_trials):
            db_path = str(tmp_path / f"shared-{trial}.db")
            dbs = [Database(db_path=db_path) for _ in range(n_services)]
            results = await asyncio.gather(
                *(asyncio.wait_for(db.initialize(), timeout=10) for db in dbs),
                return_exceptions=True,
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            assert not failures, f"trial {trial}: {[repr(f) for f in failures]}"
            for db in dbs:
                await db.close()


class TestConcurrentSteadyStateWrites:
    """Once the schema already exists, many services writing concurrently."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_from_many_connections_land_without_loss(self, tmp_path):
        db_path = str(tmp_path / "shared.db")

        # Seed the schema once, like a real deployment's first-ever startup.
        seed = Database(db_path=db_path)
        await seed.initialize()
        await seed.close()

        n_services = 12
        n_rows_per_service = 20
        dbs = [Database(db_path=db_path) for _ in range(n_services)]
        try:
            await asyncio.gather(*(db.initialize() for db in dbs))
            counts = await asyncio.gather(
                *(_write_n_rows(db, f"service-{i}", n_rows_per_service) for i, db in enumerate(dbs))
            )
        finally:
            for db in dbs:
                await db.close()

        assert counts == [n_rows_per_service] * n_services

        verify = await aiosqlite.connect(db_path)
        try:
            cursor = await verify.execute(
                "SELECT COUNT(*) FROM audit_entries WHERE action = 'concurrency_test'"
            )
            row = await cursor.fetchone()
        finally:
            await verify.close()

        assert row[0] == n_services * n_rows_per_service

    @pytest.mark.asyncio
    async def test_wal_mode_is_actually_active(self, tmp_path):
        db_path = str(tmp_path / "shared.db")
        db = Database(db_path=db_path)
        await db.initialize()
        try:
            cursor = await db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0].lower() == "wal"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_busy_timeout_is_set(self, tmp_path):
        db_path = str(tmp_path / "shared.db")
        db = Database(db_path=db_path)
        await db.initialize()
        try:
            cursor = await db.execute("PRAGMA busy_timeout")
            row = await cursor.fetchone()
            assert row[0] > 0
        finally:
            await db.close()


class TestObservedLimit:
    """Characterize the failure mode when contention genuinely exceeds what
    SQLite's busy handler will wait for -- proves it fails loud with a
    well-known, catchable exception, not silently or by hanging."""

    @pytest.mark.asyncio
    async def test_writer_holding_an_exclusive_lock_blocks_others_until_timeout(self, tmp_path):
        db_path = str(tmp_path / "shared.db")

        seed = Database(db_path=db_path)
        await seed.initialize()
        await seed.close()

        # A raw connection with a deliberately tiny busy_timeout, holding an
        # exclusive write lock open past that timeout -- simulates a stuck
        # writer, the scenario the issue calls "block services".
        blocker = await aiosqlite.connect(db_path)
        await blocker.execute("PRAGMA busy_timeout=100")
        await blocker.execute("BEGIN IMMEDIATE")
        await blocker.execute(
            "INSERT INTO audit_entries (id, trace_id, timestamp, component, action) "
            "VALUES ('blocker', 'trace-blocker', 0, 'blocker', 'hold_lock')"
        )

        contender = await aiosqlite.connect(db_path)
        await contender.execute("PRAGMA busy_timeout=100")
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                await contender.execute(
                    "INSERT INTO audit_entries (id, trace_id, timestamp, component, action) "
                    "VALUES ('contender', 'trace-contender', 0, 'contender', 'blocked_write')"
                )
        finally:
            await contender.close()
            await blocker.rollback()
            await blocker.close()
