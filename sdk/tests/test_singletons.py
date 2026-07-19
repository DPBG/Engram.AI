"""Tests for the get_database()/get_event_bus() singletons and their reset
hooks (issue #248).

Both are process-lifetime by design in production — a server just exits and
the OS reclaims the connection/thread. But a test that leaves one open leaks
state into whatever test runs next: EventBus.__init__ binds an asyncio.Event
to the event loop running at construction time, and pytest-asyncio gives each
test function its own loop, so a stale bus reused across tests holds a
_connected Event tied to a dead loop. Database.initialize() opens an aiosqlite
connection backed by a non-daemon worker thread, which hangs interpreter
shutdown if never closed. close_database()/close_event_bus() reset the module
globals so no test can leak either singleton into the next one.
"""

from __future__ import annotations

import asyncio

import pytest

from activelearning import database as database_module
from activelearning import nats_client as nats_client_module
from activelearning.database import Database, close_database, get_database
from activelearning.nats_client import EventBus, close_event_bus, get_event_bus

# conftest.py's autouse _reset_global_singletons fixture closes both
# singletons after every test in this suite, so these tests don't need their
# own teardown — they're exercising that same reset path directly.


class TestDatabaseSingleton:
    async def test_get_database_returns_same_instance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
        db1 = await get_database()
        db2 = await get_database()
        assert db1 is db2
        assert isinstance(db1, Database)

    async def test_close_database_resets_singleton(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
        db1 = await get_database()
        await close_database()
        db2 = await get_database()
        assert db1 is not db2

    async def test_close_database_is_idempotent(self):
        await close_database()
        await close_database()  # must not raise when nothing is open

    async def test_close_database_closes_the_connection(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
        db = await get_database()
        await close_database()
        with pytest.raises(RuntimeError):
            await db.execute("SELECT 1")


class TestEventBusSingleton:
    async def test_get_event_bus_returns_same_instance(self, nats_url, monkeypatch):
        monkeypatch.setenv("NATS_URL", nats_url)
        bus1 = await get_event_bus()
        bus2 = await get_event_bus()
        assert bus1 is bus2
        assert isinstance(bus1, EventBus)

    async def test_close_event_bus_resets_singleton(self, nats_url, monkeypatch):
        monkeypatch.setenv("NATS_URL", nats_url)
        bus1 = await get_event_bus()
        await close_event_bus()
        bus2 = await get_event_bus()
        assert bus2 is not bus1

    async def test_close_event_bus_is_idempotent(self):
        await close_event_bus()
        await close_event_bus()  # must not raise when nothing is open

    async def test_reopened_bus_is_usable_after_close(self, nats_url, monkeypatch):
        """The issue #248 scenario: close + reopen must yield a bus whose
        asyncio.Event is bound to the current loop, not a stale one."""
        monkeypatch.setenv("NATS_URL", nats_url)
        bus1 = await get_event_bus()
        assert bus1._connected.is_set()
        await close_event_bus()

        bus2 = await get_event_bus()
        assert bus2._connected.is_set()
        await bus2.publish("test.singleton.reopen", {"ok": True})


class TestDatabaseSingletonConcurrency:
    async def test_concurrent_first_calls_construct_only_once(self, tmp_path, monkeypatch):
        """Regression for issue #254: concurrent first-callers during startup
        must not each construct their own Database instance."""
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))

        construct_count = 0
        real_database = database_module.Database

        class CountingDatabase(real_database):
            def __init__(self, *args, **kwargs):
                nonlocal construct_count
                construct_count += 1
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(database_module, "Database", CountingDatabase)

        results = await asyncio.gather(*(get_database() for _ in range(10)))

        assert construct_count == 1
        assert all(db is results[0] for db in results)


class TestEventBusSingletonConcurrency:
    async def test_concurrent_first_calls_construct_only_once(self, nats_url, monkeypatch):
        """Regression for issue #254: concurrent first-callers during startup
        must not each construct + connect their own EventBus instance."""
        monkeypatch.setenv("NATS_URL", nats_url)

        construct_count = 0
        real_event_bus = nats_client_module.EventBus

        class CountingEventBus(real_event_bus):
            def __init__(self, *args, **kwargs):
                nonlocal construct_count
                construct_count += 1
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(nats_client_module, "EventBus", CountingEventBus)

        results = await asyncio.gather(*(get_event_bus() for _ in range(10)))

        assert construct_count == 1
        assert all(bus is results[0] for bus in results)
