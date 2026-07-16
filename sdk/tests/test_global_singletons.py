"""Regression coverage for the reset hooks added for issue #248.

``activelearning.database._db`` and ``activelearning.nats_client._global_bus``
are lazily-created module-level singletons. Before this issue, neither had a
reset hook, so a test that exercised the real singleton (instead of mocking
``get_database``/``get_event_bus``) leaked it to every later test in the same
pytest process. ``close_database()``/``close_event_bus()`` close the
underlying connection and reset the module global so the next
``get_database()``/``get_event_bus()`` call creates a fresh instance.

The autouse fixture in ``conftest.py`` already calls both after every test in
this suite; these tests exercise the reset functions directly so their
contract (idempotent no-op when nothing is open; genuinely fresh instance
after reset) is pinned independently of that fixture.
"""

from __future__ import annotations

import activelearning.database as database_module
import activelearning.nats_client as nats_client_module
from activelearning.database import close_database, get_database
from activelearning.nats_client import close_event_bus, get_event_bus


class TestDatabaseSingletonReset:
    async def test_close_database_is_noop_when_never_created(self):
        assert database_module._db is None
        await close_database()  # must not raise
        assert database_module._db is None

    async def test_get_database_returns_same_instance_until_closed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "singleton-test.db"))
        try:
            first = await get_database()
            second = await get_database()
            assert first is second
        finally:
            await close_database()

    async def test_close_database_resets_module_global(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "singleton-test.db"))
        await get_database()
        assert database_module._db is not None

        await close_database()

        assert database_module._db is None

    async def test_get_database_after_close_creates_fresh_instance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "singleton-test.db"))
        first = await get_database()
        await close_database()
        try:
            second = await get_database()
            assert second is not first
        finally:
            await close_database()


class TestEventBusSingletonReset:
    async def test_close_event_bus_is_noop_when_never_created(self):
        assert nats_client_module._global_bus is None
        await close_event_bus()  # must not raise
        assert nats_client_module._global_bus is None

    async def test_get_event_bus_returns_same_instance_until_closed(self, nats_url, monkeypatch):
        monkeypatch.setenv("NATS_URL", nats_url)
        try:
            first = await get_event_bus()
            second = await get_event_bus()
            assert first is second
        finally:
            await close_event_bus()

    async def test_close_event_bus_resets_module_global(self, nats_url, monkeypatch):
        monkeypatch.setenv("NATS_URL", nats_url)
        await get_event_bus()
        assert nats_client_module._global_bus is not None

        await close_event_bus()

        assert nats_client_module._global_bus is None

    async def test_get_event_bus_after_close_creates_fresh_instance(self, nats_url, monkeypatch):
        monkeypatch.setenv("NATS_URL", nats_url)
        first = await get_event_bus()
        await close_event_bus()
        try:
            second = await get_event_bus()
            assert second is not first
        finally:
            await close_event_bus()
