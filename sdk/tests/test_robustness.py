"""Tests for issue #47 robustness fixes (reconnect, embeddings session, DB path)."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activelearning.database import Database, default_sqlite_path
from activelearning.embeddings import EmbeddingService
from activelearning.nats_client import EventBus


class TestDefaultSqlitePath:
    def test_honors_sqlite_path_env(self, monkeypatch):
        monkeypatch.setenv("SQLITE_PATH", "/custom/path/db.sqlite")
        assert default_sqlite_path() == "/custom/path/db.sqlite"

    def test_windows_default_uses_appdata(self, monkeypatch):
        monkeypatch.delenv("SQLITE_PATH", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\test\AppData\Local")
        with patch.object(sys, "platform", "win32"):
            path = default_sqlite_path()
        normalized = path.replace("\\", "/")
        assert normalized == "C:/Users/test/AppData/Local/Engram/sqlite/unified.db"

    def test_unix_default_uses_data_dir(self, monkeypatch):
        monkeypatch.delenv("SQLITE_PATH", raising=False)
        with patch.object(sys, "platform", "linux"):
            assert default_sqlite_path() == "/data/sqlite/unified.db"


class TestDatabaseInitialize:
    @pytest.mark.asyncio
    async def test_initialize_creates_parent_dir(self, tmp_path, monkeypatch):
        db_file = tmp_path / "nested" / "test.db"
        monkeypatch.setenv("SQLITE_PATH", str(db_file))
        db = Database()
        await db.initialize()
        try:
            assert db_file.exists()
        finally:
            await db.close()


class TestEmbeddingServiceSession:
    @pytest.mark.asyncio
    async def test_reuses_single_session(self):
        service = EmbeddingService()
        session_one = await service._get_session()
        session_two = await service._get_session()
        assert session_one is session_two
        await service.close()
        assert service._session is None


class TestForceReconnectRequestHandlers:
    @pytest.mark.asyncio
    async def test_force_reconnect_restores_request_handler_flag(self):
        bus = EventBus()

        async def pub_handler(_data):
            pass

        async def req_handler(_data, _msg):
            pass

        bus._handlers = {
            "events.pub": pub_handler,
            "safety.analyze": req_handler,
        }
        bus._request_handlers = {"safety.analyze"}
        bus._subscriptions = {
            "events.pub": MagicMock(),
            "safety.analyze": MagicMock(),
        }
        bus._nc = MagicMock()
        bus._nc.close = AsyncMock()

        subscribe_calls: list[dict] = []

        async def track_subscribe(subject, handler, **kwargs):
            subscribe_calls.append({"subject": subject, **kwargs})

        bus.connect = AsyncMock()
        bus.subscribe = AsyncMock(side_effect=track_subscribe)

        await bus.force_reconnect()

        assert bus.connect.await_count == 1
        assert len(subscribe_calls) == 2
        by_subject = {call["subject"]: call for call in subscribe_calls}
        assert by_subject["events.pub"]["is_request_handler"] is False
        assert by_subject["safety.analyze"]["is_request_handler"] is True
