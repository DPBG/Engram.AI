"""Tests for structured EventBus connection-lifecycle logging (issue #256)."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activelearning.connection_logging import (
    CONNECTION_EVENT,
    TRANSITION_CLOSED,
    TRANSITION_CLOSING,
    TRANSITION_CONNECTED,
    TRANSITION_CONNECTING,
    TRANSITION_DISCONNECTED,
    TRANSITION_FORCE_RECONNECT_COMPLETE,
    TRANSITION_FORCE_RECONNECT_START,
    TRANSITION_RECONNECTED,
    log_connection_event,
)
from activelearning.nats_client import EventBus


def _parse_lifecycle_logs(caplog) -> list[dict]:
    events = []
    for record in caplog.records:
        try:
            payload = json.loads(record.message)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == CONNECTION_EVENT:
            events.append(payload)
    return events


class TestLogConnectionEvent:
    def test_emits_json_with_required_fields(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            payload = log_connection_event(
                TRANSITION_CONNECTED,
                client_name="kernel",
                nats_url="nats://localhost:4222",
                creds_mode="none",
            )

        assert payload == {
            "event": CONNECTION_EVENT,
            "transition": TRANSITION_CONNECTED,
            "client_name": "kernel",
            "nats_url": "nats://localhost:4222",
            "creds_mode": "none",
        }
        assert len(caplog.records) == 1
        assert json.loads(caplog.records[0].message) == payload

    def test_omits_none_fields(self) -> None:
        payload = log_connection_event(
            TRANSITION_CONNECTING,
            client_name="planner",
            nats_url="nats://127.0.0.1:4222",
            creds_path=None,
        )
        assert "creds_path" not in payload


class TestEventBusLifecycleLogging:
    @staticmethod
    def _mock_nc() -> MagicMock:
        nc = MagicMock()
        nc.is_connected = True
        nc.jetstream.return_value = AsyncMock()
        return nc

    @pytest.mark.asyncio
    async def test_connect_emits_structured_connecting_and_connected(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_nc = self._mock_nc()
        with (
            caplog.at_level(logging.INFO),
            patch(
                "activelearning.nats_client.nats.connect",
                new=AsyncMock(return_value=mock_nc),
            ),
            patch.object(EventBus, "_ensure_safety_stream", new=AsyncMock()),
        ):
            bus = EventBus(nats_url="nats://localhost:4222", name="test-bus")
            await bus.connect()

        events = _parse_lifecycle_logs(caplog)
        transitions = [event["transition"] for event in events]
        assert transitions == [TRANSITION_CONNECTING, TRANSITION_CONNECTED]
        assert events[0]["client_name"] == "test-bus"
        assert events[0]["creds_mode"] == "none"
        assert events[1]["transition"] == TRANSITION_CONNECTED

    @pytest.mark.asyncio
    async def test_close_emits_structured_closing_and_closed(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_nc = self._mock_nc()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()

        bus = EventBus(nats_url="nats://localhost:4222", name="test-bus")
        bus._nc = mock_nc

        with caplog.at_level(logging.INFO):
            await bus.close()

        events = _parse_lifecycle_logs(caplog)
        assert [event["transition"] for event in events] == [
            TRANSITION_CLOSING,
            TRANSITION_CLOSED,
        ]

    @pytest.mark.asyncio
    async def test_disconnect_and_reconnect_callbacks_emit_structured_events(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        bus = EventBus(nats_url="nats://localhost:4222", name="callback-bus")

        with caplog.at_level(logging.WARNING):
            await bus._disconnected_callback()
        with caplog.at_level(logging.INFO):
            await bus._reconnected_callback()

        events = _parse_lifecycle_logs(caplog)
        assert [event["transition"] for event in events] == [
            TRANSITION_DISCONNECTED,
            TRANSITION_RECONNECTED,
        ]

    @pytest.mark.asyncio
    async def test_force_reconnect_emits_structured_start_and_complete(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        bus = EventBus(nats_url="nats://localhost:4222", name="reconnect-bus")
        bus._handlers["test.subject"] = AsyncMock()
        bus._nc = MagicMock()
        bus._nc.close = AsyncMock()

        with (
            caplog.at_level(logging.INFO),
            patch.object(EventBus, "connect", new=AsyncMock()),
            patch.object(EventBus, "subscribe", new=AsyncMock()),
        ):
            await bus.force_reconnect()

        events = _parse_lifecycle_logs(caplog)
        assert [event["transition"] for event in events] == [
            TRANSITION_FORCE_RECONNECT_START,
            TRANSITION_FORCE_RECONNECT_COMPLETE,
        ]
        assert events[0]["subscription_count"] == 1
