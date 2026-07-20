"""Tests for JetStream consumer lag alerting (issue #224)."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from activelearning.consumer_lag import (
    CONSUMER_LAG_EVENT,
    ConsumerLagSnapshot,
    emit_consumer_lag_alert,
    lag_threshold,
    should_emit_lag_alert,
    should_monitor_consumer,
)
from activelearning.nats_client import SAFETY_STREAM_NAME, EventBus


class TestConsumerLagHelpers:
    def test_should_monitor_skips_ephemeral_waiters(self) -> None:
        assert should_monitor_consumer("kernel-action-proposals") is True
        assert should_monitor_consumer("waiter-action-abc") is False
        assert should_monitor_consumer("waiter-code-xyz") is False

    def test_should_emit_at_threshold_and_multiples(self) -> None:
        assert should_emit_lag_alert(10, threshold=10, alerts_emitted=0) is True
        assert should_emit_lag_alert(15, threshold=10, alerts_emitted=1) is False
        assert should_emit_lag_alert(20, threshold=10, alerts_emitted=1) is True

    def test_emit_consumer_lag_alert_json(self, caplog: pytest.LogCaptureFixture) -> None:
        snapshot = ConsumerLagSnapshot(
            stream="SAFETY_CRITICAL",
            consumer="kernel-action-proposals",
            num_pending=12,
            num_ack_pending=3,
            filter_subject="proposal.new",
        )
        with caplog.at_level(logging.WARNING):
            payload = emit_consumer_lag_alert(
                snapshot,
                threshold=10,
                monitor="kernel",
            )

        assert payload["event"] == CONSUMER_LAG_EVENT
        assert payload["lag"] == 12
        assert len(caplog.records) == 1
        assert json.loads(caplog.records[0].message) == payload

    def test_lag_threshold_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JETSTREAM_LAG_THRESHOLD", "25")
        assert lag_threshold() == 25


class TestEventBusConsumerLag:
    @pytest.mark.asyncio
    async def test_check_consumer_lags_emits_alert_over_threshold(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JETSTREAM_LAG_THRESHOLD", "5")

        bus = EventBus(name="kernel")
        bus._js = MagicMock()
        bus._nc = MagicMock()
        bus._nc.is_connected = True
        bus._connected.set()

        consumer = SimpleNamespace(
            name="kernel-action-proposals",
            num_pending=8,
            num_ack_pending=1,
            config=SimpleNamespace(filter_subject="proposal.new"),
        )
        bus._js.consumers_info = AsyncMock(return_value=[consumer])

        with caplog.at_level(logging.WARNING):
            alerts = await bus.check_consumer_lags(SAFETY_STREAM_NAME)

        assert len(alerts) == 1
        assert alerts[0]["consumer"] == "kernel-action-proposals"
        assert alerts[0]["lag"] == 8
        assert bus._lag_alerts_emitted["kernel-action-proposals"] == 1

    @pytest.mark.asyncio
    async def test_check_consumer_lags_skips_waiter_durables(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JETSTREAM_LAG_THRESHOLD", "1")

        bus = EventBus(name="kernel")
        bus._js = MagicMock()
        bus._nc = MagicMock()
        bus._nc.is_connected = True
        bus._connected.set()

        waiter = SimpleNamespace(
            name="waiter-action-trace-1",
            num_pending=99,
            num_ack_pending=0,
            config=SimpleNamespace(filter_subject="decision.trace-1"),
        )
        bus._js.consumers_info = AsyncMock(return_value=[waiter])

        alerts = await bus.check_consumer_lags(SAFETY_STREAM_NAME)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_check_consumer_lags_clears_alert_state_when_recovered(self) -> None:
        bus = EventBus(name="kernel")
        bus._js = MagicMock()
        bus._nc = MagicMock()
        bus._nc.is_connected = True
        bus._connected.set()
        bus._lag_alerts_emitted["kernel-code-proposals"] = 2

        healthy = SimpleNamespace(
            name="kernel-code-proposals",
            num_pending=0,
            num_ack_pending=0,
            config=SimpleNamespace(filter_subject="code.proposal"),
        )
        bus._js.consumers_info = AsyncMock(return_value=[healthy])

        alerts = await bus.check_consumer_lags(SAFETY_STREAM_NAME)
        assert alerts == []
        assert "kernel-code-proposals" not in bus._lag_alerts_emitted

    @pytest.mark.asyncio
    async def test_run_lag_monitor_polls_periodically(self) -> None:
        bus = EventBus(name="kernel")
        bus.check_consumer_lags = AsyncMock(return_value=[])  # type: ignore[method-assign]

        task = asyncio.create_task(bus.run_lag_monitor(interval_s=0.05))
        await asyncio.sleep(0.12)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert bus.check_consumer_lags.await_count >= 2
