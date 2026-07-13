"""Tests for EventBus in-process metrics."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from activelearning.bus_metrics import EventBusMetrics, LatencyStats
from activelearning.nats_client import EventBus
from activelearning.subjects import eventbus_metrics_subject


class TestLatencyStats:
    def test_avg_and_max(self):
        stats = LatencyStats()
        stats.record(10.0)
        stats.record(20.0)
        d = stats.to_dict()
        assert d["count"] == 2
        assert d["avg_ms"] == 15.0
        assert d["max_ms"] == 20.0


class TestEventBusMetrics:
    def test_snapshot_shape(self):
        metrics = EventBusMetrics()
        metrics.record_publish(1.5, jetstream=True)
        metrics.record_subscribe_delivery(2.0)
        metrics.record_request(3.0)
        snap = metrics.snapshot("kernel")
        assert snap["service"] == "kernel"
        assert snap["publish"]["count"] == 1
        assert snap["publish"]["jetstream_count"] == 1
        assert snap["subscribe"]["count"] == 1
        assert snap["request"]["count"] == 1
        assert "timestamp_ms" in snap


class TestEventBusInstrumentation:
    @pytest.mark.asyncio
    async def test_publish_records_latency(self):
        bus = EventBus(name="test-pub")
        bus._nc = MagicMock()
        bus._nc.is_connected = True
        bus._nc.publish = AsyncMock()
        bus._connected.set()

        await bus.publish("observation.camera", {"frame": "x"})
        assert bus._metrics.publish.count == 1
        assert bus._metrics.publish.max_ms >= 0

    @pytest.mark.asyncio
    async def test_metrics_subject_not_counted(self):
        bus = EventBus(name="test-metrics")
        bus._nc = MagicMock()
        bus._nc.is_connected = True
        bus._nc.publish = AsyncMock()
        bus._connected.set()

        await bus.publish(eventbus_metrics_subject("test-metrics"), {"service": "x"})
        assert bus._metrics.publish.count == 0

    @pytest.mark.asyncio
    async def test_request_records_latency(self):
        bus = EventBus(name="test-req")
        bus._nc = MagicMock()
        bus._nc.is_connected = True
        bus._connected.set()

        response = MagicMock()
        response.data = b'{"ok": true}'
        bus._nc.request = AsyncMock(return_value=response)

        result = await bus.request("beliefs.query", {"q": "x"})
        assert result == {"ok": True}
        assert bus._metrics.request.count == 1

    @pytest.mark.asyncio
    async def test_subscribe_callback_records_delivery(self):
        bus = EventBus(name="test-sub")
        received: list[dict] = []

        async def handler(data: dict) -> None:
            received.append(data)

        bus._nc = MagicMock()
        bus._nc.is_connected = True
        bus._nc.flush = AsyncMock()
        bus._connected.set()

        captured_cb = None

        async def capture_subscribe(subject, cb, **kwargs):
            nonlocal captured_cb
            captured_cb = cb
            return MagicMock()

        bus._nc.subscribe = AsyncMock(side_effect=capture_subscribe)
        await bus.subscribe("observation.test", handler)

        msg = MagicMock()
        msg.data = b'{"value": 1}'
        msg.reply = ""
        assert captured_cb is not None
        await captured_cb(msg)

        assert received == [{"value": 1}]
        assert bus._metrics.subscribe_delivery.count == 1

    @pytest.mark.asyncio
    async def test_metrics_reporter_publishes_snapshot(self):
        bus = EventBus(name="reporter")
        bus._nc = MagicMock()
        bus._nc.is_connected = True
        bus._nc.publish = AsyncMock()
        bus._connected.set()
        bus._metrics_interval_s = 0.05

        await bus.start_metrics_reporter(interval_s=0.05)
        await asyncio.sleep(0.12)
        await bus.stop_metrics_reporter()

        assert bus._nc.publish.await_count >= 1
        subject = bus._nc.publish.await_args_list[0].args[0]
        assert subject == eventbus_metrics_subject("reporter")

    def test_get_metrics_returns_snapshot(self):
        bus = EventBus(name="snap")
        bus._metrics.record_publish(1.0)
        snap = bus.get_metrics()
        assert snap["service"] == "snap"
        assert snap["publish"]["count"] == 1
