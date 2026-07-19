"""In-process publish/subscribe/request counters and latency stats for EventBus."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyStats:
    """Running latency aggregate (count, mean, max)."""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, latency_ms: float) -> None:
        self.count += 1
        self.total_ms += latency_ms
        if latency_ms > self.max_ms:
            self.max_ms = latency_ms

    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms(), 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass
class EventBusMetrics:
    """Mutable counters updated by EventBus publish/subscribe/request paths."""

    publish: LatencyStats = field(default_factory=LatencyStats)
    subscribe_delivery: LatencyStats = field(default_factory=LatencyStats)
    request: LatencyStats = field(default_factory=LatencyStats)
    publish_jetstream_count: int = 0
    resubscribe_failed_count: int = 0

    def record_publish(self, latency_ms: float, *, jetstream: bool = False) -> None:
        self.publish.record(latency_ms)
        if jetstream:
            self.publish_jetstream_count += 1

    def record_subscribe_delivery(self, latency_ms: float) -> None:
        self.subscribe_delivery.record(latency_ms)

    def record_request(self, latency_ms: float) -> None:
        self.request.record(latency_ms)

    def record_resubscribe_failed(self) -> None:
        self.resubscribe_failed_count += 1

    def snapshot(self, service_name: str) -> dict[str, Any]:
        """JSON-serializable payload for eventbus.metrics.{service_name}."""
        return {
            "service": service_name,
            "timestamp_ms": int(time.time() * 1000),
            "publish": {
                **self.publish.to_dict(),
                "jetstream_count": self.publish_jetstream_count,
            },
            "subscribe": self.subscribe_delivery.to_dict(),
            "request": self.request.to_dict(),
            "resubscribe_failed_count": self.resubscribe_failed_count,
        }
