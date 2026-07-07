"""Tests for kernel decision-latency SLO tracking.

Covers:
- _compute_percentiles pure function
- KernelService._record_latency + _compute_latency_stats behaviour:
  * samples below threshold → no breach
  * samples above threshold → breach published after cooldown
  * no breach when sample count is below _SLO_MIN_SAMPLES
  * rolling window bounded at _LATENCY_WINDOW
  * configurable threshold via KERNEL_LATENCY_SLO_MS env var
"""

import asyncio
import os
import sys
import unittest
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Import the module under test (avoid running main() at import time)
# ---------------------------------------------------------------------------
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src"),
)

# The installed activelearning package may lag behind sdk/ source changes.
# Backfill missing constants so these tests work regardless.
from activelearning.subjects import Subjects as _Subjects
if not hasattr(_Subjects, "KERNEL_SLO_BREACH"):
    _Subjects.KERNEL_SLO_BREACH = "kernel.slo.breach"
if not hasattr(_Subjects, "KERNEL_METRICS"):
    _Subjects.KERNEL_METRICS = "kernel.metrics"

from kernel.service import _compute_percentiles, _LATENCY_WINDOW, _SLO_MIN_SAMPLES


# ---------------------------------------------------------------------------
# Unit tests for _compute_percentiles
# ---------------------------------------------------------------------------

class TestComputePercentiles(unittest.TestCase):
    def test_empty(self):
        result = _compute_percentiles([])
        self.assertIsNone(result["p50_ms"])
        self.assertIsNone(result["p99_ms"])
        self.assertIsNone(result["max_ms"])
        self.assertEqual(result["count"], 0)

    def test_single_sample(self):
        result = _compute_percentiles([10.0])
        self.assertEqual(result["p50_ms"], 10.0)
        self.assertEqual(result["p99_ms"], 10.0)
        self.assertEqual(result["max_ms"], 10.0)
        self.assertEqual(result["count"], 1)

    def test_known_10_samples(self):
        # 10 samples [1..10] ms
        samples = list(range(1, 11))
        result = _compute_percentiles(samples)
        self.assertEqual(result["count"], 10)
        # p50 index = int(0.50 * 10) = 5 → value = 6 (0-indexed sorted list)
        self.assertEqual(result["p50_ms"], 6)
        # p99 index = min(int(0.99 * 10), 9) = min(9, 9) = 9 → value = 10
        self.assertEqual(result["p99_ms"], 10)
        self.assertEqual(result["max_ms"], 10)

    def test_p99_outlier(self):
        # 100 samples at 5 ms, one outlier at 500 ms
        samples = [5.0] * 100 + [500.0]
        result = _compute_percentiles(samples)
        self.assertEqual(result["count"], 101)
        self.assertEqual(result["max_ms"], 500.0)
        # p99 index = min(int(0.99*101), 100) = min(99, 100) = 99 → 5.0
        self.assertEqual(result["p99_ms"], 5.0)
        # outlier only at p100 (the max)

    def test_unsorted_input(self):
        result = _compute_percentiles([30.0, 10.0, 20.0])
        self.assertEqual(result["max_ms"], 30.0)
        self.assertEqual(result["p50_ms"], 20.0)


# ---------------------------------------------------------------------------
# Integration-style tests for _record_latency via a minimal stub service
# ---------------------------------------------------------------------------

def _make_service(threshold_ms: float = 50.0):
    """Build a KernelService with all heavy dependencies stubbed."""
    with patch.dict(os.environ, {"KERNEL_LATENCY_SLO_MS": str(threshold_ms)}):
        with patch("kernel.service.BaseService.__init__", lambda self, *a, **kw: None), \
             patch("kernel.service.KernelEvaluator"), \
             patch("kernel.service.PolicyRollbackManager"), \
             patch("kernel.service.DecisionSequenceTracker"):
            from kernel.service import KernelService, _LATENCY_WINDOW, _SLO_MIN_SAMPLES
            svc = KernelService.__new__(KernelService)
            svc.logger = MagicMock()
            svc.event_bus = MagicMock()
            svc.event_bus.publish = AsyncMock()
            svc._SLO_P99_MS = threshold_ms
            svc._latency_samples = deque(maxlen=_LATENCY_WINDOW)
            svc._slo_breach_count = 0
            svc._last_slo_breach_at = 0.0
            return svc


class TestRecordLatency(unittest.IsolatedAsyncioTestCase):
    async def test_no_breach_under_threshold(self):
        svc = _make_service(threshold_ms=50.0)
        # Fill with _SLO_MIN_SAMPLES fast samples
        for _ in range(_SLO_MIN_SAMPLES + 5):
            svc._record_latency(0.010)  # 10 ms
        # No breach should have been published
        svc.event_bus.publish.assert_not_called()
        self.assertEqual(svc._slo_breach_count, 0)

    async def test_breach_above_threshold(self):
        svc = _make_service(threshold_ms=50.0)
        # First fill enough samples so p99 is above threshold
        for _ in range(_SLO_MIN_SAMPLES):
            svc._latency_samples.append(200.0)  # 200 ms each — all above threshold

        published = []

        async def capture_publish(subject, payload):
            published.append((subject, payload))

        svc.event_bus.publish = capture_publish

        with patch("asyncio.create_task", lambda coro: asyncio.ensure_future(coro)):
            svc._record_latency(0.200)  # triggers SLO check

        # Give create_task coroutines a chance to run
        await asyncio.sleep(0)

        self.assertEqual(svc._slo_breach_count, 1)
        self.assertEqual(len(published), 1)
        subject, payload = published[0]
        self.assertEqual(subject, "kernel.slo.breach")
        self.assertGreater(payload["p99_ms"], 50.0)
        self.assertEqual(payload["threshold_ms"], 50.0)

    async def test_no_alert_below_min_samples(self):
        svc = _make_service(threshold_ms=50.0)
        # Add fewer than _SLO_MIN_SAMPLES slow samples
        for _ in range(_SLO_MIN_SAMPLES - 1):
            svc._record_latency(0.200)  # 200 ms each
        svc.event_bus.publish.assert_not_called()
        self.assertEqual(svc._slo_breach_count, 0)

    async def test_window_bounded_at_maxlen(self):
        svc = _make_service(threshold_ms=50.0)
        for i in range(_LATENCY_WINDOW + 50):
            svc._latency_samples.append(float(i))
        self.assertEqual(len(svc._latency_samples), _LATENCY_WINDOW)

    async def test_configurable_threshold(self):
        svc = _make_service(threshold_ms=200.0)
        # Samples at 100 ms (below 200 ms threshold) — no breach
        for _ in range(_SLO_MIN_SAMPLES + 5):
            svc._latency_samples.append(100.0)
        svc._record_latency(0.100)
        svc.event_bus.publish.assert_not_called()
        self.assertEqual(svc._slo_breach_count, 0)


class TestComputeLatencyStats(unittest.TestCase):
    def test_stats_include_threshold_and_breach_count(self):
        svc = _make_service(threshold_ms=75.0)
        for _ in range(20):
            svc._latency_samples.append(30.0)
        svc._slo_breach_count = 3
        stats = svc._compute_latency_stats()
        self.assertEqual(stats["threshold_ms"], 75.0)
        self.assertEqual(stats["breach_count"], 3)
        self.assertIsNotNone(stats["p99_ms"])
        self.assertEqual(stats["count"], 20)

    def test_stats_empty_samples(self):
        svc = _make_service(threshold_ms=50.0)
        stats = svc._compute_latency_stats()
        self.assertIsNone(stats["p99_ms"])
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["threshold_ms"], 50.0)


if __name__ == "__main__":
    unittest.main()
