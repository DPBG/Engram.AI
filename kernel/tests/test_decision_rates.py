"""Tests for Kernel decision-rate governance signal (issue #143 / M6).

Longitudinal ALLOW/TRANSFORM/DENY/DEFER rates are computed from the
kernel_decisions audit trail and surfaced via kernel.status and the
periodic kernel.decision_rates publish. This is a read-only visibility
signal — it must never feed back into evaluation, so these tests only
cover aggregation and publish behavior, not gating.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from activelearning.subjects import Subjects


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _row(decision_type: str, source: str, cnt: int) -> dict:
    # aiosqlite.Row supports item access by column name; a plain dict
    # satisfies the same `row["col"]` interface used in service.py.
    return {"decision_type": decision_type, "source": source, "cnt": cnt}


def _make_kernel_service(all_time_rows=None, window_rows=None):
    """Return a KernelService wired with a mock database and event bus."""
    from kernel.service import KernelService

    svc = KernelService.__new__(KernelService)
    svc.logger = MagicMock()
    svc._decision_rates_interval_sec = 300.0
    svc._decision_rates_window_hours = 24.0

    responses = [all_time_rows or [], window_rows or []]

    async def _fetchall(sql, params=()):
        return responses.pop(0)

    svc.database = AsyncMock()
    svc.database.fetchall.side_effect = _fetchall

    svc.event_bus = AsyncMock()
    return svc


def test_compute_decision_rates_empty_when_no_decisions():
    svc = _make_kernel_service(all_time_rows=[], window_rows=[])
    result = _run(svc._compute_decision_rates())

    assert result["all_time"]["total"] == 0
    assert result["all_time"]["rates"] == {"ALLOW": 0.0, "TRANSFORM": 0.0, "DENY": 0.0, "DEFER": 0.0}
    assert result["window"]["total"] == 0
    assert result["by_source"] == {}


def test_compute_decision_rates_aggregates_counts_and_rates():
    all_time = [
        _row("ALLOW", "meta-programmer", 6),
        _row("DENY", "meta-programmer", 2),
        _row("DEFER", "meta-programmer", 2),
        _row("ALLOW", "neuromorphic", 10),
    ]
    window = [
        _row("ALLOW", "meta-programmer", 1),
        _row("DENY", "meta-programmer", 1),
    ]
    svc = _make_kernel_service(all_time_rows=all_time, window_rows=window)
    result = _run(svc._compute_decision_rates())

    # All-time totals are summed across all sources.
    assert result["all_time"]["total"] == 20
    assert result["all_time"]["counts"]["ALLOW"] == 16
    assert result["all_time"]["counts"]["DENY"] == 2
    assert result["all_time"]["rates"]["ALLOW"] == 0.8

    # Windowed bucket only reflects the (smaller) recent rows.
    assert result["window"]["total"] == 2
    assert result["window"]["rates"]["ALLOW"] == 0.5
    assert result["window"]["rates"]["DENY"] == 0.5

    # Per-source breakdown isolates meta-programmer's own quality signal.
    mp = result["by_source"]["meta-programmer"]
    assert mp["all_time"]["total"] == 10
    assert mp["all_time"]["counts"]["ALLOW"] == 6
    assert mp["all_time"]["rates"]["DENY"] == 0.2
    assert mp["window"]["total"] == 2

    neuro = result["by_source"]["neuromorphic"]
    assert neuro["all_time"]["total"] == 10
    assert neuro["window"]["total"] == 0


def test_compute_decision_rates_falls_back_to_unknown_source():
    all_time = [_row("ALLOW", None, 3)]
    svc = _make_kernel_service(all_time_rows=all_time, window_rows=[])
    result = _run(svc._compute_decision_rates())

    assert "unknown" in result["by_source"]
    assert result["by_source"]["unknown"]["all_time"]["counts"]["ALLOW"] == 3


def test_compute_decision_rates_is_best_effort_on_db_error():
    from kernel.service import KernelService

    svc = KernelService.__new__(KernelService)
    svc.logger = MagicMock()
    svc._decision_rates_interval_sec = 300.0
    svc._decision_rates_window_hours = 24.0
    svc.database = AsyncMock()
    svc.database.fetchall.side_effect = RuntimeError("db unavailable")

    result = _run(svc._compute_decision_rates())

    assert result["all_time"]["total"] == 0
    assert result["by_source"] == {}


def test_decision_rates_loop_publishes_to_kernel_decision_rates_subject():
    svc = _make_kernel_service(all_time_rows=[_row("ALLOW", "meta-programmer", 1)], window_rows=[])

    async def _tick():
        # Run one iteration of the loop body directly rather than the
        # infinite `while True` wrapper, so the test doesn't need to
        # race a background asyncio.sleep().
        rates = await svc._compute_decision_rates()
        await svc.event_bus.publish(Subjects.KERNEL_DECISION_RATES, rates)

    _run(_tick())

    svc.event_bus.publish.assert_awaited_once()
    subject, payload = svc.event_bus.publish.call_args.args
    assert subject == "kernel.decision_rates"
    assert payload["all_time"]["counts"]["ALLOW"] == 1
