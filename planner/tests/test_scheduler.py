"""Tests for the Planner's priority Scheduler.

Loads scheduler.py directly so the test doesn't import the planner package
(which pulls in the activelearning SDK, nats, and aiohttp via service.py). The
scheduler logic is pure stdlib (asyncio), so it can be exercised in isolation.
"""

import asyncio
import importlib.util
import os
import sys

_SCHED_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "planner", "scheduler.py")
_spec = importlib.util.spec_from_file_location("planner_scheduler", _SCHED_PATH)
scheduler = importlib.util.module_from_spec(_spec)
sys.modules["planner_scheduler"] = scheduler
_spec.loader.exec_module(scheduler)

Scheduler = scheduler.Scheduler
SchedulerMode = scheduler.SchedulerMode
PendingAction = scheduler.PendingAction


def _action(trace_id, priority, proposal=None, expires_at=None):
    return PendingAction(
        trace_id=trace_id,
        priority=priority,
        proposal=proposal or {},
        expires_at=expires_at,
    )


def run(coro):
    return asyncio.run(coro)


# ── initial state ───────────────────────────────────────────────────────────


def test_starts_in_execution_mode_with_empty_queue():
    s = Scheduler()
    assert s.mode == SchedulerMode.EXECUTION
    assert s.pending_count == 0
    assert s.get_queue_status()["oldest_action"] is None


# ── priority ordering ─────────────────────────────────────────────────────────


def test_enqueue_orders_highest_priority_first():
    s = Scheduler()
    run(s.enqueue(_action("low", 1)))
    run(s.enqueue(_action("high", 10)))
    run(s.enqueue(_action("mid", 5)))
    assert s.pending_count == 3
    first = run(s.dequeue())
    second = run(s.dequeue())
    third = run(s.dequeue())
    assert [first.trace_id, second.trace_id, third.trace_id] == ["high", "mid", "low"]


def test_equal_priority_preserves_insertion_order():
    s = Scheduler()
    run(s.enqueue(_action("first", 5)))
    run(s.enqueue(_action("second", 5)))
    assert run(s.dequeue()).trace_id == "first"
    assert run(s.dequeue()).trace_id == "second"


def test_get_queue_status_reports_front_action():
    s = Scheduler()
    run(s.enqueue(_action("a", 1)))
    run(s.enqueue(_action("b", 9)))
    status = s.get_queue_status()
    assert status["mode"] == "EXECUTION"
    assert status["pending_count"] == 2
    assert status["oldest_action"] == "b"


# ── dequeue edge cases ────────────────────────────────────────────────────────


def test_dequeue_empty_returns_none():
    s = Scheduler()
    assert run(s.dequeue()) is None


def test_dequeue_drops_expired_actions():
    s = Scheduler()
    run(s.enqueue(_action("expired", 5, expires_at=1)))  # epoch-ms 1 = long past
    run(s.enqueue(_action("live", 1, expires_at=None)))
    result = run(s.dequeue())
    assert result.trace_id == "live"
    assert s.pending_count == 0


# ── SAFE_HALT behavior ────────────────────────────────────────────────────────


def test_safe_halt_clears_pending_and_rejects_enqueue():
    s = Scheduler()
    run(s.enqueue(_action("a", 1)))
    run(s.enqueue(_action("b", 2)))
    run(s.set_mode(SchedulerMode.SAFE_HALT))
    assert s.pending_count == 0
    assert run(s.enqueue(_action("c", 3))) is False
    assert run(s.dequeue()) is None


# ── mode-based priority adjustment ────────────────────────────────────────────


def test_learning_mode_boosts_learning_tagged_actions():
    s = Scheduler()
    run(s.set_mode(SchedulerMode.LEARNING))
    run(s.enqueue(_action("learn", 1, proposal={"tags": ["learning"]})))
    run(s.enqueue(_action("plain", 50, proposal={"tags": []})))
    # learn gets +100 -> 101, beating plain's 50
    assert run(s.dequeue()).trace_id == "learn"


def test_exploration_mode_boosts_exploration_tagged_actions():
    s = Scheduler()
    run(s.set_mode(SchedulerMode.EXPLORATION))
    run(s.enqueue(_action("explore", 1, proposal={"tags": ["exploration"]})))
    run(s.enqueue(_action("plain", 50, proposal={"tags": []})))
    assert run(s.dequeue()).trace_id == "explore"


def test_untagged_action_is_not_boosted_in_learning_mode():
    s = Scheduler()
    run(s.set_mode(SchedulerMode.LEARNING))
    a = _action("plain", 7, proposal={"tags": ["exploration"]})
    run(s.enqueue(a))
    # learning mode only boosts "learning" tags, so priority stays 7
    assert a.priority == 7


# ── housekeeping helpers ──────────────────────────────────────────────────────


def test_clear_pending_returns_count_and_empties():
    s = Scheduler()
    run(s.enqueue(_action("a", 1)))
    run(s.enqueue(_action("b", 2)))
    assert run(s.clear_pending()) == 2
    assert s.pending_count == 0


def test_mode_change_callback_receives_old_and_new_mode():
    s = Scheduler()
    seen = []

    async def record(old, new):
        seen.append((old, new))

    s.on_mode_change(record)
    run(s.set_mode(SchedulerMode.LEARNING))
    assert seen == [(SchedulerMode.EXECUTION, SchedulerMode.LEARNING)]


def test_failing_callback_does_not_break_mode_change():
    s = Scheduler()

    async def boom(_old, _new):
        raise RuntimeError("callback failure")

    s.on_mode_change(boom)
    run(s.set_mode(SchedulerMode.LEARNING))
    assert s.mode == SchedulerMode.LEARNING
