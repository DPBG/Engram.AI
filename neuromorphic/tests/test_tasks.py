"""Tests for MuJoCo task environments (tasks.py).

Covers all 5 tasks, TaskCurriculum, and the outcome conversion bridge.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from neuromorphic.mujoco_body import MuJoCoBody
from neuromorphic.tasks import (
    SupportedStandTask,
    StandTask,
    BalanceTask,
    ReachTask,
    HeadTrackTask,
    WalkTask,
    TaskCurriculum,
    TaskResult,
    task_result_to_outcome,
)


@pytest.fixture
def body():
    return MuJoCoBody(steps_per_command=50)


class TestTaskResult:
    def test_fields(self):
        r = TaskResult(reward=0.5, success=True, progress=0.7, info={"x": 1})
        assert r.reward == 0.5
        assert r.success is True
        assert r.progress == 0.7
        assert r.info == {"x": 1}


class TestStandTask:
    def test_reset_clears_state(self, body):
        task = StandTask(hold_steps=10, max_steps=50)
        task.reset(body)
        assert task._step_count == 0
        assert task._done is False
        assert task._steps_upright == 0

    def test_upright_gives_positive_reward(self, body):
        task = StandTask(hold_steps=10, max_steps=50)
        task.reset(body)
        result = task.evaluate(body)
        assert result.reward >= 0.0  # body starts standing

    def test_completes_after_hold_steps(self, body):
        task = StandTask(hold_steps=5, max_steps=50)
        task.reset(body)
        for _ in range(10):
            result = task.evaluate(body)
            if result.success:
                break
        assert result.success is True
        assert result.reward == 1.0
        assert task.is_complete

    def test_state_persistence(self, body):
        task = StandTask(hold_steps=10)
        task.reset(body)
        task.evaluate(body)
        state = task.get_state()
        assert state["name"] == "stand"
        assert state["step_count"] == 1
        task2 = StandTask(hold_steps=10)
        task2.set_state(state)
        assert task2._step_count == 1


class TestBalanceTask:
    def test_reset_applies_force(self, body):
        task = BalanceTask(max_steps=100)
        task.reset(body)
        assert task._step_count == 0

    def test_evaluate_returns_result(self, body):
        task = BalanceTask(max_steps=100)
        task.reset(body)
        result = task.evaluate(body)
        assert isinstance(result, TaskResult)
        assert -1.0 <= result.reward <= 1.0
        assert 0.0 <= result.progress <= 1.0


class TestReachTask:
    def test_reset_sets_target(self, body):
        task = ReachTask(max_steps=100)
        task.reset(body)
        assert np.linalg.norm(task._target) > 0
        assert task._initial_dist > 0

    def test_evaluate_returns_result(self, body):
        task = ReachTask(max_steps=100)
        task.reset(body)
        result = task.evaluate(body)
        assert isinstance(result, TaskResult)
        assert "distance" in result.info

    def test_state_roundtrip(self, body):
        task = ReachTask(max_steps=100)
        task.reset(body)
        state = task.get_state()
        task2 = ReachTask(max_steps=100)
        task2.set_state(state)
        np.testing.assert_allclose(task2._target, task._target)


class TestHeadTrackTask:
    def test_reset_picks_target(self, body):
        task = HeadTrackTask(max_steps=100)
        task.reset(body)
        assert task._initial_error > 0

    def test_evaluate_returns_result(self, body):
        task = HeadTrackTask(max_steps=100)
        task.reset(body)
        result = task.evaluate(body)
        assert isinstance(result, TaskResult)
        assert "errors" in result.info

    def test_max_steps_stops(self, body):
        task = HeadTrackTask(max_steps=5)
        task.reset(body)
        for _ in range(10):
            task.evaluate(body)
        assert task.is_complete


class TestWalkTask:
    def test_reset(self, body):
        task = WalkTask(target_distance=0.5, max_steps=100)
        task.reset(body)
        assert task._start_x is not None
        assert task._step_count == 0

    def test_evaluate_returns_result(self, body):
        task = WalkTask(target_distance=0.5, max_steps=100)
        task.reset(body)
        result = task.evaluate(body)
        assert isinstance(result, TaskResult)
        assert "distance" in result.info

    def test_state_persistence(self, body):
        task = WalkTask(target_distance=0.5, max_steps=100)
        task.reset(body)
        task.evaluate(body)
        state = task.get_state()
        assert state["name"] == "walk"
        task2 = WalkTask(target_distance=0.5, max_steps=100)
        task2.set_state(state)
        assert task2._step_count == 1


class TestTaskCurriculum:
    def test_starts_with_supported_stand(self, body):
        cur = TaskCurriculum(body)
        assert cur.current_task.name == "supported_stand"

    def test_step_returns_result(self, body):
        cur = TaskCurriculum(body)
        result = cur.step()
        assert isinstance(result, TaskResult)

    def test_advance_after_successes(self, body):
        cur = TaskCurriculum(body)
        # Skip to StandTask (index 1) and force it to succeed quickly
        cur._current_idx = 1
        cur._tasks[1].reset(body)
        cur.current_task._hold_steps = 1
        # Step until we get 3 successes
        for _ in range(100):
            cur.step()
            if cur.current_task.name != "stand":
                break
        assert cur._total_advances >= 1

    def test_state_roundtrip(self, body):
        cur = TaskCurriculum(body)
        cur.step()
        state = cur.get_state()
        cur2 = TaskCurriculum(body)
        cur2.set_state(state)
        assert cur2._current_idx == cur._current_idx
        assert cur2._successes == cur._successes

    def test_cycles_through_tasks(self, body):
        """After walk, wraps back to first task."""
        cur = TaskCurriculum(body)
        # Force to last task
        cur._current_idx = len(cur._tasks) - 1  # walk (last)
        cur._successes = cur._advance_threshold  # meet threshold
        cur.advance()
        assert cur._current_idx == 0  # wrapped to first


class TestTaskResultToOutcome:
    def test_produces_motor_outcome(self, body):
        task = StandTask()
        task.reset(body)
        result = TaskResult(0.5, True, 0.8, {})
        outcome = task_result_to_outcome(result, task, body)
        assert outcome["channel"] == "locomotion"
        assert outcome["success"] is True
        assert outcome["confidence"] == 0.5
        assert len(outcome["proprioceptive_state"]) == body.proprioceptive_size
        assert "task_name" in outcome
        assert outcome["task_name"] == "stand"
