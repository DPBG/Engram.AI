"""Structured task environments for the MuJoCo virtual body.

Goal-directed tasks with measurable success criteria and reward signals
for the motor feedback loop.  TaskCurriculum manages progressive difficulty.
task_result_to_outcome() bridges TaskResult to motor.outcome dicts.

Tasks are body-agnostic: they use the body's dynamic _root_body_name and
_joint_channel instead of hardcoded body/joint names. Tasks that require
bodies or joints not present in the model are skipped by TaskCurriculum.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mujoco_body import MuJoCoBody

logger = logging.getLogger(__name__)


def _has_body(body: MuJoCoBody, name: str) -> bool:
    """Check if a named body exists in the MuJoCo model."""
    try:
        body._model.body(name)
        return True
    except Exception:
        return False


def _has_joint(body: MuJoCoBody, name: str) -> bool:
    """Check if a named joint exists in the MuJoCo model."""
    try:
        body._model.joint(name)
        return True
    except Exception:
        return False


@dataclass
class TaskResult:
    """Outcome of a single task evaluation step."""

    reward: float  # -1 to 1
    success: bool  # task completed?
    progress: float  # 0-1 toward goal
    info: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Base class for MuJoCo body tasks."""

    def __init__(self, name: str, channel: str, max_steps: int) -> None:
        self.name = name
        self.channel = channel  # "locomotion" | "manipulation" | "head"
        self.max_steps = max_steps
        self._step_count: int = 0
        self._done: bool = False

    @abstractmethod
    def reset(self, body: MuJoCoBody) -> None: ...

    @abstractmethod
    def evaluate(self, body: MuJoCoBody) -> TaskResult: ...

    def is_compatible(self, body: MuJoCoBody) -> bool:
        """Check if this task can run on the given body."""
        return self.channel in body._channel_actuators

    @property
    def is_complete(self) -> bool:
        return self._done or self._step_count >= self.max_steps

    def get_state(self) -> dict[str, Any]:
        return {"name": self.name, "step_count": self._step_count, "done": self._done}

    def set_state(self, state: dict[str, Any]) -> None:
        self._step_count = state.get("step_count", 0)
        self._done = state.get("done", False)


class SupportedStandTask(Task):
    """Learn to bear weight with pelvis constrained (like pulling up on furniture).

    The pelvis is held at initial height via a spring-damper (PD) controller on
    the root body.  The controller exactly compensates gravity plus applies a
    corrective spring force to keep the pelvis at the target height.  Support
    strength decays linearly over 80% of the task, leaving 20% unsupported.
    """

    def __init__(self, max_steps: int = 3000) -> None:
        super().__init__("supported_stand", "locomotion", max_steps)
        self._initial_z: float = 1.2
        self._support_active: bool = False
        self._gravity_comp: float = 600.0  # exact gravity compensation (mg)
        self._decay: float = 1.0  # current support decay factor

    def reset(self, body: MuJoCoBody) -> None:
        body.reset()
        self._step_count = 0
        self._done = False
        self._initial_z = body._initial_root_z
        total_mass = float(body._model.body_subtreemass[0])  # world subtree = all
        self._gravity_comp = total_mass * 9.81  # exact gravity compensation
        self._decay = 1.0
        # Apply initial support force (exact gravity compensation)
        root_id = body._model.body(body._root_body_name).id
        body._data.xfrc_applied[root_id, 2] = self._gravity_comp
        self._support_active = True

    def _compute_support_force(self, body: MuJoCoBody) -> float:
        """PD controller: gravity compensation + spring to hold target height."""
        root_z = float(body._data.body(body._root_body_name).xpos[2])
        root_vz = float(body._data.body(body._root_body_name).cvel[5])
        error = self._initial_z - root_z
        # Spring-damper gains scaled by gravity compensation
        kp = self._gravity_comp * 5.0  # stiff spring to target height
        kd = self._gravity_comp * 1.0  # damping to prevent oscillation
        force = self._gravity_comp + kp * error - kd * root_vz
        return max(0.0, force)  # never push downward

    def evaluate(self, body: MuJoCoBody) -> TaskResult:
        self._step_count += 1
        root_z = float(body._data.body(body._root_body_name).xpos[2])
        height_ratio = root_z / self._initial_z if self._initial_z > 0 else 0.0

        # Auto-reset on fall: if orientation-based fall detected, reset the
        # body immediately.  Lying on the ground for minutes teaches nothing
        # -- a quick reset gives the brain more learning opportunities.
        if body.is_fallen():
            logger.info(
                "Fall detected at step %d (height=%.2f), resetting episode",
                self._step_count,
                root_z,
            )
            self._done = True
            return TaskResult(
                -0.5,
                False,
                1.0,
                {"root_z": root_z, "support_decay": self._decay, "fall_reset": True},
            )

        # Continuous reward: how upright is the body? (0.0-1.0 proportional)
        reward = float(np.clip(height_ratio, 0.0, 1.0)) - 0.5  # centered: -0.5 to +0.5

        # 30% full support, 50% linear decay, 20% unsupported evaluation.
        # The unsupported window gives the brain ~600 steps (at 3000 max)
        # where success can accumulate toward the advance threshold.
        support_hold = int(self.max_steps * 0.3)  # full support ends
        support_end = int(self.max_steps * 0.8)  # decay ends, unsupported begins
        if self._support_active:
            if self._step_count <= support_hold:
                self._decay = 1.0  # full support
            elif self._step_count <= support_end:
                decay_progress = (self._step_count - support_hold) / max(
                    1, support_end - support_hold
                )
                self._decay = max(0.0, 1.0 - decay_progress)
            else:
                self._decay = 0.0
            root_id = body._model.body(body._root_body_name).id
            if self._decay > 0.0:
                force = self._compute_support_force(body) * self._decay
                body._data.xfrc_applied[root_id, 2] = force
            else:
                self._support_active = False
                body._data.xfrc_applied[root_id, 2] = 0.0

        progress = self._step_count / self.max_steps
        # Success: standing upright during the unsupported window
        success = height_ratio > 0.7 and not self._support_active
        if self._step_count >= self.max_steps:
            self._done = True
        return TaskResult(
            reward, success, progress, {"root_z": root_z, "support_decay": self._decay}
        )

    def get_state(self) -> dict[str, Any]:
        s = super().get_state()
        s.update(
            initial_z=self._initial_z,
            support_active=self._support_active,
            gravity_comp=self._gravity_comp,
            decay=self._decay,
        )
        return s

    def apply_continuous_support(self, body: MuJoCoBody) -> None:
        """Apply PD support force -- call at physics rate (50 Hz) for smooth support.

        Unlike evaluate() (1 Hz), this only updates the force without advancing
        the decay schedule or step count.  Also applies gentle horizontal
        centering to prevent long-term drift from pushes/motor noise.
        """
        root_id = body._model.body(body._root_body_name).id

        # Gentle horizontal centering (prevents drift, doesn't fight pushes).
        # kp_h ~ 0.3*mg gives ~180 N/m spring.  At 0.1 m offset = 18 N
        # (much weaker than a 15 N push impulse over 150 steps).
        root_pos = body._data.body(body._root_body_name).xpos
        root_vel = body._data.body(body._root_body_name).cvel
        kp_h = self._gravity_comp * 0.3
        kd_h = self._gravity_comp * 0.2  # damping to avoid oscillation
        body._data.xfrc_applied[root_id, 0] = -kp_h * float(root_pos[0]) - kd_h * float(root_vel[3])
        body._data.xfrc_applied[root_id, 1] = -kp_h * float(root_pos[1]) - kd_h * float(root_vel[4])

        if not self._support_active or self._decay <= 0:
            return
        force = self._compute_support_force(body) * self._decay
        body._data.xfrc_applied[root_id, 2] = force

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._initial_z = state.get("initial_z", 1.2)
        self._support_active = state.get("support_active", False)
        # backward compat: old states used "support_force" at 1.5x
        self._gravity_comp = state.get("gravity_comp", state.get("support_force", 600.0) / 1.5)
        self._decay = state.get("decay", 1.0)


class StandTask(Task):
    """Maintain upright posture for hold_steps consecutive evaluations.

    Uses continuous reward shaping: reward proportional to height ratio,
    not binary +0.1/-0.1. This gives the brain a gradient to learn from.
    """

    def __init__(self, hold_steps: int = 150, max_steps: int = 500) -> None:
        super().__init__("stand", "locomotion", max_steps)
        self._hold_steps = hold_steps
        self._steps_upright: int = 0
        self._initial_z: float = 1.2

    def reset(self, body: MuJoCoBody) -> None:
        body.reset()
        self._step_count = 0
        self._done = False
        self._steps_upright = 0
        self._initial_z = body._initial_root_z

    def evaluate(self, body: MuJoCoBody) -> TaskResult:
        self._step_count += 1
        root_z = float(body._data.body(body._root_body_name).xpos[2])
        threshold = 0.8 * self._initial_z
        height_ratio = root_z / self._initial_z if self._initial_z > 0 else 0.0
        if body.is_fallen():
            self._done = True
            return TaskResult(
                -1.0,
                False,
                self._steps_upright / self._hold_steps,
                {"root_z": root_z, "reason": "fallen"},
            )
        if root_z >= threshold:
            self._steps_upright += 1
        else:
            self._steps_upright = max(0, self._steps_upright - 1)
        progress = min(1.0, self._steps_upright / self._hold_steps)
        if self._steps_upright >= self._hold_steps:
            self._done = True
            return TaskResult(1.0, True, 1.0, {"root_z": root_z})
        # Continuous reward: proportional to height (not binary +/-0.1)
        reward = float(np.clip(height_ratio - 0.5, -0.5, 0.5))
        return TaskResult(
            reward, False, progress, {"root_z": root_z, "steps_upright": self._steps_upright}
        )

    def get_state(self) -> dict[str, Any]:
        s = super().get_state()
        s.update(
            steps_upright=self._steps_upright,
            initial_z=self._initial_z,
            hold_steps=self._hold_steps,
        )
        return s

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._steps_upright = state.get("steps_upright", 0)
        self._initial_z = state.get("initial_z", 1.2)
        self._hold_steps = state.get("hold_steps", self._hold_steps)


class BalanceTask(Task):
    """Recover upright posture after random perturbation force (10-50 N)."""

    def __init__(self, max_steps: int = 200) -> None:
        super().__init__("balance", "locomotion", max_steps)
        self._target_z: float = 1.2
        self._initial_z: float = 1.2

    def reset(self, body: MuJoCoBody) -> None:
        body.reset()
        self._step_count = 0
        self._done = False
        self._initial_z = body._initial_root_z
        self._target_z = 0.7 * self._initial_z
        # Apply random horizontal force to the root body
        rng = np.random.default_rng()
        mag = rng.uniform(10.0, 50.0)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        root_id = body._model.body(body._root_body_name).id
        body._data.xfrc_applied[root_id, :3] = [mag * np.cos(ang), mag * np.sin(ang), 0.0]
        try:
            body.step_batch(25)
        finally:
            body._data.xfrc_applied[root_id, :3] = 0.0

    def evaluate(self, body: MuJoCoBody) -> TaskResult:
        self._step_count += 1
        root_z = float(body._data.body(body._root_body_name).xpos[2])
        if body.is_fallen():
            self._done = True
            return TaskResult(
                -1.0,
                False,
                max(0.0, root_z / self._target_z),
                {"root_z": root_z, "reason": "fallen"},
            )
        height_ratio = root_z / self._initial_z
        progress = min(1.0, root_z / self._target_z)
        if root_z >= self._target_z:
            self._done = True
            return TaskResult(1.0, True, 1.0, {"root_z": root_z, "recovered": True})
        return TaskResult(
            float(np.clip(height_ratio - 0.5, -0.5, 0.5)),
            False,
            progress,
            {"root_z": root_z, "height_ratio": height_ratio},
        )

    def get_state(self) -> dict[str, Any]:
        s = super().get_state()
        s.update(target_z=self._target_z, initial_z=self._initial_z)
        return s

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._target_z = state.get("target_z", 1.2 * 0.7)
        self._initial_z = state.get("initial_z", 1.2)


class ReachTask(Task):
    """Move an end-effector to a random target within reach sphere.

    Auto-discovers the end-effector by finding the deepest body in
    the manipulation channel's kinematic chain.
    """

    def __init__(
        self, reach_radius: float = 0.3, tolerance: float = 0.1, max_steps: int = 300
    ) -> None:
        super().__init__("reach", "manipulation", max_steps)
        self._reach_radius = reach_radius
        self._tolerance = tolerance
        self._target: np.ndarray = np.zeros(3, dtype=np.float64)
        self._initial_dist: float = 1.0
        self._ee_body: str = ""  # end-effector body name (discovered at reset)
        self._anchor_body: str = ""  # shoulder/base body (discovered at reset)

    def is_compatible(self, body: MuJoCoBody) -> bool:
        return "manipulation" in body._channel_actuators

    def _discover_ee(self, body: MuJoCoBody) -> tuple[str, str]:
        """Find end-effector and anchor bodies for the manipulation chain."""
        manip_acts = body._channel_actuators.get("manipulation", [])
        if not manip_acts:
            return "", ""
        # Collect body IDs attached to manipulation actuators
        chain_bodies: list[int] = []
        for act_name in manip_acts:
            act_idx = body._actuator_idx.get(act_name)
            if act_idx is not None:
                jnt_id = body._model.actuator_trnid[act_idx, 0]
                bid = body._model.jnt_bodyid[jnt_id]
                chain_bodies.append(bid)
        if not chain_bodies:
            return "", ""

        # End-effector = deepest body (highest child depth).
        # Anchor = shallowest body (closest to root).
        def _depth(bid: int) -> int:
            d = 0
            cur = bid
            while cur > 0:
                cur = body._model.body_parentid[cur]
                d += 1
            return d

        depths = [(bid, _depth(bid)) for bid in set(chain_bodies)]
        depths.sort(key=lambda x: x[1])
        anchor_id = depths[0][0]
        ee_id = depths[-1][0]
        # The actual end-effector is often a child of the deepest actuated body
        # (e.g., "r_hand" is a child of "r_forearm"). Walk one child deeper.
        for ci in range(body._model.nbody):
            if body._model.body_parentid[ci] == ee_id:
                ee_id = ci
                break
        return body._model.body(ee_id).name, body._model.body(anchor_id).name

    def reset(self, body: MuJoCoBody) -> None:
        body.reset()
        self._step_count = 0
        self._done = False
        self._ee_body, self._anchor_body = self._discover_ee(body)
        if not self._ee_body:
            self._done = True
            return
        anchor_pos = body._data.body(self._anchor_body).xpos.copy()
        rng = np.random.default_rng()
        d = rng.standard_normal(3)
        n = np.linalg.norm(d)
        if n > 0:
            d /= n
        self._target = anchor_pos + d * rng.uniform(0.1, self._reach_radius)
        ee_pos = body._data.body(self._ee_body).xpos
        self._initial_dist = max(float(np.linalg.norm(ee_pos - self._target)), 1e-4)

    def evaluate(self, body: MuJoCoBody) -> TaskResult:
        self._step_count += 1
        if not self._ee_body:
            return TaskResult(0.0, False, 0.0, {"reason": "no end-effector"})
        ee_pos = body._data.body(self._ee_body).xpos
        dist = float(np.linalg.norm(ee_pos - self._target))
        progress = max(0.0, min(1.0, 1.0 - dist / self._initial_dist))
        if dist <= self._tolerance:
            self._done = True
            return TaskResult(1.0, True, 1.0, {"distance": dist})
        reward = float(np.clip(-dist / self._initial_dist, -1.0, 0.0))
        return TaskResult(
            reward, False, progress, {"distance": dist, "target": self._target.tolist()}
        )

    def get_state(self) -> dict[str, Any]:
        s = super().get_state()
        s.update(
            target=self._target.tolist(),
            initial_dist=self._initial_dist,
            reach_radius=self._reach_radius,
            tolerance=self._tolerance,
            ee_body=self._ee_body,
            anchor_body=self._anchor_body,
        )
        return s

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._target = np.array(state.get("target", [0, 0, 0]), dtype=np.float64)
        self._initial_dist = state.get("initial_dist", 1.0)
        self._reach_radius = state.get("reach_radius", self._reach_radius)
        self._tolerance = state.get("tolerance", self._tolerance)
        self._ee_body = state.get("ee_body", "")
        self._anchor_body = state.get("anchor_body", "")


class HeadTrackTask(Task):
    """Orient head joints toward random target angles.

    Auto-discovers head joints from the body's channel_actuators["head"].
    Works with any number of head DOFs (1 joint, 2 joints, or more).
    """

    def __init__(self, tolerance_rad: float = 0.1, max_steps: int = 200) -> None:
        super().__init__("head_track", "head", max_steps)
        self._tolerance = tolerance_rad
        self._targets: dict[str, float] = {}  # joint_name -> target_rad
        self._initial_error: float = 1.0

    def is_compatible(self, body: MuJoCoBody) -> bool:
        return "head" in body._channel_actuators and len(body._channel_actuators["head"]) > 0

    def _get_head_joints(self, body: MuJoCoBody) -> list[str]:
        """Get joint names for head channel actuators."""
        joints = []
        for act_name in body._channel_actuators.get("head", []):
            act_idx = body._actuator_idx.get(act_name)
            if act_idx is not None:
                jnt_id = body._model.actuator_trnid[act_idx, 0]
                joints.append(body._model.joint(jnt_id).name)
        return joints

    def reset(self, body: MuJoCoBody) -> None:
        body.reset()
        self._step_count = 0
        self._done = False
        rng = np.random.default_rng()
        self._targets = {}
        for jname in self._get_head_joints(body):
            jnt_id = body._model.joint(jname).id
            jnt_range = body._model.jnt_range[jnt_id]
            self._targets[jname] = float(rng.uniform(jnt_range[0], jnt_range[1]))
        self._initial_error = max(self._error(body), 1e-4)

    def _error(self, body: MuJoCoBody) -> float:
        total = 0.0
        for jname, target in self._targets.items():
            jnt_id = body._model.joint(jname).id
            qpos_adr = body._model.jnt_qposadr[jnt_id]
            total += abs(float(body._data.qpos[qpos_adr]) - target)
        return total

    def evaluate(self, body: MuJoCoBody) -> TaskResult:
        self._step_count += 1
        if not self._targets:
            return TaskResult(0.0, False, 0.0, {"reason": "no head joints"})
        errors = {}
        all_ok = True
        for jname, target in self._targets.items():
            jnt_id = body._model.joint(jname).id
            qpos_adr = body._model.jnt_qposadr[jnt_id]
            e = abs(float(body._data.qpos[qpos_adr]) - target)
            errors[jname] = e
            if e > self._tolerance:
                all_ok = False
        total_error = sum(errors.values())
        progress = max(0.0, min(1.0, 1.0 - total_error / self._initial_error))
        if all_ok:
            self._done = True
            return TaskResult(1.0, True, 1.0, {"errors": errors})
        reward = float(np.clip(-total_error / self._initial_error, -1.0, 0.0))
        return TaskResult(reward, False, progress, {"errors": errors, "targets": self._targets})

    def get_state(self) -> dict[str, Any]:
        s = super().get_state()
        s.update(
            targets=self._targets, initial_error=self._initial_error, tolerance=self._tolerance
        )
        return s

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._targets = state.get("targets", {})
        self._initial_error = state.get("initial_error", 1.0)
        self._tolerance = state.get("tolerance", self._tolerance)


class WalkTask(Task):
    """Move root body forward by target_distance while staying upright."""

    def __init__(self, target_distance: float = 0.5, max_steps: int = 500) -> None:
        super().__init__("walk", "locomotion", max_steps)
        self._target_distance = target_distance
        self._start_x: float = 0.0
        self._prev_x: float = 0.0
        self._initial_z: float = 1.2

    def reset(self, body: MuJoCoBody) -> None:
        body.reset()
        self._step_count = 0
        self._done = False
        self._initial_z = body._initial_root_z
        self._start_x = float(body._data.body(body._root_body_name).xpos[0])
        self._prev_x = self._start_x

    def evaluate(self, body: MuJoCoBody) -> TaskResult:
        self._step_count += 1
        pos = body._data.body(body._root_body_name).xpos
        cx, cz = float(pos[0]), float(pos[2])
        if body.is_fallen():
            self._done = True
            d = cx - self._start_x
            return TaskResult(
                -1.0,
                False,
                min(1.0, max(0.0, d / self._target_distance)),
                {"root_z": cz, "distance": d, "reason": "fallen"},
            )
        d = cx - self._start_x
        progress = min(1.0, max(0.0, d / self._target_distance))
        fwd = cx - self._prev_x
        self._prev_x = cx
        if d >= self._target_distance:
            self._done = True
            return TaskResult(1.0, True, 1.0, {"distance": d})
        reward = float(np.clip(fwd * 10.0, -0.5, 0.5))
        return TaskResult(
            reward, False, progress, {"distance": d, "forward_delta": fwd, "root_z": cz}
        )

    def get_state(self) -> dict[str, Any]:
        s = super().get_state()
        s.update(
            start_x=self._start_x,
            prev_x=self._prev_x,
            target_distance=self._target_distance,
            initial_z=self._initial_z,
        )
        return s

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._start_x = state.get("start_x", 0.0)
        self._prev_x = state.get("prev_x", 0.0)
        self._target_distance = state.get("target_distance", self._target_distance)
        self._initial_z = state.get("initial_z", 1.2)


class TaskCurriculum:
    """Progressive task difficulty. Only includes tasks compatible with the body."""

    def __init__(self, body: MuJoCoBody) -> None:
        self._body = body
        # Build task list, filtering out tasks incompatible with this body
        all_tasks = [
            SupportedStandTask(),
            StandTask(),
            HeadTrackTask(),
            BalanceTask(),
            ReachTask(),
            WalkTask(),
        ]
        self._tasks: list[Task] = [t for t in all_tasks if t.is_compatible(body)]
        if not self._tasks:
            logger.warning("No compatible tasks for this body configuration")
            self._tasks = [StandTask()]  # fallback
        self._current_idx: int = 0
        self._successes: int = 0
        self._advance_threshold: int = 50
        self._total_advances: int = 0
        self._tasks[0].reset(body)
        logger.info(
            "TaskCurriculum: %d tasks active: %s",
            len(self._tasks),
            [t.name for t in self._tasks],
        )

    @property
    def current_task(self) -> Task:
        return self._tasks[self._current_idx]

    def step(self) -> TaskResult:
        """Evaluate current task. Auto-reset/advance when complete."""
        task = self.current_task
        if task.is_complete:
            self.advance()
            task = self.current_task
            task.reset(self._body)
        result = task.evaluate(self._body)
        if result.success:
            self._successes += 1
            logger.debug(
                "Task '%s' succeeded (%d/%d)", task.name, self._successes, self._advance_threshold
            )
        elif task.is_complete:
            # Decay instead of wipe -- one failed episode shouldn't erase
            # all progress.  Halving means ~7 consecutive failures to reach 0.
            self._successes = self._successes // 2
        return result

    def advance(self) -> bool:
        """Move to next task if enough successes. Returns True if advanced."""
        if self._successes >= self._advance_threshold:
            prev = self.current_task.name
            self._current_idx = (self._current_idx + 1) % len(self._tasks)
            self._successes = 0
            self._total_advances += 1
            logger.info(
                "Curriculum: '%s' -> '%s' (#%d)", prev, self.current_task.name, self._total_advances
            )
            return True
        return False

    def reset_to_first(self) -> None:
        """Reset curriculum back to the first task (e.g., supported_stand).

        Used when deploying motor learning fixes to re-train from scratch
        with improved parameters.
        """
        self._current_idx = 0
        self._successes = 0
        self._tasks[0].reset(self._body)
        logger.info("Curriculum reset to '%s'", self.current_task.name)

    def apply_continuous_support(self, body: MuJoCoBody) -> None:
        """Apply support force at physics rate. No-op if current task has no support."""
        task = self.current_task
        if isinstance(task, SupportedStandTask):
            task.apply_continuous_support(body)

    @property
    def support_active(self) -> bool:
        """True when current task is SupportedStandTask and support force is on."""
        task = self.current_task
        if isinstance(task, SupportedStandTask):
            return task._support_active
        return False

    def get_state(self) -> dict[str, Any]:
        return {
            "current_idx": self._current_idx,
            "successes": self._successes,
            "advance_threshold": self._advance_threshold,
            "total_advances": self._total_advances,
            "tasks": [t.get_state() for t in self._tasks],
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._current_idx = state.get("current_idx", 0)
        self._successes = state.get("successes", 0)
        self._advance_threshold = state.get("advance_threshold", 50)
        self._total_advances = state.get("total_advances", 0)
        for i, ts in enumerate(state.get("tasks", [])):
            if i < len(self._tasks):
                self._tasks[i].set_state(ts)


def task_result_to_outcome(result: TaskResult, task: Task, body: MuJoCoBody) -> dict[str, Any]:
    """Convert TaskResult to motor.outcome-compatible dict for the feedback loop."""
    return {
        "channel": task.channel,
        "success": result.success,
        "confidence": abs(result.reward),
        "proprioceptive_state": body.get_proprioceptive_vector().tolist(),
        "error_magnitude": max(0.0, -result.reward),
        "task_name": task.name,
        "task_progress": result.progress,
    }
