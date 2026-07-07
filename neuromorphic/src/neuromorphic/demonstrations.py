"""Phase 2.14 (2026-06-17): Demonstration pipeline.

Provides a body- and skill-agnostic interface for injecting external
demonstrations into the brain. Phase 2.14a delivers the data pipeline
+ dashboard viewer. Phase 2.14b (same patch) wires brain-side injection
as a SEPARATE sensory modality so the brain can distinguish "what I am
feeling" (real proprioception) from "what the ghost is showing me"
(demonstration).

Design intent:

- ``DemonstrationProvider`` interface is generic. Future providers
  (motion-capture playback, video-derived pose estimation, other-brain
  motor trace) ship under the same interface and produce the same
  ``DemonstrationFrame`` shape.
- The initial provider, ``KinematicGaitGenerator``, is a clean phase
  oscillator producing deterministic walking joint angles. We use a
  phase oscillator -- not an LIF-spiking CPG -- because the LIF model
  in ``regions.LocomotorCPG`` needs brainstem drive to fire reliably
  and we want the demonstration to look correct from step 1 with no
  warmup. Biologically a CPG IS a phase oscillator; the LIF version is
  a downstream implementation choice. Same target trajectory.
- Brain-side: the ghost trajectory is published as
  ``observation.ghost_proprioceptive`` (a NEW modality) so the brain
  receives it on a separate sub-range of sensory cortex, satisfying
  patent Claim 4 (modality-specific sub-ranges) and giving the brain a
  natural way to distinguish self vs ghost. The brain learns the
  binding via STDP on sensory->association (also Claim 4).
"""

from __future__ import annotations

import json
import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DemonstrationFrame:
    """One step of demonstration data.

    Body- and skill-agnostic. ``joint_angles`` is the lingua franca:
    every demonstration source (kinematic gait, mocap, video pose,
    other-brain motor trace) can produce per-joint target angles in
    radians. ``foot_contacts`` and ``phase`` are optional gait-
    specific fields; non-locomotion providers leave them empty.
    """

    timestamp: float  # service monotonic clock
    source: str  # provider identifier
    joint_angles: dict[str, float]  # joint name -> radians
    foot_contacts: dict[str, bool] = field(default_factory=dict)
    phase: float = 0.0  # gait cycle phase [0, 1)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable dict for NATS transport."""
        return {
            "timestamp": float(self.timestamp),
            "source": self.source,
            "joint_angles": {k: float(v) for k, v in self.joint_angles.items()},
            "foot_contacts": {k: bool(v) for k, v in self.foot_contacts.items()},
            "phase": float(self.phase),
            "metadata": dict(self.metadata),
        }


class DemonstrationProvider(ABC):
    """Pluggable source of demonstration frames."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def step(self, timestamp: float = 0.0) -> DemonstrationFrame | None:
        """Produce one frame, or None if the provider is idle this tick.

        ``timestamp`` is the service-side monotonic clock; providers
        embed it into the frame so downstream consumers can align
        frames with brain steps.

        Idle-tick behavior lets the provider gate itself (e.g. only
        produce frames during walk-family tasks); the service does not
        need to know provider-internal logic.
        """

    def is_active(self) -> bool:
        """Whether the provider is currently demonstrating.

        Default True; subclasses override to gate by task or input
        availability.
        """
        return True

    def reset(self) -> None:
        """Reset provider state. Default no-op."""
        pass

    @property
    def joint_order(self) -> tuple[str, ...]:
        """Stable joint name order for the ghost proprio vector encoder.

        Subclasses override when they publish joints other than the
        walker_4dof default. H1 (2026-06-20): added so the same encoding
        path can produce a positional vector for any body's joint set
        without the brain needing to know joint names.
        """
        return ("hip_left", "hip_right", "knee_left", "knee_right")

    @property
    def angle_range(self) -> tuple[float, float]:
        """(lo, hi) normalization bounds the encoder clips against.

        Default tracks the walker amplitudes (`-_HIP_AMP, +_KNEE_AMP`).
        H1.1 (2026-06-20): humanoid knee flexes negative to -_KNEE_BEND
        and would clip against the walker default's `-0.45` lower bound,
        so HumanoidGaitGenerator overrides this to widen the range.
        """
        return (-_HIP_AMPLITUDE_RAD, _KNEE_AMPLITUDE_RAD)


# ---------------------------------------------------------------------------
# KinematicGaitGenerator -- initial provider
# ---------------------------------------------------------------------------
#
# Deterministic phase oscillator: phase advances by ``rate_hz * dt`` per
# step, joint angles are sine functions of phase with appropriate
# offsets. No noise, no warmup, no LIF dynamics. From step 1 the output
# is a clean walking trajectory.
#
# Mapping to walker_4dof joints:
#   hip_left  =  HIP_AMP * sin(2*pi*phase)
#   hip_right =  HIP_AMP * sin(2*pi*(phase + 0.5))      # anti-phase
#   knee_left =  KNEE_AMP * max(0, sin(2*pi*phase + 0.4 pi))   # bent during swing
#   knee_right=  KNEE_AMP * max(0, sin(2*pi*(phase+0.5) + 0.4 pi))
#
# Foot contact: extensor leg (negative hip angle, stance phase) is in
# contact. The mapping plus the user-visible dashboard renderer make it
# obvious whether this looks like walking.


_HIP_AMPLITUDE_RAD = 0.45  # ~26 degrees -- hip flexion / extension
_KNEE_AMPLITUDE_RAD = 0.70  # ~40 degrees -- knee bend during swing
_KNEE_PHASE_OFFSET = 0.20  # fraction of cycle -- knee leads hip slightly


class KinematicGaitGenerator(DemonstrationProvider):
    """Phase-oscillator gait demonstrator.

    Pure NumPy, deterministic, microseconds per step. The semantics
    are: ``phase`` 0..1 is the gait cycle, joint angles are smooth
    functions of phase. No dependence on the brain's spiking CPG --
    this is a clean reference the brain can compare its proprio
    against.
    """

    def __init__(self, rate_hz: float = 1.4, dt_seconds: float = 0.1) -> None:
        # 1.4 Hz step rate ~ moderate walking gait. dt 0.1 s matches the
        # default 10 Hz demonstration publish rate. The math is dimensionally
        # safe at any (rate, dt) combination.
        self._rate_hz = float(rate_hz)
        self._dt = float(dt_seconds)
        self._phase = 0.0
        self._tick_count = 0

    @property
    def name(self) -> str:
        return "cpg_walker"  # kept under this name for compose compatibility

    def reset(self) -> None:
        self._phase = 0.0
        self._tick_count = 0

    def step(self, timestamp: float = 0.0) -> DemonstrationFrame | None:
        # Advance the gait phase.
        self._phase = (self._phase + self._rate_hz * self._dt) % 1.0
        self._tick_count += 1
        p = self._phase
        tau = 2.0 * math.pi
        # Hips: smooth sinusoid, anti-phase legs.
        hip_left = _HIP_AMPLITUDE_RAD * math.sin(tau * p)
        hip_right = _HIP_AMPLITUDE_RAD * math.sin(tau * (p + 0.5))
        # Knees: bend during swing (peaks when hip is at max flexion).
        # Half-wave rectified sine so knee only goes positive.
        knee_left = _KNEE_AMPLITUDE_RAD * max(0.0, math.sin(tau * (p + _KNEE_PHASE_OFFSET)))
        knee_right = _KNEE_AMPLITUDE_RAD * max(0.0, math.sin(tau * (p + 0.5 + _KNEE_PHASE_OFFSET)))
        # Foot contact: leg is in contact when hip extends (negative angle).
        # During the second half of each leg's cycle the foot is planted.
        foot_contacts = {
            "left": hip_left < 0.0 and knee_left < 0.05,
            "right": hip_right < 0.0 and knee_right < 0.05,
        }
        return DemonstrationFrame(
            timestamp=timestamp,
            source=self.name,
            joint_angles={
                "hip_left": hip_left,
                "hip_right": hip_right,
                "knee_left": knee_left,
                "knee_right": knee_right,
            },
            foot_contacts=foot_contacts,
            phase=p,
            metadata={"tick": self._tick_count, "rate_hz": self._rate_hz},
        )


# Backward-compat alias for the earlier name. Same class.
CPGKinematicWalker = KinematicGaitGenerator


# ---------------------------------------------------------------------------
# H1 (2026-06-20): HumanoidGaitGenerator -- 29-DOF humanoid gait ghost
# ---------------------------------------------------------------------------
#
# Walker's `cpg_walker` publishes only the four joint names walker_4dof
# uses (`hip_left`, `hip_right`, `knee_left`, `knee_right`). The 29-DOF
# humanoid body has none of those names -- joints are `r_hip_pitch`,
# `r_knee`, etc. -- so the A3 ghost-tracking reward (cosine similarity
# over shared joint names) was silently zero on the humanoid.
#
# This provider produces the same kinematic gait shape mapped onto the
# humanoid's joint set: sagittal hip/knee/ankle pitches drive the legs,
# anti-phase shoulder pitches give natural arm swing, slight elbow bend
# during forward swing, a constant subtle waist forward lean.
#
# Knee sign convention: humanoid r_knee/l_knee MuJoCo range is `-145 0`
# (flexion is NEGATIVE), unlike walker_4dof where knee flexion is
# positive. Knee output here is NEGATIVE during swing.
#
# Operator dial: `amplitude_scale` scales every joint output. 0.0 = pure
# zero-pose target (standing still) for early supported_stand training.
# 1.0 = full walking ghost. The env knob `NEURO_DEMONSTRATION_AMPLITUDE`
# exposes this without code change.

_HIP_PITCH_AMP_RAD = 0.45  # ~26 deg flexion / extension
_KNEE_BEND_AMP_RAD = 0.70  # ~40 deg knee bend (applied as negative)
_ANKLE_PITCH_AMP_RAD = 0.20  # subtle foot pitch for ground clearance
_SHOULDER_PITCH_AMP_RAD = 0.35  # ~20 deg arm swing
_ELBOW_BASELINE_RAD = 0.15  # slight standing elbow bend
_ELBOW_SWING_AMP_RAD = 0.10
_WAIST_PITCH_LEAN_RAD = 0.05  # ~3 deg constant forward lean


class HumanoidGaitGenerator(DemonstrationProvider):
    """Phase-oscillator gait demonstrator for humanoid_29dof bodies.

    Pure NumPy, deterministic. Publishes 11 sagittal-plane joints whose
    names match `models/humanoid_29dof.xml`. Anti-phase legs, contralateral
    arm swing, subtle ankle pitch + waist lean. Amplitude scale defaults
    to 1.0 (full walking) and can be dialed to 0.0 (pure standing target)
    via `amplitude_scale=0.0` or `NEURO_DEMONSTRATION_AMPLITUDE=0`.

    The brain uses the joint_angles dict for the A3 ghost-tracking reward
    (cosine over shared joint names with the body qpos) and the encoded
    proprio vector for the ghost sensory sub-range.
    """

    _JOINT_ORDER: tuple[str, ...] = (
        "r_hip_pitch",
        "l_hip_pitch",
        "r_knee",
        "l_knee",
        "r_ankle_pitch",
        "l_ankle_pitch",
        "r_shoulder_pitch",
        "l_shoulder_pitch",
        "r_elbow",
        "l_elbow",
        "waist_pitch",
    )

    def __init__(
        self,
        rate_hz: float = 1.4,
        dt_seconds: float = 0.1,
        amplitude_scale: float = 1.0,
    ) -> None:
        self._rate_hz = float(rate_hz)
        self._dt = float(dt_seconds)
        self._amp = float(amplitude_scale)
        self._phase = 0.0
        self._tick_count = 0

    @property
    def name(self) -> str:
        return "cpg_humanoid"

    @property
    def joint_order(self) -> tuple[str, ...]:
        return _GETUP_JOINT_ORDER

    @property
    def angle_range(self) -> tuple[float, float]:
        # Knee flexion is negative on humanoid; widen the lower bound so
        # the encoder does not clip knee at -1.0. Upper bound matches the
        # walker default for symmetry with arm_swing / hip flexion.
        return (-_KNEE_BEND_AMP_RAD, _KNEE_BEND_AMP_RAD)

    def reset(self) -> None:
        self._phase = 0.0
        self._tick_count = 0

    def step(self, timestamp: float = 0.0) -> DemonstrationFrame | None:
        self._phase = (self._phase + self._rate_hz * self._dt) % 1.0
        self._tick_count += 1
        p = self._phase
        tau = 2.0 * math.pi
        s = self._amp
        # Anti-phase legs. Right leg leads with phase 0; left at phase 0.5.
        r_hip = s * _HIP_PITCH_AMP_RAD * math.sin(tau * p)
        l_hip = s * _HIP_PITCH_AMP_RAD * math.sin(tau * (p + 0.5))
        # Knee bend during forward swing -- humanoid knee flexion is NEGATIVE.
        r_knee = -s * _KNEE_BEND_AMP_RAD * max(0.0, math.sin(tau * (p + _KNEE_PHASE_OFFSET)))
        l_knee = -s * _KNEE_BEND_AMP_RAD * max(0.0, math.sin(tau * (p + 0.5 + _KNEE_PHASE_OFFSET)))
        # Ankle dorsiflexion to clear the ground during swing.
        r_ankle = s * _ANKLE_PITCH_AMP_RAD * math.sin(tau * (p + _KNEE_PHASE_OFFSET))
        l_ankle = s * _ANKLE_PITCH_AMP_RAD * math.sin(tau * (p + 0.5 + _KNEE_PHASE_OFFSET))
        # Contralateral arm swing: left arm forward when right leg forward.
        r_sho = s * _SHOULDER_PITCH_AMP_RAD * math.sin(tau * (p + 0.5))
        l_sho = s * _SHOULDER_PITCH_AMP_RAD * math.sin(tau * p)
        # Slight elbow flexion during forward arm swing.
        r_elbow = s * (
            _ELBOW_BASELINE_RAD + _ELBOW_SWING_AMP_RAD * max(0.0, math.sin(tau * (p + 0.5)))
        )
        l_elbow = s * (_ELBOW_BASELINE_RAD + _ELBOW_SWING_AMP_RAD * max(0.0, math.sin(tau * p)))
        # Subtle steady forward lean.
        waist = s * _WAIST_PITCH_LEAN_RAD
        # Foot contact: leg in stance when hip is extending and knee near
        # straight (knee close to 0). Mirrors walker convention.
        foot_contacts = {
            "left": l_hip < 0.0 and l_knee > -0.05,
            "right": r_hip < 0.0 and r_knee > -0.05,
        }
        return DemonstrationFrame(
            timestamp=timestamp,
            source=self.name,
            joint_angles={
                "r_hip_pitch": r_hip,
                "l_hip_pitch": l_hip,
                "r_knee": r_knee,
                "l_knee": l_knee,
                "r_ankle_pitch": r_ankle,
                "l_ankle_pitch": l_ankle,
                "r_shoulder_pitch": r_sho,
                "l_shoulder_pitch": l_sho,
                "r_elbow": r_elbow,
                "l_elbow": l_elbow,
                "waist_pitch": waist,
            },
            foot_contacts=foot_contacts,
            phase=p,
            metadata={
                "tick": self._tick_count,
                "rate_hz": self._rate_hz,
                "amplitude": self._amp,
            },
        )


# ---------------------------------------------------------------------------
# Phase 2.17 A2-followup (2026-06-22): HumanoidGetupGenerator
# ---------------------------------------------------------------------------
#
# Scripted supine -> stand trajectory for humanoid_29dof. The brain
# observes joint targets via observation.ghost_proprioceptive and learns
# to imitate. Mirrors cpg_walker / cpg_humanoid scaffolding.
#
# Trajectory phases (body time, seconds):
#   0   - 1  : settle on back (all joints 0)
#   1   - 4  : roll onto right side (waist_roll, hip_roll, shoulder pitch
#              left/right swing across body, right knee curls)
#   4   - 7  : prop on right elbow (r_elbow flexes, waist_pitch sits up
#              partly, left hip pulls leg under)
#   7   - 10 : sit cross-legged (waist back to neutral, both hips flex,
#              both knees bend)
#   10  - 12 : hands-and-knees (deep waist forward lean, shoulder pitch
#              negative so arms reach forward to the ground)
#   12  - 14 : push to crouch (lift hands off ground, hip + knee partially
#              extend)
#   14  - 16 : half-stand transition
#   16  - 18 : full stand (all joints back to neutral)
#   18+      : hold neutral standing pose (target for the brain to lock in)
#
# All 11 target joints are written every frame so linear interpolation is
# continuous (no "snap back to zero" between waypoints that omit a joint).
# Joints not in the trajectory (wrist, hip_yaw, ankle_roll, neck) stay at
# 0 throughout and are NOT published -- the brain's A3 cosine matches
# on the joint-name intersection, so unpublished joints contribute zero.
#
# Sign conventions verified against models/humanoid_29dof.xml:
#   r_knee / l_knee range = [-145, 0] deg -> flexion is NEGATIVE
#   r_hip_pitch / l_hip_pitch range = [-30, 120] deg -> forward is POSITIVE
#   waist_pitch range = [-30, 45] deg -> forward bend is POSITIVE
#   waist_roll range = [-30, 30] deg -> roll right is POSITIVE
#   r_hip_roll range = [-25, 45] deg -> abduction-direction is POSITIVE
#
# Standing root_z ~ 0.92 m (pelvis pos in XML). Trajectory validation
# (scripts/test_getup_trajectory.py) asserts max_root_z >= 0.20 m and
# no NaN over the full PD-driven trajectory. The bar is a demonstration
# feasibility check, not a kinematic standing test -- see TUNING-LOG.md
# "2026-06-22 (evening): getup trajectory test unblock" for rationale.

_GETUP_TOTAL_S_DEFAULT = 18.0

# Canonical joint set the getup trajectory targets. Single source of
# truth used by _GETUP_WAYPOINTS (built via _wp), HumanoidGetupGenerator
# (exposed as the joint_order property), and the joint-range test in
# tests/test_cpg_humanoid_getup.py. Adding a 12th joint = update this
# tuple only; waypoint dicts auto-extend via _NEUTRAL_POSE.
_GETUP_JOINT_ORDER: tuple[str, ...] = (
    "waist_roll",
    "waist_pitch",
    "r_shoulder_pitch",
    "l_shoulder_pitch",
    "r_elbow",
    "l_elbow",
    "r_hip_roll",
    "r_hip_pitch",
    "l_hip_pitch",
    "r_knee",
    "l_knee",
)
_NEUTRAL_POSE: dict[str, float] = {j: 0.0 for j in _GETUP_JOINT_ORDER}


def _wp(**overrides: float) -> dict[str, float]:
    """Build a full waypoint dict from the canonical neutral pose plus
    per-joint overrides. Keeps every waypoint guaranteed-complete
    (every canonical joint always present) without requiring each
    entry to spell out the zeros."""
    unknown = set(overrides) - set(_GETUP_JOINT_ORDER)
    if unknown:
        raise ValueError(
            f"Unknown joints in waypoint override: {sorted(unknown)} "
            f"(canonical set: {_GETUP_JOINT_ORDER})"
        )
    return {**_NEUTRAL_POSE, **overrides}


# Each tuple is (t_seconds, joint -> radians dict). Joints kept stable
# (always written) so linear interpolation between any two adjacent
# waypoints is continuous. After the last waypoint, the trajectory holds
# the final pose indefinitely (the brain's target while standing).
# 2026-06-22 (evening): waypoints re-authored for the actual side-lying
# start pose (the "supine" keyframe is empirically a right-side-lying
# pose; see TUNING-LOG.md and scripts/probe_shoulder_signs.py for the
# per-joint probe outputs). Sign conventions used here, in side-lying:
#   waist_pitch    NEGATIVE = sit-up direction (torso rises +z)
#                  POSITIVE = drops torso into floor (avoid early)
#   r_shoulder_pitch POSITIVE = right arm rises off floor (away from
#                                 -z) and toward +x (head direction)
#                    NEGATIVE = right arm drives into floor
#                                 (useful AFTER body has rotated)
#   l_shoulder_pitch POSITIVE = left arm lifts strongly (already
#                                 starts at z=0.36 in side-lying)
#   r_hip_pitch / l_hip_pitch  POSITIVE = leg lifts (sweep toward
#                                 head), most powerful vertical
#                                 driver available in side-lying
#   r_knee / l_knee NEGATIVE = flexion
#   waist_roll POSITIVE = rolls torso further onto right side (avoid)
#                NEGATIVE = unrolls toward back
_GETUP_WAYPOINTS: tuple[tuple[float, dict[str, float]], ...] = (
    # settle: body lying on right side, all joints neutral.
    (0.0, _wp()),
    (1.0, _wp()),
    # roll: build angular momentum upward off right side. Left arm
    # sweeps overhead and left leg lifts -- both throw mass away from
    # the floor. Right arm plants slightly so the right elbow becomes
    # the pivot. waist_pitch negative starts the sit-up direction.
    (
        4.0,
        _wp(
            waist_pitch=-0.3,
            r_shoulder_pitch=-0.4,
            l_shoulder_pitch=1.4,
            r_elbow=0.3,
            l_elbow=0.3,
            r_hip_pitch=0.4,
            l_hip_pitch=0.9,
            r_knee=-0.5,
            l_knee=-1.0,
        ),
    ),
    # prop: lever upper body off the floor on right elbow. waist_pitch
    # at max negative (range max is -0.52). r_shoulder_pitch positive
    # actively lifts the right shoulder; r_elbow folds forearm under
    # torso (upper arm pinned by body weight, elbow flex pushes
    # shoulder up). Both knees tuck deeply so legs are out of the way.
    (
        7.0,
        _wp(
            waist_pitch=-0.5,
            r_shoulder_pitch=0.8,
            l_shoulder_pitch=1.2,
            r_elbow=1.2,
            l_elbow=0.8,
            r_hip_pitch=1.0,
            l_hip_pitch=1.2,
            r_knee=-1.3,
            l_knee=-1.4,
        ),
    ),
    # sit: pull body into compact seated configuration. waist_pitch
    # eased toward 0 (body more upright, less need for sit-up bias).
    # Knees + hips at deepest flex. Both arms tucked.
    (
        10.0,
        _wp(
            waist_pitch=-0.2,
            r_shoulder_pitch=0.5,
            l_shoulder_pitch=0.5,
            r_elbow=1.0,
            l_elbow=1.0,
            r_hip_pitch=1.2,
            l_hip_pitch=1.2,
            r_knee=-1.4,
            l_knee=-1.4,
        ),
    ),
    # kneel: rotate forward to hands-and-knees. waist_pitch flips sign
    # -- body no longer side-lying, positive is forward bend toward
    # floor in front of body. Shoulders reach arms forward to plant.
    (
        12.0,
        _wp(
            waist_pitch=0.5,
            r_shoulder_pitch=-1.0,
            l_shoulder_pitch=-1.0,
            r_elbow=0.1,
            l_elbow=0.1,
            r_hip_pitch=1.2,
            l_hip_pitch=1.2,
            r_knee=-1.4,
            l_knee=-1.4,
        ),
    ),
    # crouch: hands lift, hips/knees partly extend.
    (
        14.0,
        _wp(
            waist_pitch=0.3,
            r_shoulder_pitch=-0.4,
            l_shoulder_pitch=-0.4,
            r_elbow=0.3,
            l_elbow=0.3,
            r_hip_pitch=0.9,
            l_hip_pitch=0.9,
            r_knee=-1.0,
            l_knee=-1.0,
        ),
    ),
    # half_stand: continued extension.
    (
        16.0,
        _wp(
            waist_pitch=0.1,
            r_shoulder_pitch=-0.1,
            l_shoulder_pitch=-0.1,
            r_elbow=0.1,
            l_elbow=0.1,
            r_hip_pitch=0.4,
            l_hip_pitch=0.4,
            r_knee=-0.4,
            l_knee=-0.4,
        ),
    ),
    # stand: neutral upright.
    (18.0, _wp()),
)

_GETUP_PHASE_NAMES: tuple[tuple[float, str], ...] = (
    (1.0, "settle"),
    (4.0, "roll"),
    (7.0, "prop"),
    (10.0, "sit"),
    (12.0, "kneel"),
    (14.0, "crouch"),
    (16.0, "half_stand"),
    (18.0, "stand"),
)


class HumanoidGetupGenerator(DemonstrationProvider):
    """Scripted supine -> stand trajectory for humanoid_29dof.

    Linear interpolation between hand-tuned keyframes. After the last
    keyframe (default t=18 s), holds the final standing pose so the
    brain has a steady imitation target. Resets via reset() which the
    task system can call on episode start (operator concern -- not the
    provider's job to detect "fell back to supine").

    Body coupling: this provider hard-codes joint names that match
    humanoid_29dof.xml. On a different body, the published joint dict
    simply won't intersect the body's qpos and the A3 ghost-tracking
    reward will be zero -- no crash. Future bodies with a "getup-like"
    skill should ship their own provider.
    """

    def __init__(
        self,
        total_duration_s: float = _GETUP_TOTAL_S_DEFAULT,
        dt_seconds: float = 0.1,
        amplitude_scale: float = 1.0,
    ) -> None:
        if total_duration_s <= 0.0:
            raise ValueError(f"total_duration_s must be positive, got {total_duration_s}")
        self._total_s = float(total_duration_s)
        # Scale the waypoint anchor times so a custom duration stretches /
        # compresses the trajectory uniformly. Anchor in the table is built
        # against the default duration; scale factor preserves shape.
        self._time_scale = self._total_s / _GETUP_TOTAL_S_DEFAULT
        self._dt = float(dt_seconds)
        self._amp = float(amplitude_scale)
        self._body_time = 0.0
        self._tick_count = 0

    @property
    def name(self) -> str:
        return "cpg_humanoid_getup"

    @property
    def total_duration_s(self) -> float:
        """Total trajectory duration in seconds (post-construction).
        Read-only; set via the constructor."""
        return self._total_s

    @property
    def joint_order(self) -> tuple[str, ...]:
        return _GETUP_JOINT_ORDER

    @property
    def angle_range(self) -> tuple[float, float]:
        # Widest needed by any waypoint: r_knee bottoms at -1.4 rad on
        # the kneel phase; l_shoulder_pitch peaks at +1.5 rad during the
        # roll. (-1.5, 1.5) covers both without clipping. Symmetric for
        # encoder consistency.
        return (-1.5, 1.5)

    def reset(self) -> None:
        self._body_time = 0.0
        self._tick_count = 0

    def _scaled_waypoints(self) -> tuple[tuple[float, dict[str, float]], ...]:
        # Anchor times scaled by self._time_scale so a non-default
        # total_duration stretches the trajectory uniformly. Cheap pure
        # transformation; called per step but the cost is trivial relative
        # to the rest of step().
        ts = self._time_scale
        return tuple((t * ts, angles) for t, angles in _GETUP_WAYPOINTS)

    def _phase_name_at(self, t: float) -> str:
        ts = self._time_scale
        for boundary, name in _GETUP_PHASE_NAMES:
            if t <= boundary * ts:
                return name
        return "stand"

    def _interpolate(self, t: float) -> dict[str, float]:
        """Linear interpolation between adjacent waypoints. Holds the
        final pose for t > last waypoint time. Amplitude scale applied
        AFTER interpolation so 0.0 -> all joints stay at supine zeros.
        """
        wps = self._scaled_waypoints()
        # Edge cases: before first / after last waypoint.
        if t <= wps[0][0]:
            base = wps[0][1]
        elif t >= wps[-1][0]:
            base = wps[-1][1]
        else:
            # Find the bracketing pair.
            lo_t, lo_angles = wps[0]
            hi_t, hi_angles = wps[-1]
            for (a_t, a_a), (b_t, b_a) in zip(wps[:-1], wps[1:]):
                if a_t <= t <= b_t:
                    lo_t, lo_angles = a_t, a_a
                    hi_t, hi_angles = b_t, b_a
                    break
            span = hi_t - lo_t
            # span > 0 since we built waypoints with strictly increasing t.
            u = 0.0 if span <= 0.0 else (t - lo_t) / span
            base = {j: lo_angles[j] + u * (hi_angles[j] - lo_angles[j]) for j in _GETUP_JOINT_ORDER}
        if self._amp == 1.0:
            return dict(base)
        return {j: base[j] * self._amp for j in _GETUP_JOINT_ORDER}

    def step(self, timestamp: float = 0.0) -> DemonstrationFrame | None:
        # Advance body time. After total duration the body_time keeps
        # climbing but _interpolate clamps to the last waypoint, so the
        # frame stays at the standing pose. reset() puts us back at t=0.
        self._body_time += self._dt
        self._tick_count += 1
        t = self._body_time
        angles = self._interpolate(t)
        # phase: normalized progress in [0, 1]. Clamped after the last
        # waypoint so the brain sees a steady "trajectory complete" signal.
        progress = min(1.0, t / self._total_s)
        return DemonstrationFrame(
            timestamp=timestamp,
            source=self.name,
            joint_angles=angles,
            foot_contacts={},  # not gait; foot contacts come from real body
            phase=progress,
            metadata={
                "tick": self._tick_count,
                "body_time_s": t,
                "phase_name": self._phase_name_at(t),
                "amplitude": self._amp,
                "total_duration_s": self._total_s,
            },
        )


# ---------------------------------------------------------------------------
# Initiative 1 (2026-06-23): HumanoidStandGenerator
# ---------------------------------------------------------------------------
#
# Quiet neutral-standing ghost. Holds the final getup pose (zeros) with
# a small per-knee bias and a slow waist_pitch sway so the ghost looks
# alive, not frozen. Joint set is reused verbatim from
# `_GETUP_JOINT_ORDER` (iron rule: no new hardcoded joint names beyond
# the existing humanoid set). Designed to pair with the `stand` /
# `supported_stand` tasks; the factory warns if paired with anything
# else. Patent alignment: Claim 4 (ghost rides the modality-specific
# ghost_proprioceptive sub-range) and Claim 6 (steady target lets
# eligibility traces accumulate around the task reward instead of being
# torn between contradictory teaching signals).


class HumanoidStandGenerator(DemonstrationProvider):
    """Quiet standing ghost for the humanoid: neutral pose + slow sway.

    Conceptually the trailing waypoint of the getup trajectory held
    forever. Each step emits the canonical neutral pose with:

    - `knee_bias` added to both r_knee and l_knee (slight flex so the
      target is biomechanically plausible against gravity).
    - `ankle_bias` added to any ankle pitch joint that exists in
      `joint_order`. Reserved for forward compatibility: the current
      `_GETUP_JOINT_ORDER` has no ankle joints, so this kwarg is a
      no-op until ankles are added to the canonical set.
    - `waist_pitch` modulated by `sway_amp * sin(2*pi*sway_hz*t)` so
      the ghost looks alive rather than statue-frozen.

    Provider never idles: `is_active` is always True. `reset` zeroes
    the internal clock so a new episode starts at phase 0.
    """

    _ANKLE_JOINTS: tuple[str, ...] = (
        "r_ankle_pitch",
        "l_ankle_pitch",
    )

    def __init__(
        self,
        sway_amp: float = 0.05,
        sway_hz: float = 0.3,
        knee_bias: float = -0.1,
        ankle_bias: float = 0.05,
        dt_seconds: float = 0.1,
    ) -> None:
        self._sway_amp = float(sway_amp)
        self._sway_hz = float(sway_hz)
        self._knee_bias = float(knee_bias)
        self._ankle_bias = float(ankle_bias)
        self._dt = float(dt_seconds)
        self._body_time = 0.0
        self._tick_count = 0

    @property
    def name(self) -> str:
        return "cpg_humanoid_stand"

    @property
    def joint_order(self) -> tuple[str, ...]:
        return _GETUP_JOINT_ORDER

    @property
    def angle_range(self) -> tuple[float, float]:
        return (-0.5, 0.5)

    def is_active(self) -> bool:
        return True

    def reset(self) -> None:
        self._body_time = 0.0
        self._tick_count = 0

    def step(self, timestamp: float = 0.0) -> DemonstrationFrame | None:
        self._body_time += self._dt
        self._tick_count += 1
        t = self._body_time
        angles: dict[str, float] = {j: 0.0 for j in _GETUP_JOINT_ORDER}
        if "r_knee" in angles:
            angles["r_knee"] += self._knee_bias
        if "l_knee" in angles:
            angles["l_knee"] += self._knee_bias
        for ankle in self._ANKLE_JOINTS:
            if ankle in angles:
                angles[ankle] += self._ankle_bias
        if "waist_pitch" in angles:
            angles["waist_pitch"] += self._sway_amp * math.sin(2.0 * math.pi * self._sway_hz * t)
        # Phase: position in the sway cycle, normalized to [0, 1).
        period = 1.0 / self._sway_hz if self._sway_hz > 0.0 else 1.0
        phase = (t % period) / period if period > 0.0 else 0.0
        return DemonstrationFrame(
            timestamp=timestamp,
            source=self.name,
            joint_angles=angles,
            foot_contacts={"left": True, "right": True},
            phase=phase,
            metadata={
                "tick": self._tick_count,
                "body_time_s": t,
                "sway_amp": self._sway_amp,
                "sway_hz": self._sway_hz,
                "knee_bias": self._knee_bias,
                "ankle_bias": self._ankle_bias,
            },
        )


class HumanoidStandStepGenerator(DemonstrationProvider):
    """Stand-step-stand ghost for the humanoid: three-phase cycle.

    Paired demonstrator for StandStepStandTask. Cycles through:

    - **stand_a** (stand_a_seconds): identical neutral pose + slow waist
      sway to HumanoidStandGenerator. The brain sees the same target it
      learned during supported_stand training -- continuity matters.
    - **step** (step_seconds): half-sine swing on one leg's hip_pitch +
      knee, with the contralateral leg holding the neutral stand pose.
      Stepping leg alternates each cycle so the brain learns symmetric
      gait initiation rather than dominant-side bias.
    - **stand_b** (stand_b_seconds): neutral pose again. Same as
      stand_a; the brain rehearses the post-step recovery.

    Phase durations default to 1.5s / 1.0s / 1.5s = 4s cycle, matching
    the StandStepStandTask's 50 / 100 / 50 default budgets at the
    canonical 50 Hz proprio rate.

    Joint order: ``_GETUP_JOINT_ORDER`` (canonical humanoid_29dof set).
    Hip and knee joints are body-coupled by design per the existing
    HumanoidStandGenerator pattern; this generator stays gated to
    humanoid bodies via the demonstrator-task pairing warning in the
    factory.
    """

    def __init__(
        self,
        stand_a_seconds: float = 1.5,
        step_seconds: float = 1.0,
        stand_b_seconds: float = 1.5,
        sway_amp: float = 0.05,
        sway_hz: float = 0.3,
        knee_bias: float = -0.1,
        ankle_bias: float = 0.05,
        step_hip_amp: float = 0.5,
        step_knee_amp: float = 0.6,
        dt_seconds: float = 0.1,
    ) -> None:
        self._stand_a = float(stand_a_seconds)
        self._step = float(step_seconds)
        self._stand_b = float(stand_b_seconds)
        self._cycle = self._stand_a + self._step + self._stand_b
        self._sway_amp = float(sway_amp)
        self._sway_hz = float(sway_hz)
        self._knee_bias = float(knee_bias)
        self._ankle_bias = float(ankle_bias)
        self._step_hip_amp = float(step_hip_amp)
        self._step_knee_amp = float(step_knee_amp)
        self._dt = float(dt_seconds)
        self._body_time = 0.0
        self._tick_count = 0
        self._cycle_count = 0

    @property
    def name(self) -> str:
        return "cpg_humanoid_step"

    @property
    def joint_order(self) -> tuple[str, ...]:
        return _GETUP_JOINT_ORDER

    @property
    def angle_range(self) -> tuple[float, float]:
        return (-0.7, 0.7)

    def is_active(self) -> bool:
        return True

    def reset(self) -> None:
        self._body_time = 0.0
        self._tick_count = 0
        self._cycle_count = 0

    def _stand_pose(self, t_for_sway: float) -> dict[str, float]:
        """Neutral pose with optional sway -- identical math to
        HumanoidStandGenerator.step() so the brain sees a consistent
        target during the stand_a and stand_b phases.
        """
        angles: dict[str, float] = {j: 0.0 for j in _GETUP_JOINT_ORDER}
        if "r_knee" in angles:
            angles["r_knee"] += self._knee_bias
        if "l_knee" in angles:
            angles["l_knee"] += self._knee_bias
        for ankle in ("r_ankle_pitch", "l_ankle_pitch"):
            if ankle in angles:
                angles[ankle] += self._ankle_bias
        if "waist_pitch" in angles:
            angles["waist_pitch"] += self._sway_amp * math.sin(
                2.0 * math.pi * self._sway_hz * t_for_sway
            )
        return angles

    def _step_pose(self, phase_t: float, right_leg_swings: bool) -> dict[str, float]:
        """Pose during the step phase. phase_t in [0, self._step]."""
        progress = phase_t / max(self._step, 1e-6)
        # Half-sine swing: 0 -> 1 -> 0 across the step phase. The swing
        # leg's hip flexes forward and the knee bends; the stance leg
        # holds the neutral knee_bias pose.
        swing = math.sin(math.pi * progress)
        angles = self._stand_pose(t_for_sway=phase_t)
        swing_prefix = "r" if right_leg_swings else "l"
        swing_hip = f"{swing_prefix}_hip_pitch"
        swing_knee = f"{swing_prefix}_knee"
        if swing_hip in angles:
            angles[swing_hip] = self._step_hip_amp * swing
        if swing_knee in angles:
            # Bend the knee in addition to the standing knee_bias so the
            # foot clears the ground. Knee flex is negative-going past
            # the bias.
            angles[swing_knee] = self._knee_bias - self._step_knee_amp * swing
        return angles

    def step(self, timestamp: float = 0.0) -> DemonstrationFrame | None:
        self._body_time += self._dt
        self._tick_count += 1
        t = self._body_time % self._cycle
        cycle_index = int(self._body_time // self._cycle)
        if cycle_index != self._cycle_count:
            self._cycle_count = cycle_index

        if t < self._stand_a:
            phase = "stand_a"
            phase_t = t
            angles = self._stand_pose(t_for_sway=t)
            foot_contacts = {"left": True, "right": True}
        elif t < self._stand_a + self._step:
            phase = "step"
            phase_t = t - self._stand_a
            right_leg_swings = self._cycle_count % 2 == 0
            angles = self._step_pose(phase_t, right_leg_swings)
            # Swinging leg lifts mid-step (foot off the ground); other
            # leg stays planted. Both feet down at the very start and
            # end of the step window so the demonstrator's foot contacts
            # match the task's hold-then-step-then-hold expectation.
            progress = phase_t / max(self._step, 1e-6)
            airborne = 0.2 < progress < 0.8
            if right_leg_swings:
                foot_contacts = {"left": True, "right": not airborne}
            else:
                foot_contacts = {"left": not airborne, "right": True}
        else:
            phase = "stand_b"
            phase_t = t - self._stand_a - self._step
            angles = self._stand_pose(t_for_sway=phase_t)
            foot_contacts = {"left": True, "right": True}

        return DemonstrationFrame(
            timestamp=timestamp,
            source=self.name,
            joint_angles=angles,
            foot_contacts=foot_contacts,
            phase=float(t) / self._cycle if self._cycle > 0 else 0.0,
            metadata={
                "tick": self._tick_count,
                "cycle": self._cycle_count,
                "phase_name": phase,
                "phase_t_s": round(phase_t, 3),
                "body_time_s": round(self._body_time, 3),
                "right_leg_swings": (self._cycle_count % 2 == 0),
            },
        )


# ---------------------------------------------------------------------------
# Phase 2.15 (2026-06-17): VideoPoseProvider -- replay extracted pose JSON
# ---------------------------------------------------------------------------
#
# Reads a pre-extracted JSON file produced by the dashboard's pose
# extractor (mediapipe-based). File shape:
#
#   {
#     "video_id": "abc123",
#     "source_video": "walking.mp4",
#     "fps": 30.0,
#     "duration_seconds": 4.2,
#     "frames": [
#       {"hip_left": 0.1, "hip_right": -0.1, "knee_left": 0.0,
#        "knee_right": 0.4, "foot_left": false, "foot_right": true},
#       ...
#     ]
#   }
#
# The provider loops the trajectory indefinitely, advancing the frame
# index by `dt * fps` per step. dt is the inverse of the demonstration
# publish rate.
#
# Missing-file / malformed-file behaviour: provider is constructed with
# the JSON content already validated by the factory; runtime step()
# never raises -- it just returns the current frame or, in degenerate
# cases, a zero-pose frame.


_VIDEO_TRAJECTORY_DIR_DEFAULT = "/data/demonstrations/extracted"


class VideoPoseProvider(DemonstrationProvider):
    """Replays a pre-extracted joint-angle trajectory from a video file.

    The brain side is unaware of the source modality (video vs mocap
    vs cpg). Only the ``source`` string changes so the dashboard can
    label the demo.
    """

    def __init__(
        self,
        video_id: str,
        trajectory: dict,
        dt_seconds: float = 0.1,
    ) -> None:
        self._video_id = video_id
        self._fps = float(trajectory.get("fps", 30.0))
        self._frames = trajectory.get("frames", [])
        if not self._frames:
            raise ValueError(f"Video trajectory {video_id!r} has no frames")
        self._frame_count = len(self._frames)
        self._duration = float(
            trajectory.get("duration_seconds", self._frame_count / max(self._fps, 1.0))
        )
        self._dt = float(dt_seconds)
        # Frame index as float -- accumulate dt*fps so loops are smooth
        # at any (dt, fps) combination.
        self._frame_pos = 0.0
        self._source_video = trajectory.get("source_video", video_id)

    @property
    def name(self) -> str:
        return f"video:{self._video_id}"

    def reset(self) -> None:
        self._frame_pos = 0.0

    def step(self, timestamp: float = 0.0) -> DemonstrationFrame | None:
        # Advance frame position; wrap around.
        self._frame_pos = (self._frame_pos + self._dt * self._fps) % self._frame_count
        idx = int(self._frame_pos)
        frame_data = self._frames[idx]
        joint_angles = {
            "hip_left": float(frame_data.get("hip_left", 0.0)),
            "hip_right": float(frame_data.get("hip_right", 0.0)),
            "knee_left": float(frame_data.get("knee_left", 0.0)),
            "knee_right": float(frame_data.get("knee_right", 0.0)),
        }
        foot_contacts = {
            "left": bool(frame_data.get("foot_left", False)),
            "right": bool(frame_data.get("foot_right", False)),
        }
        # Phase = position within the loop, 0..1.
        phase = (idx / self._frame_count) if self._frame_count else 0.0
        return DemonstrationFrame(
            timestamp=timestamp,
            source=self.name,
            joint_angles=joint_angles,
            foot_contacts=foot_contacts,
            phase=phase,
            metadata={
                "frame_idx": idx,
                "frame_count": self._frame_count,
                "fps": self._fps,
                "source_video": self._source_video,
            },
        )


def _load_video_trajectory(video_id: str, base_dir: str | None = None) -> dict:
    """Read the extracted JSON for ``video_id`` from the shared volume.

    Raises FileNotFoundError if the trajectory has not been extracted.
    """
    base = Path(
        base_dir
        or os.environ.get(
            "NEURO_DEMONSTRATION_DATA_DIR",
            _VIDEO_TRAJECTORY_DIR_DEFAULT,
        )
    )
    path = base / f"{video_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Video trajectory not found: {path} " f"(extract via dashboard upload first)"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Brain-side encoding helper
# ---------------------------------------------------------------------------


def encode_frame_to_proprio_vector(
    frame: DemonstrationFrame,
    joint_order: tuple[str, ...] = (
        "hip_left",
        "hip_right",
        "knee_left",
        "knee_right",
    ),
    angle_range: tuple[float, float] = (-_HIP_AMPLITUDE_RAD, _KNEE_AMPLITUDE_RAD),
) -> list[float]:
    """Pack a frame into a fixed-length vector for transport as proprio.

    Order is stable (sorted joint names by default) so the brain side
    can rely on positional encoding without knowing joint names. Each
    angle is normalized to [-1, 1] using ``angle_range``. Phase
    appended last, normalized [0, 1] -> [-1, 1] so the brain has an
    explicit cycle position signal alongside the joint targets.

    The returned vector becomes the ``data`` field of an
    ``observation.ghost_proprioceptive`` NATS message, routed by the
    brain to a dedicated sub-range of sensory cortex (Patent Claim 4:
    modality-specific sub-ranges).
    """
    lo, hi = angle_range
    span = max(hi - lo, 1e-6)
    vec = []
    for j in joint_order:
        a = frame.joint_angles.get(j, 0.0)
        # Clamp and rescale to [-1, 1].
        clipped = max(lo, min(hi, a))
        vec.append(2.0 * (clipped - lo) / span - 1.0)
    # Phase: 0..1 -> -1..1
    vec.append(2.0 * frame.phase - 1.0)
    return [float(v) for v in vec]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_demonstration_provider(
    type_name: str | None = None,
    video_id: str | None = None,
) -> DemonstrationProvider | None:
    """Construct a provider from env, explicit type, or video_id.

    ``type_name`` defaults to ``NEURO_DEMONSTRATION_TYPE`` env var.
    Returns ``None`` when no provider is configured (the value
    ``"none"`` or empty), so the service can skip the publish path
    entirely without further conditionals.

    Phase 2.15: ``type_name="video"`` requires ``video_id``; loads the
    pre-extracted trajectory from the shared demonstrations volume.
    Falls back gracefully (logs + returns ``None``) when the file is
    missing, so a bad video_id does not crash the service.
    """
    name_raw = type_name or os.environ.get("NEURO_DEMONSTRATION_TYPE", "none") or "none"
    name = name_raw.lower()
    if name in ("", "none", "off", "0", "false"):
        return None
    dt = 1.0 / float(os.environ.get("NEURO_DEMONSTRATION_RATE", "10.0"))
    if name in ("cpg_walker", "cpg", "walker", "kinematic_gait", "gait"):
        rate_hz = float(os.environ.get("NEURO_DEMONSTRATION_GAIT_HZ", "1.4"))
        return KinematicGaitGenerator(rate_hz=rate_hz, dt_seconds=dt)
    if name in ("cpg_humanoid", "humanoid", "humanoid_gait"):
        rate_hz = float(os.environ.get("NEURO_DEMONSTRATION_GAIT_HZ", "1.4"))
        amp = float(os.environ.get("NEURO_DEMONSTRATION_AMPLITUDE", "1.0"))
        return HumanoidGaitGenerator(
            rate_hz=rate_hz,
            dt_seconds=dt,
            amplitude_scale=amp,
        )
    if name in ("cpg_humanoid_stand", "humanoid_stand"):
        # Pairing check: the stand ghost holds a neutral upright pose
        # with mild sway. If applied while NEURO_TASK_PIN points at a
        # locomotion task (walk_short, walk, etc.) the brain sees a
        # static target while the reward function demands forward
        # velocity. Warn loudly so operators catch the mismatch at
        # startup. Empty pin (auto-selection) is fine; `stand` /
        # `supported_stand` are the intended pairs.
        _task_pin = os.environ.get("NEURO_TASK_PIN", "").lower().strip()
        if _task_pin and _task_pin not in ("stand", "supported_stand"):
            logger.warning(
                "Demonstrator <-> task pairing mismatch: "
                "NEURO_DEMONSTRATION_TYPE=%s with NEURO_TASK_PIN=%s. "
                "The stand ghost holds a neutral pose; a locomotion "
                "task will fight it. Flip both together in "
                "deploy/docker-compose.1m.yml.",
                name_raw,
                _task_pin,
            )
        sway_amp = float(os.environ.get("NEURO_STAND_SWAY_AMP", "0.05"))
        sway_hz = float(os.environ.get("NEURO_STAND_SWAY_HZ", "0.3"))
        knee_bias = float(os.environ.get("NEURO_STAND_KNEE_BIAS", "-0.1"))
        ankle_bias = float(os.environ.get("NEURO_STAND_ANKLE_BIAS", "0.05"))
        return HumanoidStandGenerator(
            sway_amp=sway_amp,
            sway_hz=sway_hz,
            knee_bias=knee_bias,
            ankle_bias=ankle_bias,
            dt_seconds=dt,
        )
    if name in ("cpg_humanoid_step", "humanoid_step", "stand_step_stand"):
        # Pairing check: this ghost cycles stand -> step -> stand and is
        # designed to teach the compound StandStepStandTask. Paired with
        # any other task pin (balance, walk_short, etc.) it produces a
        # contradictory teaching signal (the static stand phases would
        # fight a sustained-locomotion task; the step phase would fight
        # a pure-stand task). Empty pin (auto-selection) is fine.
        _task_pin = os.environ.get("NEURO_TASK_PIN", "").lower().strip()
        if _task_pin and _task_pin != "stand_step_stand":
            logger.warning(
                "Demonstrator <-> task pairing mismatch: "
                "NEURO_DEMONSTRATION_TYPE=%s with NEURO_TASK_PIN=%s. "
                "The stand-step-stand ghost cycles stand -> step -> stand "
                "and will fight any task other than stand_step_stand. "
                "Flip both together in deploy/docker-compose.1m.yml.",
                name_raw,
                _task_pin,
            )
        stand_a_s = float(os.environ.get("NEURO_STAND_STEP_STAND_STAND_A_S", "1.5"))
        step_s = float(os.environ.get("NEURO_STAND_STEP_STAND_STEP_S", "1.0"))
        stand_b_s = float(os.environ.get("NEURO_STAND_STEP_STAND_STAND_B_S", "1.5"))
        sway_amp = float(os.environ.get("NEURO_STAND_SWAY_AMP", "0.05"))
        sway_hz = float(os.environ.get("NEURO_STAND_SWAY_HZ", "0.3"))
        knee_bias = float(os.environ.get("NEURO_STAND_KNEE_BIAS", "-0.1"))
        ankle_bias = float(os.environ.get("NEURO_STAND_ANKLE_BIAS", "0.05"))
        step_hip_amp = float(os.environ.get("NEURO_STAND_STEP_HIP_AMP", "0.5"))
        step_knee_amp = float(os.environ.get("NEURO_STAND_STEP_KNEE_AMP", "0.6"))
        return HumanoidStandStepGenerator(
            stand_a_seconds=stand_a_s,
            step_seconds=step_s,
            stand_b_seconds=stand_b_s,
            sway_amp=sway_amp,
            sway_hz=sway_hz,
            knee_bias=knee_bias,
            ankle_bias=ankle_bias,
            step_hip_amp=step_hip_amp,
            step_knee_amp=step_knee_amp,
            dt_seconds=dt,
        )
    if name in ("cpg_humanoid_getup", "humanoid_getup", "getup"):
        # Pairing check: the getup ghost trajectory walks the body
        # through supine waypoints (curl knees, prop on elbow, kneel).
        # If applied while NEURO_TASK_PIN points anywhere other than
        # `getup`, the trajectory actively works against the active
        # task (e.g. tells the brain to flex knees while
        # SupportedStandTask spawns the body standing). The mismatch
        # was hit live on 2026-06-23 -- warn loudly at startup so
        # future operator slips are caught.
        _task_pin = os.environ.get("NEURO_TASK_PIN", "").lower().strip()
        if _task_pin and _task_pin != "getup":
            logger.warning(
                "Demonstrator <-> task pairing mismatch: "
                "NEURO_DEMONSTRATION_TYPE=%s with NEURO_TASK_PIN=%s. "
                "The getup ghost trajectory will fight the active "
                "task. Flip both together in deploy/docker-compose.1m.yml.",
                name_raw,
                _task_pin,
            )
        amp = float(os.environ.get("NEURO_DEMONSTRATION_AMPLITUDE", "1.0"))
        total_s = float(os.environ.get("NEURO_GETUP_DURATION_S", str(_GETUP_TOTAL_S_DEFAULT)))
        return HumanoidGetupGenerator(
            total_duration_s=total_s,
            dt_seconds=dt,
            amplitude_scale=amp,
        )
    if name in ("video", "video_pose"):
        if not video_id:
            raise ValueError(
                "video provider requires video_id (factory arg or "
                "NEURO_DEMONSTRATION_VIDEO_ID env)"
            )
        try:
            trajectory = _load_video_trajectory(video_id)
        except FileNotFoundError as e:
            logger.warning("Cannot load video demonstration: %s", e)
            return None
        return VideoPoseProvider(video_id=video_id, trajectory=trajectory, dt_seconds=dt)
    # Shorthand: "video:<id>" syntax
    if name.startswith("video:"):
        return make_demonstration_provider("video", video_id=name_raw.split(":", 1)[1])
    raise ValueError(f"Unknown demonstration provider type: {name!r}")
