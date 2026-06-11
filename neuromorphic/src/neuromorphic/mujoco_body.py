"""MuJoCo virtual body for motor feedback simulation.

Supports any MJCF model (humanoid, quadruped, robotic arm, drone) via
the ``model_xml`` and ``channel_actuators`` constructor parameters.
Default model is a 29-DOF humanoid based on industry-standard configs
(Tesla Optimus / Unitree G1 / Atlas class).

Joint-to-channel mapping, body-to-channel routing, proprioceptive vector
size, and fall thresholds are all derived dynamically from the loaded
model and channel_actuators dict. No hardcoded body assumptions.

The body persists physics state between commands, moving through a
continuous simulation rather than isolated snapshots.

Requires: ``pip install mujoco`` (~24 MB, CPU-only, headless).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default MJCF — 29-DOF humanoid with realistic proportions and joint ranges
# ---------------------------------------------------------------------------
# Body proportions loosely based on a 1.72m, 70kg humanoid (Optimus-class).
# Joint ranges from human biomechanics literature, clamped to safe operating
# ranges typical of brushless rotary actuators with harmonic drives.
# ---------------------------------------------------------------------------
_HUMANOID_MJCF = """\
<mujoco model="engram_humanoid_v2">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <default>
    <joint damping="5" armature="0.02"/>
    <geom rgba="0.3 0.6 0.9 1" friction="1.0 0.005 0.0001"/>
    <motor ctrllimited="true"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="track" mode="trackcom" pos="0 -2.5 1.5" xyaxes="1 0 0 0 0.5 1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- ===== Pelvis (root) ===== -->
    <body name="pelvis" pos="0 0 0.92">
      <freejoint name="root"/>
      <geom type="box" size="0.12 0.08 0.06" mass="6" rgba="0.25 0.25 0.28 1"/>

      <!-- ===== Waist / lower torso ===== -->
      <body name="lower_torso" pos="0 0 0.08">
        <joint name="waist_yaw"   type="hinge" axis="0 0 1" range="-75 75" damping="8"/>
        <joint name="waist_roll"  type="hinge" axis="1 0 0" range="-30 30" damping="8"/>
        <joint name="waist_pitch" type="hinge" axis="0 1 0" range="-30 45" damping="8"/>
        <geom type="box" size="0.11 0.07 0.1" mass="8" rgba="0.25 0.25 0.28 1"/>

        <!-- ===== Upper torso / chest ===== -->
        <body name="torso" pos="0 0 0.18">
          <geom type="box" size="0.14 0.08 0.12" mass="12" rgba="0.3 0.3 0.33 1"/>

          <!-- ===== Head ===== -->
          <body name="neck_base" pos="0 0 0.15">
            <joint name="neck_pitch" type="hinge" axis="0 1 0" range="-40 40" damping="2"/>
            <joint name="neck_yaw"   type="hinge" axis="0 0 1" range="-55 55" damping="2"/>
            <body name="head" pos="0 0 0.08">
              <geom type="ellipsoid" size="0.09 0.08 0.1" mass="3" rgba="0.85 0.82 0.78 1"/>
            </body>
          </body>

          <!-- ===== Right arm ===== -->
          <body name="r_shoulder_link" pos="0.18 0 0.08">
            <joint name="r_shoulder_pitch" type="hinge" axis="0 1 0"  range="-90 180"  damping="3"/>
            <joint name="r_shoulder_roll"  type="hinge" axis="1 0 0"  range="-30 150"  damping="3"/>
            <joint name="r_shoulder_yaw"   type="hinge" axis="0 0 1"  range="-90 70"   damping="3"/>
            <body name="r_upper_arm" pos="0 0 -0.02">
              <geom type="capsule" size="0.035 0.13" mass="1.5" pos="0 0 -0.13"/>
              <body name="r_forearm" pos="0 0 -0.28">
                <joint name="r_elbow" type="hinge" axis="0 1 0" range="0 145" damping="2"/>
                <geom type="capsule" size="0.03 0.12" mass="1.0" pos="0 0 -0.12"/>
                <body name="r_hand" pos="0 0 -0.26">
                  <joint name="r_wrist_pitch" type="hinge" axis="0 1 0" range="-70 70" damping="1"/>
                  <joint name="r_wrist_yaw"   type="hinge" axis="0 0 1" range="-45 45" damping="1"/>
                  <geom type="box" size="0.04 0.025 0.06" mass="0.4" rgba="0.25 0.25 0.28 1"/>
                </body>
              </body>
            </body>
          </body>

          <!-- ===== Left arm ===== -->
          <body name="l_shoulder_link" pos="-0.18 0 0.08">
            <joint name="l_shoulder_pitch" type="hinge" axis="0 1 0"  range="-90 180"  damping="3"/>
            <joint name="l_shoulder_roll"  type="hinge" axis="1 0 0"  range="-150 30"  damping="3"/>
            <joint name="l_shoulder_yaw"   type="hinge" axis="0 0 1"  range="-70 90"   damping="3"/>
            <body name="l_upper_arm" pos="0 0 -0.02">
              <geom type="capsule" size="0.035 0.13" mass="1.5" pos="0 0 -0.13"/>
              <body name="l_forearm" pos="0 0 -0.28">
                <joint name="l_elbow" type="hinge" axis="0 1 0" range="0 145" damping="2"/>
                <geom type="capsule" size="0.03 0.12" mass="1.0" pos="0 0 -0.12"/>
                <body name="l_hand" pos="0 0 -0.26">
                  <joint name="l_wrist_pitch" type="hinge" axis="0 1 0" range="-70 70" damping="1"/>
                  <joint name="l_wrist_yaw"   type="hinge" axis="0 0 1" range="-45 45" damping="1"/>
                  <geom type="box" size="0.04 0.025 0.06" mass="0.4" rgba="0.25 0.25 0.28 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <!-- ===== Right leg ===== -->
      <body name="r_hip_link" pos="0.09 0 -0.06">
        <joint name="r_hip_yaw"   type="hinge" axis="0 0 1"  range="-40 40"   damping="6"/>
        <joint name="r_hip_roll"  type="hinge" axis="1 0 0"  range="-25 45"   damping="6"/>
        <joint name="r_hip_pitch" type="hinge" axis="0 1 0"  range="-30 120"  damping="6"/>
        <body name="r_thigh" pos="0 0 -0.02">
          <geom type="capsule" size="0.05 0.18" mass="4" pos="0 0 -0.18"/>
          <body name="r_shin" pos="0 0 -0.38">
            <joint name="r_knee" type="hinge" axis="0 1 0" range="-145 0" damping="5"/>
            <geom type="capsule" size="0.04 0.17" mass="2.5" pos="0 0 -0.17"/>
            <body name="r_foot" pos="0 0 -0.36">
              <joint name="r_ankle_pitch" type="hinge" axis="0 1 0" range="-40 50" damping="3"/>
              <joint name="r_ankle_roll"  type="hinge" axis="1 0 0" range="-25 25" damping="3"/>
              <geom type="box" size="0.06 0.04 0.015" mass="0.8"
                    pos="0.02 0 -0.015" rgba="0.25 0.25 0.28 1"/>
            </body>
          </body>
        </body>
      </body>

      <!-- ===== Left leg ===== -->
      <body name="l_hip_link" pos="-0.09 0 -0.06">
        <joint name="l_hip_yaw"   type="hinge" axis="0 0 1"  range="-40 40"   damping="6"/>
        <joint name="l_hip_roll"  type="hinge" axis="1 0 0"  range="-45 25"   damping="6"/>
        <joint name="l_hip_pitch" type="hinge" axis="0 1 0"  range="-30 120"  damping="6"/>
        <body name="l_thigh" pos="0 0 -0.02">
          <geom type="capsule" size="0.05 0.18" mass="4" pos="0 0 -0.18"/>
          <body name="l_shin" pos="0 0 -0.38">
            <joint name="l_knee" type="hinge" axis="0 1 0" range="-145 0" damping="5"/>
            <geom type="capsule" size="0.04 0.17" mass="2.5" pos="0 0 -0.17"/>
            <body name="l_foot" pos="0 0 -0.36">
              <joint name="l_ankle_pitch" type="hinge" axis="0 1 0" range="-40 50" damping="3"/>
              <joint name="l_ankle_roll"  type="hinge" axis="1 0 0" range="-25 25" damping="3"/>
              <geom type="box" size="0.06 0.04 0.015" mass="0.8"
                    pos="0.02 0 -0.015" rgba="0.25 0.25 0.28 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- Waist (torso articulation) — 3 DOF -->
    <motor name="waist_yaw_m"   joint="waist_yaw"   ctrlrange="-120 120"/>
    <motor name="waist_roll_m"  joint="waist_roll"   ctrlrange="-100 100"/>
    <motor name="waist_pitch_m" joint="waist_pitch"  ctrlrange="-100 100"/>
    <!-- Head — 2 DOF -->
    <motor name="neck_pitch_m"  joint="neck_pitch"   ctrlrange="-20 20"/>
    <motor name="neck_yaw_m"    joint="neck_yaw"     ctrlrange="-20 20"/>
    <!-- Right arm — 6 DOF -->
    <motor name="r_shoulder_pitch_m" joint="r_shoulder_pitch" ctrlrange="-80 80"/>
    <motor name="r_shoulder_roll_m"  joint="r_shoulder_roll"  ctrlrange="-80 80"/>
    <motor name="r_shoulder_yaw_m"   joint="r_shoulder_yaw"   ctrlrange="-50 50"/>
    <motor name="r_elbow_m"          joint="r_elbow"          ctrlrange="-50 50"/>
    <motor name="r_wrist_pitch_m"    joint="r_wrist_pitch"    ctrlrange="-20 20"/>
    <motor name="r_wrist_yaw_m"      joint="r_wrist_yaw"      ctrlrange="-20 20"/>
    <!-- Left arm — 6 DOF -->
    <motor name="l_shoulder_pitch_m" joint="l_shoulder_pitch" ctrlrange="-80 80"/>
    <motor name="l_shoulder_roll_m"  joint="l_shoulder_roll"  ctrlrange="-80 80"/>
    <motor name="l_shoulder_yaw_m"   joint="l_shoulder_yaw"   ctrlrange="-50 50"/>
    <motor name="l_elbow_m"          joint="l_elbow"          ctrlrange="-50 50"/>
    <motor name="l_wrist_pitch_m"    joint="l_wrist_pitch"    ctrlrange="-20 20"/>
    <motor name="l_wrist_yaw_m"      joint="l_wrist_yaw"      ctrlrange="-20 20"/>
    <!-- Right leg — 6 DOF -->
    <motor name="r_hip_yaw_m"     joint="r_hip_yaw"     ctrlrange="-80 80"/>
    <motor name="r_hip_roll_m"    joint="r_hip_roll"    ctrlrange="-80 80"/>
    <motor name="r_hip_pitch_m"   joint="r_hip_pitch"   ctrlrange="-100 100"/>
    <motor name="r_knee_m"        joint="r_knee"        ctrlrange="-80 80"/>
    <motor name="r_ankle_pitch_m" joint="r_ankle_pitch" ctrlrange="-50 50"/>
    <motor name="r_ankle_roll_m"  joint="r_ankle_roll"  ctrlrange="-30 30"/>
    <!-- Left leg — 6 DOF -->
    <motor name="l_hip_yaw_m"     joint="l_hip_yaw"     ctrlrange="-80 80"/>
    <motor name="l_hip_roll_m"    joint="l_hip_roll"    ctrlrange="-80 80"/>
    <motor name="l_hip_pitch_m"   joint="l_hip_pitch"   ctrlrange="-100 100"/>
    <motor name="l_knee_m"        joint="l_knee"        ctrlrange="-80 80"/>
    <motor name="l_ankle_pitch_m" joint="l_ankle_pitch" ctrlrange="-50 50"/>
    <motor name="l_ankle_roll_m"  joint="l_ankle_roll"  ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""

# ---------------------------------------------------------------------------
# Channel → actuator mapping (default humanoid)
# ---------------------------------------------------------------------------
# Humanoid example:
#   Locomotion = legs (12 DOF) + waist (3 DOF) = 15 actuators
#   Manipulation = arms (12 DOF) = 12 actuators
#   Head = neck (2 DOF) = 2 actuators
#   Total (humanoid): 29 actuators
# For other body types, override channel_actuators in MotorFeedbackConfig.
# ---------------------------------------------------------------------------
# Default 29-DOF humanoid mapping. Kept as module-level constant for backward
# compatibility (tests and legacy code import this).  At runtime, MuJoCoBody
# and PopulationVectorDecoder read from MotorFeedbackConfig.channel_actuators
# so any body type (quadruped, arm, drone) can override without touching brain code.
CHANNEL_ACTUATORS: dict[str, list[str]] = {
    "locomotion": [
        "waist_yaw_m", "waist_roll_m", "waist_pitch_m",
        "r_hip_yaw_m", "r_hip_roll_m", "r_hip_pitch_m",
        "r_knee_m", "r_ankle_pitch_m", "r_ankle_roll_m",
        "l_hip_yaw_m", "l_hip_roll_m", "l_hip_pitch_m",
        "l_knee_m", "l_ankle_pitch_m", "l_ankle_roll_m",
    ],
    "manipulation": [
        "r_shoulder_pitch_m", "r_shoulder_roll_m", "r_shoulder_yaw_m",
        "r_elbow_m", "r_wrist_pitch_m", "r_wrist_yaw_m",
        "l_shoulder_pitch_m", "l_shoulder_roll_m", "l_shoulder_yaw_m",
        "l_elbow_m", "l_wrist_pitch_m", "l_wrist_yaw_m",
    ],
    "head": ["neck_pitch_m", "neck_yaw_m"],
}

# Channels that don't have physical simulation
STOCHASTIC_CHANNELS = {"expression", "speech", "cognitive"}


class MuJoCoBody:
    """Simulated body using MuJoCo physics (any MJCF model supported).

    Default is a 29-DOF humanoid based on industry-standard configurations
    (Tesla Optimus, Unitree G1, Boston Dynamics Atlas class). Pass custom
    ``model_xml`` and ``channel_actuators`` for other body types.

    Maps brain motor channels to actuator groups and returns
    proprioceptive state after physics simulation.  Physics state
    persists between calls. Joint mapping, body routing, and fall
    thresholds are derived dynamically from the loaded model.
    """

    def __init__(
        self,
        model_xml: Optional[str] = None,
        steps_per_command: int = 500,
        channel_actuators: Optional[dict[str, list[str]]] = None,
    ):
        try:
            import mujoco
        except ImportError:
            raise ImportError(
                "mujoco is required for virtual body simulation. "
                "Install with: pip install mujoco"
            )

        self._mujoco = mujoco
        xml = model_xml or _HUMANOID_MJCF
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        self._steps_per_command = steps_per_command

        # Body-specific channel-to-actuator mapping. Falls back to the
        # module-level humanoid default when not provided.
        self._channel_actuators = channel_actuators or CHANNEL_ACTUATORS

        # Build actuator name -> index mapping
        self._actuator_idx: dict[str, int] = {}
        for i in range(self._model.nu):
            name = self._model.actuator(i).name
            self._actuator_idx[name] = i

        # Build joint_name -> channel mapping dynamically from channel_actuators.
        # Each actuator in MuJoCo links to a joint via actuator_trnid.
        self._joint_channel: dict[str, str] = {}
        for channel, act_names in self._channel_actuators.items():
            for act_name in act_names:
                act_idx = self._actuator_idx.get(act_name)
                if act_idx is not None:
                    jnt_id = self._model.actuator_trnid[act_idx, 0]
                    jnt_name = self._model.joint(jnt_id).name
                    self._joint_channel[jnt_name] = channel

        # Build body_name -> channel mapping for force routing.
        # Walk each body's parent chain to find which joint (and thus channel)
        # it belongs to.
        self._body_channel: dict[str, str] = {}
        joint_body_ids: dict[int, str] = {}  # body_id -> channel
        for jname, ch in self._joint_channel.items():
            jnt_id = self._model.joint(jname).id
            body_id = self._model.jnt_bodyid[jnt_id]
            joint_body_ids[body_id] = ch
        # Propagate: for each body, walk up the parent chain until we find
        # a body that owns a channeled joint.
        for i in range(self._model.nbody):
            bname = self._model.body(i).name
            bid = i
            while bid >= 0:
                if bid in joint_body_ids:
                    self._body_channel[bname] = joint_body_ids[bid]
                    break
                parent = self._model.body_parentid[bid]
                if parent == bid:
                    break
                bid = parent

        # Identify the "root body" for height tracking. Prefer "torso", fall
        # back to the heaviest non-world body.
        self._root_body_name = self._find_root_body()

        # Lazy-initialized renderer (requires OSMesa for headless CPU rendering)
        self._renderer: Any = None

        # Step once to settle the body, then record initial root body height
        self._mujoco.mj_step(self._model, self._data)
        self._initial_root_z = float(
            self._data.body(self._root_body_name).xpos[2]
        )
        if self._initial_root_z <= 0.0:
            self._initial_root_z = 1.2  # fallback to nominal height

        logger.info(
            "MuJoCoBody initialized: %d joints, %d actuators, %d bodies, dt=%.3fs",
            self._model.njnt, self._model.nu, self._model.nbody,
            self._model.opt.timestep,
        )

    def _find_root_body(self) -> str:
        """Find the primary reference body for height tracking.

        Prefers "torso" if it exists. Otherwise picks the heaviest
        non-world body (excluding floor/ground plane bodies).
        """
        # Check for common names
        for candidate in ("torso", "upper_torso", "chest", "base_link", "body"):
            try:
                self._model.body(candidate)
                return candidate
            except Exception:
                continue
        # Fallback: heaviest non-world body
        best_name = "world"
        best_mass = -1.0
        for i in range(1, self._model.nbody):  # skip world (0)
            name = self._model.body(i).name
            mass = float(self._model.body_mass[i])
            if mass > best_mass:
                best_mass = mass
                best_name = name
        return best_name

    def step_command(self, channel: str, intensity: float) -> dict[str, Any]:
        """Apply motor command and simulate physics.

        Args:
            channel: "locomotion", "manipulation", or "head"
            intensity: 0.0-1.0 from brain's population vector

        Returns:
            motor.outcome-compatible dict with success, confidence,
            proprioceptive_state, and error_magnitude.
        """
        intensity = float(np.clip(intensity, 0.0, 1.0))

        if channel in STOCHASTIC_CHANNELS:
            return self._stochastic_outcome(channel, intensity)

        actuator_names = self._channel_actuators.get(channel)
        if not actuator_names:
            logger.warning("Unknown motor channel for MuJoCo: %s", channel)
            return self._stochastic_outcome(channel, intensity)

        # Snapshot pre-command state for success evaluation
        pre_torso_z = float(self._data.body(self._root_body_name).xpos[2])
        pre_joint_pos = self._get_channel_joint_positions(channel).copy()

        # Map intensity (0-1) to torque: 0 firing = 0 torque (relaxed).
        # Positive firing produces positive torque. Gravity provides the
        # opposing force -- like real muscles, the brain must fire to resist.
        intensity = float(np.clip(intensity, 0.0, 1.0))
        for name in actuator_names:
            idx = self._actuator_idx.get(name)
            if idx is not None:
                ctrl_range = self._model.actuator_ctrlrange[idx]
                max_torque = ctrl_range[1]
                self._data.ctrl[idx] = intensity * max_torque

        # Step physics
        for _ in range(self._steps_per_command):
            self._mujoco.mj_step(self._model, self._data)

        # Read post-command state
        post_torso_z = float(self._data.body(self._root_body_name).xpos[2])
        post_joint_pos = self._get_channel_joint_positions(channel)

        # Evaluate success
        success, confidence, error_mag = self._evaluate_outcome(
            channel, pre_torso_z, post_torso_z, pre_joint_pos, post_joint_pos,
        )

        # Build proprioceptive state: joint positions + velocities for this channel
        proprio = self._get_channel_proprioception(channel)

        return {
            "channel": channel,
            "success": success,
            "confidence": confidence,
            "proprioceptive_state": proprio,
            "error_magnitude": error_mag,
        }

    def get_full_state(self) -> dict[str, Any]:
        """Return complete body state for monitoring and visualization."""
        bodies = []
        for i in range(self._model.nbody):
            bodies.append({
                "name": self._model.body(i).name,
                "xpos": self._data.body(i).xpos.tolist(),
                "xquat": self._data.body(i).xquat.tolist(),
            })
        return {
            "torso_height": float(self._data.body(self._root_body_name).xpos[2]),
            "torso_pos": self._data.body(self._root_body_name).xpos.tolist(),
            "joint_positions": self._data.qpos.tolist(),
            "joint_velocities": self._data.qvel.tolist(),
            "num_contacts": int(self._data.ncon),
            "sim_time": float(self._data.time),
            "bodies": bodies,
        }

    def get_model_geometry(self) -> list[dict[str, Any]]:
        """Return static geometry definitions for visualization (call once).

        Each entry describes a body's shape so a 3D viewer can build
        matching meshes without parsing MJCF.
        """
        geoms = []
        for i in range(self._model.ngeom):
            g = self._model.geom(i)
            body_id = g.bodyid
            body_name = self._model.body(body_id).name if body_id >= 0 else "world"
            geom_type = int(g.type)  # 0=plane, 2=sphere, 3=ellipsoid, 5=capsule, 6=box
            type_names = {0: "plane", 2: "sphere", 3: "ellipsoid", 5: "capsule", 6: "box"}
            geoms.append({
                "body": body_name,
                "type": type_names.get(geom_type, f"unknown_{geom_type}"),
                "size": g.size.tolist(),
                "rgba": g.rgba.tolist(),
            })
        return geoms

    # ── Guided teaching (human-in-the-loop) ───────────────────────

    @property
    def JOINT_CHANNEL(self) -> dict[str, str]:
        """Joint name -> channel mapping (built dynamically from channel_actuators)."""
        return self._joint_channel

    def guide_to_pose(
        self, joint_angles: dict[str, float], settle_steps: int = 100,
    ) -> dict[str, Any]:
        """Kinematically guide body to a target pose.

        Uses PD-controlled actuators to drive joints toward target angles
        while keeping the pelvis upright (like a parent supporting a child).

        Args:
            joint_angles: Joint name -> target angle (degrees).
                See JOINT_CHANNEL property for valid names.
            settle_steps: Physics steps to hold the pose.

        Returns:
            motor.outcome-compatible dict (success=True, high confidence).
        """
        channels_touched: set[str] = set()

        # Convert angles to radians and clamp to joint limits
        target_rad: dict[str, float] = {}
        for joint_name, angle in joint_angles.items():
            if joint_name not in self._joint_channel:
                logger.warning("Unknown joint name in guide_to_pose: %s", joint_name)
                continue
            jnt_id = self._model.joint(joint_name).id
            jnt_range = self._model.jnt_range[jnt_id]
            angle_rad = float(np.clip(
                np.radians(angle) if abs(angle) > np.pi else angle,
                jnt_range[0], jnt_range[1],
            ))
            target_rad[joint_name] = angle_rad
            # Only count non-zero angles for channel routing
            if abs(angle) > 0.5:
                ch = self._joint_channel.get(joint_name)
                if ch:
                    channels_touched.add(ch)

        # Reset pelvis to standing before applying pose
        # (freejoint qpos: x,y,z, qw,qx,qy,qz)
        try:
            root_adr = self._model.jnt_qposadr[self._model.joint("root").id]
            self._data.qpos[root_adr:root_adr + 3] = [0.0, 0.0, self._initial_root_z]
            self._data.qpos[root_adr + 3:root_adr + 7] = [1, 0, 0, 0]    # upright quat
        except Exception:
            pass  # no freejoint (fixed-base robot)

        # Set target joint angles directly
        for joint_name, angle_rad in target_rad.items():
            jnt_id = self._model.joint(joint_name).id
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            self._data.qpos[qpos_adr] = angle_rad

        # Zero all velocities for clean start
        self._data.qvel[:] = 0.0

        # Build actuator name → index map
        act_map: dict[str, int] = {}
        for i in range(self._model.nu):
            act_map[self._model.actuator(i).name] = i

        # Use actuators as PD controllers to HOLD the pose during settle.
        kp = 40.0  # stiff enough to hold against gravity
        _min_height = 0.6 * self._initial_root_z
        for _ in range(settle_steps):
            # Hold pelvis upright: re-pin root position each step
            try:
                self._data.qpos[root_adr + 2] = max(
                    self._data.qpos[root_adr + 2], _min_height
                )
                self._data.qvel[0:6] = 0.0  # zero root velocities
            except Exception:
                pass  # no freejoint

            # PD control on each actuated joint
            for joint_name, target in target_rad.items():
                act_name = joint_name + "_m"
                act_idx = act_map.get(act_name)
                if act_idx is None:
                    continue
                jnt_id = self._model.joint(joint_name).id
                qpos_adr = self._model.jnt_qposadr[jnt_id]
                current = self._data.qpos[qpos_adr]
                error = target - current
                ctrl = np.clip(kp * error,
                               self._model.actuator_ctrlrange[act_idx][0],
                               self._model.actuator_ctrlrange[act_idx][1])
                self._data.ctrl[act_idx] = ctrl

            self._mujoco.mj_step(self._model, self._data)

        # Clear actuator commands after settling
        self._data.ctrl[:] = 0.0

        # Determine primary channel for feedback routing
        channel = "locomotion"
        if "manipulation" in channels_touched:
            channel = "manipulation"
        elif "head" in channels_touched:
            channel = "head"
        if len(channels_touched) == 1:
            channel = next(iter(channels_touched))

        proprio = self._get_channel_proprioception(channel)

        return {
            "channel": channel,
            "success": True,  # guided = always success (this is the target)
            "confidence": 0.95,
            "proprioceptive_state": proprio,
            "error_magnitude": 0.0,
            "guided": True,
        }

    def apply_force(
        self,
        body_name: str,
        force: tuple[float, float, float],
        steps: int = 150,
    ) -> dict[str, Any]:
        """Apply external force to a body part (perturbation / balance test).

        Resets to standing first, then applies the force.
        Returns the proprioceptive outcome — if the body stays upright,
        it's a success (learned to resist/compensate).

        Args:
            body_name: MuJoCo body name (torso, head, r_upper_arm, etc.)
            force: (fx, fy, fz) force vector in Newtons.
            steps: How long to apply the force.

        Returns:
            motor.outcome-compatible dict.
        """
        # Reset to standing so the force test starts from a known state
        self.reset()

        body_id = self._model.body(body_name).id
        root_id = self._model.body(self._root_body_name).id

        # PD support: compensate gravity on Z-axis while allowing natural
        # horizontal reaction to the push (no artificial velocity damping).
        total_mass = float(self._model.body_subtreemass[0])
        gravity_comp = total_mass * 9.81
        kp = gravity_comp * 5.0
        kd = gravity_comp * 1.0

        # Apply push force
        self._data.xfrc_applied[body_id, :3] = force
        for _ in range(steps):
            # PD support on vertical axis only
            root_z = float(self._data.body(self._root_body_name).xpos[2])
            root_vz = float(self._data.body(self._root_body_name).cvel[5])
            error = self._initial_root_z - root_z
            support = max(0.0, gravity_comp + kp * error - kd * root_vz)
            # Combine push Z + support Z if same body, else set independently
            if body_id == root_id:
                self._data.xfrc_applied[root_id, 2] = force[2] + support
            else:
                self._data.xfrc_applied[root_id, 2] = support
            self._mujoco.mj_step(self._model, self._data)

        # Remove push force and let settle with PD support + horizontal centering
        self._data.xfrc_applied[body_id, :3] = 0.0
        kp_h = gravity_comp * 0.3
        kd_h = gravity_comp * 0.2
        for _ in range(50):
            root_pos = self._data.body(self._root_body_name).xpos
            root_vel = self._data.body(self._root_body_name).cvel
            root_z = float(root_pos[2])
            error = self._initial_root_z - root_z
            support = max(0.0, gravity_comp + kp * error - kd * float(root_vel[5]))
            self._data.xfrc_applied[root_id, 0] = -kp_h * float(root_pos[0]) - kd_h * float(root_vel[3])
            self._data.xfrc_applied[root_id, 1] = -kp_h * float(root_pos[1]) - kd_h * float(root_vel[4])
            self._data.xfrc_applied[root_id, 2] = support
            self._mujoco.mj_step(self._model, self._data)

        post_z = float(self._data.body(self._root_body_name).xpos[2])
        fall_threshold = 0.33 * self._initial_root_z
        height_ok = post_z > fall_threshold

        # Determine channel from body name using dynamic body->channel map
        channel = self._body_channel.get(body_name, "locomotion")

        proprio = self._get_channel_proprioception(channel)

        return {
            "channel": channel,
            "success": height_ok,
            "confidence": 0.85,
            "proprioceptive_state": proprio,
            "error_magnitude": float(np.clip(1.0 - post_z / self._initial_root_z, 0, 1)),
            "guided": True,
        }

    def get_joint_info(self) -> list[dict[str, Any]]:
        """Return joint names, ranges, and current positions for the UI."""
        joints = []
        for jname, channel in self._joint_channel.items():
            jnt_id = self._model.joint(jname).id
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            jnt_range = self._model.jnt_range[jnt_id]
            joints.append({
                "name": jname,
                "channel": channel,
                "angle_rad": float(self._data.qpos[qpos_adr]),
                "range_rad": [float(jnt_range[0]), float(jnt_range[1])],
                "range_deg": [
                    int(round(float(np.degrees(jnt_range[0])))),
                    int(round(float(np.degrees(jnt_range[1])))),
                ],
            })
        return joints

    def is_fallen(self) -> bool:
        """Detect if the body has fallen using both height AND orientation.

        Height alone is insufficient -- a body lying on its side can have
        the torso at 70-80% of initial height if propped on an arm/shoulder.
        We check the pelvis/root quaternion's "up" component: quat [w,x,y,z]
        maps world-Z onto the body.  For an upright body the local Z axis
        aligns with world Z, giving cos(tilt) ~ 1.0.  When tilted >60 deg,
        the body is effectively fallen.
        """
        root_z = float(self._data.body(self._root_body_name).xpos[2])
        # Classic height check
        if root_z < 0.33 * self._initial_root_z:
            return True
        # Orientation check: extract tilt angle from quaternion.
        # For quat [w, x, y, z], the Z-component of the rotated up-vector is:
        #   up_z = 1 - 2*(x^2 + y^2)  (equivalent to rotation matrix R[2][2])
        # up_z = 1.0 means upright, 0.0 means 90-deg tilt, -1.0 means inverted.
        quat = self._data.body(self._root_body_name).xquat
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        up_z = 1.0 - 2.0 * (x * x + y * y)
        # Fallen if tilted more than ~60 degrees (up_z < 0.5 = cos(60))
        if up_z < 0.5:
            return True
        return False

    def reset(self) -> None:
        """Reset body to standing pose."""
        self._mujoco.mj_resetData(self._model, self._data)
        self._mujoco.mj_forward(self._model, self._data)
        z = float(self._data.body(self._root_body_name).xpos[2])
        self._initial_root_z = z if z > 0.0 else 1.2

    # ── Continuous physics API ────────────────────────────────────

    def set_control(self, channel: str, intensity: float) -> None:
        """Set actuator control vector for a channel without stepping physics.

        This allows decoupling control updates from physics stepping, which
        is required for continuous physics mode where mj_step runs at a
        fixed rate and motor commands arrive asynchronously.

        Args:
            channel: "locomotion", "manipulation", or "head"
            intensity: 0.0-1.0 from brain's population vector
        """
        if channel in STOCHASTIC_CHANNELS:
            return
        actuator_names = self._channel_actuators.get(channel)
        if not actuator_names:
            return
        intensity = float(np.clip(intensity, 0.0, 1.0))
        for name in actuator_names:
            idx = self._actuator_idx.get(name)
            if idx is not None:
                ctrl_range = self._model.actuator_ctrlrange[idx]
                max_torque = ctrl_range[1]
                self._data.ctrl[idx] = intensity * max_torque

    def set_control_vector(
        self, channel: str, actuator_intensities: dict[str, float],
    ) -> None:
        """Set per-actuator controls from population vector decoding.

        Each actuator gets its own intensity derived from a dedicated neuron
        group, enabling individuated joint control (e.g., extend left knee
        while flexing right hip).

        Actuators missing from *actuator_intensities* are set to 0
        (zero torque / relaxed) to avoid stale controls.

        Args:
            channel: "locomotion", "manipulation", or "head"
            actuator_intensities: {actuator_name: intensity} where intensity
                is 0.0-1.0 (0.0 = relaxed/zero torque, 1.0 = max torque).
        """
        if channel in STOCHASTIC_CHANNELS:
            return
        actuator_names = self._channel_actuators.get(channel)
        if not actuator_names:
            return
        for name in actuator_names:
            idx = self._actuator_idx.get(name)
            if idx is None:
                continue
            # Default to 0.0 for missing actuators — relaxed (zero torque)
            intensity = actuator_intensities.get(name, 0.0)
            intensity = float(np.clip(intensity, 0.0, 1.0))
            # Guard: np.clip(nan) returns nan — fall back to relaxed
            if not np.isfinite(intensity):
                intensity = 0.0
            # 0 firing = 0 torque (relaxed). Brain fires to produce force.
            ctrl_range = self._model.actuator_ctrlrange[idx]
            max_torque = ctrl_range[1]
            self._data.ctrl[idx] = intensity * max_torque

    def step_batch(self, n: int = 1) -> None:
        """Step MuJoCo physics N times (no outcome evaluation).

        Used by the continuous physics loop to advance the simulation at a
        fixed rate, independent of motor commands.

        Args:
            n: Number of mj_step calls.
        """
        for _ in range(n):
            self._mujoco.mj_step(self._model, self._data)

    def render_frame(self, width: int = 64, height: int = 64) -> Optional[np.ndarray]:
        """Render a camera frame from the 'track' camera.

        Returns 64x64 grayscale uint8 array, or None if rendering fails.
        Requires OSMesa (``MUJOCO_GL=osmesa``, ``apt install libosmesa6-dev``).
        """
        if self._renderer is None:
            try:
                self._renderer = self._mujoco.Renderer(self._model, height=height, width=width)
            except Exception as e:
                logger.warning("MuJoCo renderer init failed (OSMesa missing?): %s", e)
                self._renderer = False  # sentinel: don't retry
                return None
        if self._renderer is False:
            return None
        try:
            self._renderer.update_scene(self._data, camera="track")
            rgb = self._renderer.render()  # (H, W, 3) uint8
            # Convert to grayscale: standard luminance weights
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            return gray
        except Exception as e:
            logger.debug("MuJoCo render failed: %s", e)
            return None

    @property
    def proprioceptive_size(self) -> int:
        """Size of the proprioceptive vector (depends on number of joints)."""
        n_joints = len(self._joint_channel)
        # pos + vel + quat + height + contacts + com_xy(2) + angular_vel(3) + foot_forces(2)
        return n_joints * 2 + 4 + 1 + 1 + 2 + 3 + 2

    def compute_standing_torques(self, kp: float = 10.0) -> dict[str, float]:
        """Compute per-actuator intensities for maintaining an upright standing pose.

        Uses a simple PD controller: each joint's target is the neutral
        position (midpoint of joint range).  Returns intensities in [0, 1]
        suitable for ``set_control_vector`` or ``inject_standing_pattern``.

        Args:
            kp: Proportional gain for PD controller.

        Returns:
            {actuator_name: intensity} where intensity is 0-1.
        """
        intensities: dict[str, float] = {}
        for act_name, act_idx in self._actuator_idx.items():
            # Find the joint this actuator controls
            jnt_id = self._model.actuator_trnid[act_idx, 0]
            jnt_range = self._model.jnt_range[jnt_id]
            # Target: midpoint of joint range (neutral standing pose)
            target = (jnt_range[0] + jnt_range[1]) * 0.5
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            current = float(self._data.qpos[qpos_adr])
            error = target - current
            # Compute control signal, normalize to [0, 1] via ctrl range
            ctrl_range = self._model.actuator_ctrlrange[act_idx]
            max_torque = max(abs(ctrl_range[0]), abs(ctrl_range[1]), 1e-6)
            raw_ctrl = kp * error
            intensity = float(np.clip((raw_ctrl / max_torque + 1.0) * 0.5, 0.0, 1.0))
            intensities[act_name] = intensity
        return intensities

    def get_proprioceptive_vector(self) -> np.ndarray:
        """Return a compact proprioceptive state vector for sensory injection.

        The vector contains:
          - N joint positions (normalized to [-1, 1] by joint range)
          - N joint velocities (clipped to [-10, 10], then /10)
          - Root body orientation quaternion (4 values)
          - Root body height normalized by initial height (1 value)
          - Number of ground contacts (1 value, /10 for scale)
          - Center of mass xy offset from root body (2 values, clipped [-1, 1])
          - Angular velocity xyz of root body (3 values, /5.0 for scale)
          - Per-foot contact forces left/right (2 values, 0.0 for non-humanoid)

        Total: N + N + 4 + 1 + 1 + 2 + 3 + 2 = 2N + 13 floats.
        """
        # Joint positions — normalized to [-1, 1] by joint range
        positions = []
        velocities = []
        for jname in self._joint_channel:
            jnt_id = self._model.joint(jname).id
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            qvel_adr = self._model.jnt_dofadr[jnt_id]
            jnt_range = self._model.jnt_range[jnt_id]
            pos = float(self._data.qpos[qpos_adr])
            # Normalize to [-1, 1]
            mid = (jnt_range[0] + jnt_range[1]) / 2.0
            half = (jnt_range[1] - jnt_range[0]) / 2.0
            if half > 0:
                positions.append(float(np.clip((pos - mid) / half, -1.0, 1.0)))
            else:
                positions.append(0.0)
            vel = float(self._data.qvel[qvel_adr])
            velocities.append(float(np.clip(vel / 10.0, -1.0, 1.0)))

        # Root body orientation quaternion
        root_quat = self._data.body(self._root_body_name).xquat.tolist()

        # Root body height (normalized)
        root_z = float(self._data.body(self._root_body_name).xpos[2])
        height_norm = float(np.clip(root_z / self._initial_root_z, 0.0, 2.0))

        # Ground contacts (scaled)
        contacts = float(np.clip(self._data.ncon / 10.0, 0.0, 1.0))

        # Center of mass position relative to support (xy only, normalized).
        # Critical for balance -- tells the brain if CoM is over the feet.
        subtree_com = self._data.subtree_com[0]  # world body subtree CoM
        root_xy = self._data.body(self._root_body_name).xpos[:2]
        com_offset = subtree_com[:2] - root_xy
        com_x = float(np.clip(com_offset[0], -1.0, 1.0))
        com_y = float(np.clip(com_offset[1], -1.0, 1.0))

        # Angular velocity of root body (vestibular equivalent).
        # Tells the brain how fast it's tipping -- essential for balance reflexes.
        root_body_id = self._model.body(self._root_body_name).id
        # cvel is 6D (3 angular + 3 linear) per body
        ang_vel = self._data.cvel[root_body_id, :3]  # angular velocity (rad/s)
        ang_x = float(np.clip(ang_vel[0] / 5.0, -1.0, 1.0))
        ang_y = float(np.clip(ang_vel[1] / 5.0, -1.0, 1.0))
        ang_z = float(np.clip(ang_vel[2] / 5.0, -1.0, 1.0))

        # Per-foot contact forces (left/right).
        # Tells the brain which foot is bearing weight -- needed for gait.
        left_force = 0.0
        right_force = 0.0
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            body1 = self._model.body(self._model.geom(c.geom1).bodyid.item()).name
            body2 = self._model.body(self._model.geom(c.geom2).bodyid.item()).name
            # Sum normal forces for left/right foot geoms
            force = float(np.abs(c.dist))  # penetration depth as proxy
            for bname in (body1, body2):
                if "l_foot" in bname or "l_ankle" in bname:
                    left_force += force
                elif "r_foot" in bname or "r_ankle" in bname:
                    right_force += force
        left_force = float(np.clip(left_force * 100.0, 0.0, 1.0))
        right_force = float(np.clip(right_force * 100.0, 0.0, 1.0))

        vec = np.array(
            positions + velocities + root_quat + [height_norm, contacts,
            com_x, com_y, ang_x, ang_y, ang_z, left_force, right_force],
            dtype=np.float32,
        )
        return vec

    def compute_pain_vector(self, limit_zone: float = 0.2) -> np.ndarray:
        """Compute per-joint pain signal from proximity to joint limits.

        Returns float32 array of length N (one per actuated joint), values in [0, 1]:
          0.0 = safe (within center of range)
          1.0 = at hard limit

        Pain activates in the outer ``limit_zone`` fraction of joint range
        on each side.  E.g. zone=0.2 means the outer 20% triggers pain.
        """
        pain = []
        for jname in self._joint_channel:
            jnt_id = self._model.joint(jname).id
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            jnt_range = self._model.jnt_range[jnt_id]
            pos = float(self._data.qpos[qpos_adr])
            lo, hi = float(jnt_range[0]), float(jnt_range[1])
            half = (hi - lo) / 2.0
            if half <= 0:
                pain.append(0.0)
                continue
            # Distance from center, normalized to [0, 1]
            mid = (lo + hi) / 2.0
            dist = abs(pos - mid) / half  # 0 = center, 1 = at limit
            # Pain ramps from 0 at (1 - limit_zone) to 1 at 1.0
            threshold = 1.0 - limit_zone
            if dist <= threshold:
                pain.append(0.0)
            else:
                pain.append(min((dist - threshold) / limit_zone, 1.0))
        return np.array(pain, dtype=np.float32)

    def _get_channel_joint_positions(self, channel: str) -> np.ndarray:
        """Get joint positions for actuators in a channel."""
        positions = []
        for name in self._channel_actuators.get(channel, []):
            idx = self._actuator_idx.get(name)
            if idx is not None:
                jnt_id = self._model.actuator_trnid[idx, 0]
                qpos_adr = self._model.jnt_qposadr[jnt_id]
                positions.append(float(self._data.qpos[qpos_adr]))
        return np.array(positions, dtype=np.float32)

    def _get_channel_proprioception(self, channel: str) -> list[float]:
        """Get combined position + velocity for a channel's joints."""
        proprio: list[float] = []
        for name in self._channel_actuators.get(channel, []):
            idx = self._actuator_idx.get(name)
            if idx is not None:
                jnt_id = self._model.actuator_trnid[idx, 0]
                qpos_adr = self._model.jnt_qposadr[jnt_id]
                qvel_adr = self._model.jnt_dofadr[jnt_id]
                proprio.append(float(self._data.qpos[qpos_adr]))
                proprio.append(float(self._data.qvel[qvel_adr]))
        return proprio

    def _evaluate_outcome(
        self,
        channel: str,
        pre_z: float,
        post_z: float,
        pre_joints: np.ndarray,
        post_joints: np.ndarray,
    ) -> tuple[bool, float, float]:
        """Evaluate whether the motor command succeeded.

        Returns (success, confidence, error_magnitude).

        Success criteria are physically grounded:
        - locomotion: root body height > 33% of initial (didn't fall) AND joints moved
        - manipulation: arm joints moved AND no catastrophic instability
        - head: neck joints moved toward commanded direction
        """
        joint_delta = float(np.abs(post_joints - pre_joints).sum())
        fall_threshold = 0.33 * self._initial_root_z
        height_ok = post_z > fall_threshold

        if channel == "locomotion":
            moved = joint_delta > 0.01
            success = height_ok and moved
            ratio = post_z / self._initial_root_z
            confidence = float(np.clip(ratio, 0.0, 1.0))
            error_mag = float(np.clip(1.0 - ratio, 0.0, 1.0))

        elif channel == "manipulation":
            moved = joint_delta > 0.005
            success = moved and height_ok
            confidence = 0.7 if moved else 0.3
            error_mag = 0.0 if moved else 0.5

        elif channel == "head":
            moved = joint_delta > 0.002
            success = moved
            confidence = 0.8 if moved else 0.4
            error_mag = 0.0 if moved else 0.3

        else:
            success = joint_delta > 0.001
            confidence = 0.5
            error_mag = 0.5

        return success, confidence, error_mag

    @staticmethod
    def _stochastic_outcome(channel: str, intensity: float) -> dict[str, Any]:
        """Simple stochastic outcome for non-physics channels."""
        base_success = 0.7 - 0.3 * (intensity - 0.5) ** 2
        success = np.random.random() < base_success
        return {
            "channel": channel,
            "success": bool(success),
            "confidence": 0.5 + 0.3 * np.random.random(),
            "proprioceptive_state": [intensity * (1.0 if success else 0.3)],
            "error_magnitude": 0.0 if success else 0.3,
        }
