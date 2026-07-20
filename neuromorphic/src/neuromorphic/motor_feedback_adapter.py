"""Motor Feedback Adapter — routes motor commands to MuJoCo or real hardware.

Per-channel auto-detection: if a real actuator publishes heartbeats on
``actuator.heartbeat.{channel}``, that channel routes to hardware.
Otherwise it routes to MuJoCo (physics) or stochastic (expression/speech).

Transition is automatic and per-channel — plug in a real arm and
manipulation goes real while locomotion stays virtual.  Unplug it and
after ``heartbeat_timeout_s`` it falls back to MuJoCo.

All motor proposals flow through the Kernel first.  This adapter only
acts on ALLOW decisions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from neuromorphic.config import MotorFeedbackConfig
from neuromorphic.mujoco_body import STOCHASTIC_CHANNELS, MuJoCoBody
from neuromorphic.tasks import TaskCurriculum, task_result_to_outcome

logger = logging.getLogger(__name__)

_HEARTBEAT_SUBJECT = "actuator.heartbeat.>"


class MotorFeedbackAdapter:
    """Routes motor commands to virtual (MuJoCo) or real actuators per-channel.

    Lifecycle:
        adapter = MotorFeedbackAdapter(config, event_bus)
        await adapter.start()     # subscribes to heartbeats
        ...
        await adapter.handle_motor_command(cmd)  # called per ALLOW decision
        ...
        await adapter.stop()
    """

    def __init__(self, config: MotorFeedbackConfig, event_bus: Any):
        self._config = config
        self._bus = event_bus
        self._real_channels: dict[str, float] = {}  # channel → last heartbeat ts
        self._mujoco_body: MuJoCoBody | None = None  # lazy-loaded
        # Rate limiting: last command time per channel (monotonic)
        self._last_cmd_time: dict[str, float] = {}
        self._rate_limit_interval: float = (
            1.0 / config.motor_rate_limit_hz if config.motor_rate_limit_hz > 0 else 0.0
        )
        # Guards concurrent MuJoCo body access (guide_to_pose, apply_force,
        # step_command all mutate physics state — must not overlap).
        self._body_lock = asyncio.Lock()
        # Teach sequence cancellation: set to True to abort the current
        # teach loop at the next rep boundary.
        self._teach_cancel = False
        self._teach_running = False
        # Continuous physics loop task handle
        self._physics_task: asyncio.Task | None = None
        # Task curriculum (structured goals — stand, balance, reach, walk)
        self._curriculum: TaskCurriculum | None = None
        self._tasks_enabled = config.tasks_enabled
        self._physics_running = False
        # ActuatorPlugin registry: channel → plugin.  When a plugin is
        # registered and is_real() is True for that channel, handle_motor_command
        # dispatches via plugin.execute() instead of publishing to motor.execute.{channel}.
        self._actuator_plugins: dict[str, Any] = {}  # ActuatorPlugin, lazy-imported
        self._started = False

    async def start(self) -> None:
        """Subscribe to actuator heartbeats and guidance commands."""
        await self._bus.subscribe(_HEARTBEAT_SUBJECT, self._handle_heartbeat)
        await self._bus.subscribe("motor.guidance", self._handle_guidance)
        # Start continuous physics loop if configured
        if self._config.mujoco_continuous:
            self._physics_running = True
            self._physics_task = asyncio.create_task(self._physics_loop())
            logger.info(
                "Continuous physics loop started: physics=%.0fHz, proprio=%.0fHz, viz=%.0fHz",
                self._config.mujoco_physics_hz,
                self._config.mujoco_proprio_hz,
                self._config.mujoco_viz_hz,
            )
        if self._config.population_vector and not self._config.mujoco_continuous:
            logger.warning(
                "population_vector=True requires mujoco_continuous=True for per-joint "
                "control. Non-continuous step_command uses uniform torque."
            )
        for ch, plugin in self._actuator_plugins.items():
            try:
                await plugin.start(self._bus)
                logger.info("Started pre-registered ActuatorPlugin for channel '%s'", ch)
            except Exception as exc:
                logger.error("Failed to start ActuatorPlugin for channel '%s': %s", ch, exc)
        self._started = True
        logger.info("MotorFeedbackAdapter started — listening for actuator heartbeats + guidance")

    async def stop(self) -> None:
        """Unsubscribe and clean up."""
        self._physics_running = False
        if self._physics_task is not None:
            self._physics_task.cancel()
            try:
                await self._physics_task
            except asyncio.CancelledError:
                pass
            self._physics_task = None
        try:
            await self._bus.unsubscribe(_HEARTBEAT_SUBJECT)
        except Exception:
            pass  # best-effort on shutdown
        self._started = False
        for ch, plugin in self._actuator_plugins.items():
            try:
                await plugin.stop()
            except Exception as exc:
                logger.debug("Error stopping ActuatorPlugin for channel '%s': %s", ch, exc)

    def _ensure_mujoco(self) -> MuJoCoBody:
        """Lazy-load MuJoCo body on first physics command."""
        if self._mujoco_body is None:
            steps = getattr(self._config, "mujoco_steps_per_command", 200)
            self._mujoco_body = MuJoCoBody(
                steps_per_command=steps,
                channel_actuators=self._config.channel_actuators,
            )
        return self._mujoco_body

    def register_plugin(self, channel: str, plugin: Any) -> None:
        """Register an ActuatorPlugin for a motor channel.

        When a plugin is registered for a channel and ``is_real(channel)`` is
        True, ``handle_motor_command()`` dispatches directly through
        ``plugin.execute()`` instead of publishing to ``motor.execute.{channel}``.
        The NATS fallback is preserved for channels without a registered plugin.

        Args:
            channel: Motor channel name (e.g. ``"manipulation"``).
            plugin:  The ActuatorPlugin instance to use for that channel.
        """
        self._actuator_plugins[channel] = plugin
        logger.info("ActuatorPlugin registered for channel '%s': %s", channel, plugin.actuator_id)
        if self._started:
            asyncio.create_task(self._launch_plugin(channel, plugin))

    async def _launch_plugin(self, channel: str, plugin: Any) -> None:
        """Start a plugin registered after the adapter was already started."""
        try:
            await plugin.start(self._bus)
            logger.info(
                "Started ActuatorPlugin for channel '%s' (registered post-start)",
                channel,
            )
        except Exception as exc:
            logger.error(
                "Failed to start ActuatorPlugin for channel '%s': %s",
                channel,
                exc,
            )

    def is_real(self, channel: str) -> bool:
        """Check if channel has a live real actuator (heartbeat within timeout)."""
        last_hb = self._real_channels.get(channel)
        if last_hb is None:
            return False
        timeout = getattr(self._config, "heartbeat_timeout_s", 10.0)
        return (time.time() - last_hb) < timeout

    async def handle_motor_command(
        self,
        channel: str,
        intensity: float,
        trace_id: str = "",
        actuator_intensities: dict[str, float] | None = None,
    ) -> None:
        """Process an ALLOW-ed motor command.

        Routes to real actuator (if heartbeat active) or virtual body.
        Publishes ``motor.outcome.{channel}`` for the brain to consume.

        Args:
            channel: Motor channel name (locomotion, manipulation, head, etc.)
            intensity: 0.0-1.0 from brain's population vector decoding
            trace_id: Original proposal trace_id for correlation
            actuator_intensities: Per-actuator intensities from population vector
                decoder.  When provided and ``population_vector`` is enabled,
                routes to ``set_control_vector()`` for individuated joint control.
        """
        # Rate limiting: skip commands that arrive faster than the configured limit.
        # Protects physical servos from rapid direction changes.
        if self._rate_limit_interval > 0:
            now = time.monotonic()
            last = self._last_cmd_time.get(channel, 0.0)
            if (now - last) < self._rate_limit_interval:
                return  # too fast, skip
            self._last_cmd_time[channel] = now

        if self.is_real(channel):
            plugin = self._actuator_plugins.get(channel)
            if plugin is not None:
                # Dispatch directly through the ActuatorPlugin interface.
                # Lazy import so the module loads without the activelearning SDK.
                from activelearning.core import ActionProposal, generate_trace_id  # noqa: PLC0415

                action: dict[str, Any] = {"channel": channel, "intensity": intensity}
                if actuator_intensities:
                    action["actuator_intensities"] = actuator_intensities
                proposal: ActionProposal = ActionProposal(
                    trace_id=trace_id or generate_trace_id(),
                    provenance=f"motor.{channel}",
                    action=action,
                )
                outcome_obj = await plugin.execute(proposal)
                motor_outcome: dict[str, Any] = {
                    "channel": channel,
                    "success": outcome_obj.success,
                    "confidence": 0.9 if outcome_obj.success else 0.1,
                    "proprioceptive_state": [],
                    "error_magnitude": 0.0 if outcome_obj.success else 1.0,
                }
                await self._bus.publish(f"motor.outcome.{channel}", motor_outcome)
            else:
                # Fallback: publish to NATS for an external hardware process.
                msg: dict[str, Any] = {
                    "channel": channel,
                    "intensity": intensity,
                    "trace_id": trace_id,
                }
                if actuator_intensities:
                    msg["actuator_intensities"] = actuator_intensities
                await self._bus.publish(f"motor.execute.{channel}", msg)
            logger.debug("Motor command routed to real actuator: %s", channel)
            return

        # Virtual body simulation
        if channel in STOCHASTIC_CHANNELS:
            outcome = MuJoCoBody._stochastic_outcome(channel, intensity)
        elif self._config.mujoco_continuous:
            # Continuous mode: set controls, physics loop does stepping.
            # Return a lightweight acknowledgment — real proprioceptive
            # feedback arrives via the continuous proprio emission.
            async with self._body_lock:
                body = self._ensure_mujoco()
                if actuator_intensities and self._config.population_vector:
                    body.set_control_vector(channel, actuator_intensities)
                else:
                    body.set_control(channel, intensity)
            outcome = {
                "channel": channel,
                "success": True,
                "confidence": 0.6,
                "proprioceptive_state": [],
                "error_magnitude": 0.0,
            }
        else:
            async with self._body_lock:
                body = self._ensure_mujoco()
                loop = asyncio.get_running_loop()
                outcome = await loop.run_in_executor(
                    None,
                    body.step_command,
                    channel,
                    intensity,
                )

        # Simulate real-world feedback delay
        delay_ms = getattr(self._config, "virtual_delay_ms", 0)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

        # Publish outcome — brain's _handle_motor_outcome picks this up
        await self._bus.publish(f"motor.outcome.{channel}", outcome)
        logger.debug(
            "Virtual motor outcome: %s success=%s confidence=%.2f",
            channel,
            outcome.get("success"),
            outcome.get("confidence", 0),
        )

        # Publish full body state for visualization (MuJoCo physics channels only)
        if self._mujoco_body is not None and channel not in STOCHASTIC_CHANNELS:
            await self._publish_body_state(channel, outcome)

    async def _publish_body_state(
        self,
        channel: str,
        outcome: dict[str, Any],
    ) -> None:
        """Publish full body state to NATS for visualization."""
        try:
            async with self._body_lock:
                state = self._mujoco_body.get_full_state()
            state["active_channel"] = channel
            state["success"] = outcome.get("success", False)
            await self._bus.publish("mujoco.body.state", state)
        except Exception as e:
            logger.debug("Failed to publish body state: %s", e)

    async def _handle_guidance(self, data: dict[str, Any]) -> None:
        """Handle human guidance commands (pose setting, pushing, reward, teach).

        Message format:
            {"action": "pose", "joints": {"r_hip_pitch": 30, "l_hip_pitch": -10, ...}}
            {"action": "push", "body": "torso", "force": [0, 5, 0]}
            {"action": "reward", "channel": "locomotion", "success": true}
            {"action": "teach", "joints": {...}, "repeats": 3}
            {"action": "reset"}
        """
        action = data.get("action", "")
        loop = asyncio.get_running_loop()

        if action == "teach":
            # Cancel any running teach sequence before starting a new one
            if self._teach_running:
                self._teach_cancel = True
            await self._handle_teach(data)
            return

        # Cancel running teach sequence if user sends a manual pose/push/reset
        if self._teach_running and action in ("pose", "push", "reset"):
            self._teach_cancel = True

        if action == "pose":
            joints = data.get("joints", {})
            if not joints:
                return
            async with self._body_lock:
                body = self._ensure_mujoco()
                outcome = await loop.run_in_executor(
                    None,
                    body.guide_to_pose,
                    joints,
                )

        elif action == "push":
            body_name = data.get("body", "torso")
            force = data.get("force", [0, 0, 0])
            async with self._body_lock:
                body = self._ensure_mujoco()
                outcome = await loop.run_in_executor(
                    None,
                    body.apply_force,
                    body_name,
                    tuple(force),
                )

        elif action == "reward":
            channel = data.get("channel", "locomotion")
            success = data.get("success", True)
            outcome = {
                "channel": channel,
                "success": success,
                "confidence": 1.0,  # human reward = high certainty
                "proprioceptive_state": [],
                "error_magnitude": 0.0 if success else 0.5,
                "guided": True,
            }

        elif action == "reset":
            async with self._body_lock:
                body = self._ensure_mujoco()
                body.reset()
                # Reset curriculum so tasks don't hold stale target positions
                if self._curriculum is not None:
                    self._curriculum = TaskCurriculum(body)
            logger.info("MuJoCo body reset by guidance command")
            await self._publish_body_state("locomotion", {"success": True})
            return

        else:
            logger.warning("Unknown guidance action: %s", action)
            return

        channel = outcome.get("channel", "locomotion")
        await self._bus.publish(f"motor.outcome.{channel}", outcome)
        logger.info(
            "Guidance %s → %s success=%s",
            action,
            channel,
            outcome.get("success"),
        )

        # Publish body state for viz
        if self._mujoco_body is not None:
            await self._publish_body_state(channel, outcome)

    async def _handle_teach(self, data: dict[str, Any]) -> None:
        """Execute a teach sequence: pose + auto-reward + DA boost, repeated N times.

        Teach mode auto-pairs each guided pose with a success reward and a DA
        boost event timed so eligibility traces from the sensory→motor pathway
        are still active.  Repeats are spaced ~2s apart to allow partial trace
        accumulation between reps.

        The adapter publishes ``neuromod.teach.da`` for the brain service to
        apply a DA burst, keeping three-factor learning alive during the
        feedback window.

        Concurrent teach sequences are serialised: a new teach request cancels
        the running one at the next rep boundary.
        """
        joints = data.get("joints", {})
        if not joints:
            return
        try:
            repeats = max(1, min(int(data.get("repeats", 1)), 20))
        except (ValueError, TypeError):
            repeats = 1

        # Wait for any previous teach sequence to finish cancelling
        while self._teach_running:
            self._teach_cancel = True
            await asyncio.sleep(0.1)

        self._teach_running = True
        self._teach_cancel = False
        loop = asyncio.get_running_loop()

        logger.info("Teach sequence: %d reps, joints=%s", repeats, list(joints.keys()))

        try:
            for i in range(repeats):
                if self._teach_cancel:
                    logger.info("Teach sequence cancelled at rep %d/%d", i + 1, repeats)
                    break

                # 1. Guide body to target pose (under lock to avoid concurrent body access)
                async with self._body_lock:
                    body = self._ensure_mujoco()
                    outcome = await loop.run_in_executor(
                        None,
                        body.guide_to_pose,
                        joints,
                    )
                channel = outcome.get("channel", "locomotion")
                outcome["teach_rep"] = i + 1
                outcome["teach_total"] = repeats

                # 2. Publish motor outcome (proprioceptive feedback for the brain)
                await self._bus.publish(f"motor.outcome.{channel}", outcome)

                # 3. Publish DA boost event so eligibility traces get reinforced.
                await self._bus.publish(
                    "neuromod.teach.da",
                    {
                        "channel": channel,
                        "amount": 1.5,
                        "source": "teach",
                    },
                )

                # 4. Publish progress for UI
                await self._bus.publish(
                    "teach.progress",
                    {
                        "current": i + 1,
                        "total": repeats,
                        "channel": channel,
                    },
                )

                # 5. Publish body state for visualization
                if self._mujoco_body is not None:
                    await self._publish_body_state(channel, outcome)

                logger.info("Teach rep %d/%d → %s", i + 1, repeats, channel)

                # 6. Wait between reps for trace accumulation (~2s spacing).
                # Check cancel flag after sleep to respond promptly.
                if i < repeats - 1:
                    await asyncio.sleep(2.0)
                    if self._teach_cancel:
                        logger.info("Teach sequence cancelled after rep %d/%d", i + 1, repeats)
                        break
        finally:
            self._teach_running = False
            self._teach_cancel = False

    async def _physics_loop(self) -> None:
        """Continuous physics loop — steps MuJoCo at fixed rate.

        Runs as a background asyncio task.  Decouples physics from motor
        commands so the body is always "alive" (gravity, contacts, damping)
        even when the brain isn't firing motor neurons.

        Four sub-rates:
          - physics_hz: mj_step calls (50Hz default = 20ms per tick)
          - proprio_hz: proprioceptive vector → NATS observation.proprioceptive
          - camera_hz: 64x64 grayscale frame → NATS observation.visual.body
          - viz_hz: full body state → NATS mujoco.body.state
        """
        physics_dt = 1.0 / self._config.mujoco_physics_hz
        proprio_interval = 1.0 / self._config.mujoco_proprio_hz
        viz_interval = 1.0 / self._config.mujoco_viz_hz
        camera_interval = 0.5  # 2 Hz — matches video sensor rate

        last_proprio = 0.0
        last_viz = 0.0
        last_camera = 0.0
        last_task = 0.0
        task_interval = 1.0  # 1 Hz — ~1 task eval per brain step at 0.35 steps/sec
        loop = asyncio.get_running_loop()

        logger.info("Physics loop running at %.0f Hz", self._config.mujoco_physics_hz)

        while self._physics_running:
            try:
                t0 = loop.time()

                # Step physics (under lock to avoid concurrent body mutation)
                async with self._body_lock:
                    body = self._ensure_mujoco()
                    # Apply PD support force every physics step (not just 1 Hz)
                    if self._curriculum is not None:
                        self._curriculum.apply_continuous_support(body)
                    body.step_batch(1)

                now = loop.time()

                # Emit proprioceptive state at proprio_hz
                if now - last_proprio >= proprio_interval:
                    last_proprio = now
                    try:
                        async with self._body_lock:
                            vec = body.get_proprioceptive_vector()
                        await self._bus.publish(
                            "observation.proprioceptive",
                            {
                                "data": vec.tolist(),
                                "provenance": "observation.proprioceptive",
                            },
                        )
                    except Exception as e:
                        logger.debug("Failed to emit proprioceptive state: %s", e)

                    # Body posture signal: height ratio + standing torques for
                    # continuous DA modulation and standing pattern injection.
                    try:
                        async with self._body_lock:
                            root_z = float(body._data.body(body._root_body_name).xpos[2])
                            height_ratio = (
                                root_z / body._initial_root_z if body._initial_root_z > 0 else 0.0
                            )
                            task_name = ""
                            support_active = False
                            if self._curriculum is not None:
                                task_name = self._curriculum.current_task.name
                                support_active = self._curriculum.support_active
                            # Only compute PD torques when support is active (expensive)
                            standing_torques = (
                                body.compute_standing_torques() if support_active else {}
                            )
                        await self._bus.publish(
                            "body.posture",
                            {
                                "height_ratio": float(np.clip(height_ratio, 0.0, 1.5)),
                                "task_name": task_name,
                                "support_active": support_active,
                                "standing_torques": standing_torques,
                            },
                        )
                    except Exception as e:
                        logger.debug("Failed to emit body posture: %s", e)

                    # Pain signal at same rate as proprioception
                    if self._config.pain_enabled:
                        try:
                            async with self._body_lock:
                                pain_vec = body.compute_pain_vector(self._config.pain_limit_zone)
                            if pain_vec.max() > 0.01:
                                await self._bus.publish(
                                    "observation.pain",
                                    {
                                        "data": pain_vec.tolist(),
                                        "provenance": "observation.pain",
                                    },
                                )
                        except Exception as e:
                            logger.debug("Failed to emit pain signal: %s", e)

                # Task curriculum evaluation at 1 Hz (~1 eval per brain step)
                if self._tasks_enabled and now - last_task >= task_interval:
                    last_task = now
                    try:
                        async with self._body_lock:
                            if self._curriculum is None:
                                self._curriculum = TaskCurriculum(body)
                                logger.info(
                                    "TaskCurriculum started: %s", self._curriculum.current_task.name
                                )
                            result = self._curriculum.step()
                            outcome = task_result_to_outcome(
                                result, self._curriculum.current_task, body
                            )
                        await self._bus.publish(f"motor.outcome.{outcome['channel']}", outcome)
                    except Exception as e:
                        logger.debug("Task evaluation error: %s", e)

                # Emit camera frame at 2 Hz — brain sees its own body
                if now - last_camera >= camera_interval:
                    last_camera = now
                    try:
                        async with self._body_lock:
                            frame = body.render_frame(64, 64)
                        if frame is not None:
                            # Normalize to [0, 1] float32 list for sensory encoding
                            data = (frame.astype("float32") / 255.0).flatten().tolist()
                            await self._bus.publish(
                                "observation.visual.body",
                                {
                                    "data": data,
                                    "provenance": "sensor.video.body",
                                },
                            )
                    except Exception as e:
                        logger.debug("Failed to emit camera frame: %s", e)

                # Emit visualization state at viz_hz
                if now - last_viz >= viz_interval:
                    last_viz = now
                    try:
                        async with self._body_lock:
                            state = body.get_full_state()
                        task_name = ""
                        if self._curriculum is not None:
                            task_name = self._curriculum.current_task.name
                        state["active_channel"] = task_name or "idle"
                        state["success"] = False
                        await self._bus.publish("mujoco.body.state", state)
                    except Exception as e:
                        logger.debug("Failed to publish viz state: %s", e)

                # Sleep until next physics tick
                elapsed = loop.time() - t0
                sleep_time = physics_dt - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                raise  # let task cancellation propagate
            except Exception as e:
                logger.error("Physics loop error (recovering): %s", e)
                await asyncio.sleep(0.1)  # prevent tight error loop

    async def _handle_heartbeat(self, data: dict[str, Any]) -> None:
        """Track real actuator announcements."""
        channel = data.get("channel", "")
        if not channel:
            return

        was_real = self.is_real(channel)
        self._real_channels[channel] = time.time()

        if not was_real:
            actuator_id = data.get("actuator_id", "unknown")
            logger.info(
                "Real actuator detected for '%s': %s — switching from virtual",
                channel,
                actuator_id,
            )
