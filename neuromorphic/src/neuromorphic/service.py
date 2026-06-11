"""NeuromorphicService -- main entry point, integrates with NATS event bus."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

from activelearning import BaseService
from activelearning.core import generate_trace_id

from neuromorphic.auditory_stm import AuditorySTM, AuditorySTMConfig
from neuromorphic.config import NeuromorphicConfig
from neuromorphic.encoding import _resolve_modality
from neuromorphic.network import NeuromorphicNetwork
from neuromorphic.persistence import NeuromorphicPersistence
from neuromorphic.watchdog import NeuralWatchdog, WatchdogConfig, WatchdogStatus, AlertLevel

if TYPE_CHECKING:
    from neuromorphic.motor_feedback_adapter import MotorFeedbackAdapter

logger = logging.getLogger(__name__)


class NeuromorphicService(BaseService):
    """
    Neuromorphic cognitive core service.

    Subscribes to observations, processes them through the spiking neural
    network, and emits motor proposals (Kernel-gated) and metrics.

    NATS subjects:
    - Subscribe: observation.*, neuromorphic.status, neuromorphic.drives.event,
                 neuromorphic.teach, neuromorphic.train_bulk,
                 cognitive.response.validated, motor.outcome.> (when enabled)
    - Publish: proposal.new, speech.output, cognitive.query,
               beliefs.add_node, memory.store, knowledge.gap
    """

    MAX_TEACH_REPETITIONS = 100
    MAX_BULK_CORPUS = 1000
    MAX_BULK_REPS = 20

    def __init__(self):
        super().__init__("neuromorphic", use_database=False, use_event_bus=True)
        self._config = NeuromorphicConfig.from_env()
        self._network: Optional[NeuromorphicNetwork] = None
        self._persistence: Optional[NeuromorphicPersistence] = None

        # Background task handles
        self._sim_task: Optional[asyncio.Task] = None
        self._drive_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._save_task: Optional[asyncio.Task] = None

        # Latest observation per modality — written by _handle_observation,
        # atomically swapped by the simulation loop.  Replaces the old
        # per-modality asyncio.Queue system which couldn't drain fast enough
        # and caused NATS slow-consumer disconnects.
        self._latest_obs: dict[str, dict[str, Any]] = {}

        # Auditory short-term memory (echoic memory buffer)
        self._auditory_stm = AuditorySTM(AuditorySTMConfig(
            window_steps=self._config.auditory_stm_window,
            decay_rate=self._config.auditory_stm_decay,
        ))

        # Thread executor for running network.step() off the event loop.
        # This keeps the asyncio event loop responsive so the NATS client
        # can continuously drain incoming messages during the ~1.6s step.
        self._step_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="brain-step")

        # Simulation running flag
        self._running = False

        # Zero-observation watchdog — warns if brain runs too long without input
        self._last_obs_step: int = 0
        self._obs_warning_logged: bool = False
        self._gateway_restart_requested: bool = False
        self._gateway_restart_threshold: int = 500  # steps without obs before requesting restart

        # Lock protects _network from concurrent mutation
        self._net_lock = asyncio.Lock()

        # Motor echo tracker — per-channel record of when motor commands fire.
        # The echo sustains a DA boost so eligibility traces on motor pathways
        # stay alive until proprioceptive feedback arrives.
        # Written ONLY from the async event loop (after sim step returns),
        # read ONLY from _step_in_thread (inside _net_lock).  Safe because
        # writes happen after the lock is released, reads happen under lock.
        self._motor_echo_steps: dict[str, int] = {}  # channel → step when fired
        self._motor_echo_ts: dict[str, float] = {}  # channel → monotonic time when fired (for wall-clock mode)
        # Queued motor outcomes — outcomes that arrive via NATS are queued
        # here and consumed by the next simulation loop iteration.  This
        # avoids blocking on _net_lock inside the outcome handler.
        self._pending_motor_outcomes: list[dict[str, Any]] = []

        # Outcome-contingent DA — success boosts DA, failure dips DA.
        # This is what allows three-factor learning to distinguish good
        # from bad motor commands.  Decays after outcome_da_steps.
        self._outcome_da_multiplier: float = 1.0  # current multiplier
        self._outcome_da_steps_remaining: int = 0  # steps until expiry

        # Body posture state (updated at proprio rate from adapter)
        self._body_height_ratio: float = 0.0
        self._body_task_name: str = ""
        self._body_support_active: bool = False
        self._body_standing_torques: dict[str, float] = {}

        # Latest pain signal — max pain intensity from most recent observation.
        # Written by _handle_observation when provenance is observation.pain,
        # consumed by _step_in_thread to modulate DA (punishment).
        self._latest_pain_max: float = 0.0

        # Timing log counter — used to emit step timing breakdown every 5th metrics emission
        self._timing_log_counter: int = 0

        # Motor feedback adapter — routes commands to MuJoCo or real hardware
        self._motor_adapter: Optional["MotorFeedbackAdapter"] = None

        # Safety gate — pending Kernel decisions keyed by trace_id.
        # Motor proposals register a Future here; the decision handler resolves it.
        self._pending_decisions: dict[str, asyncio.Future] = {}
        # Track proposals by trace_id for feedback injection
        self._pending_proposals: dict[str, dict[str, Any]] = {}
        # Track background decision-waiting tasks for clean shutdown
        self._decision_tasks: set[asyncio.Task] = set()
        # Cap on concurrent pending decisions to prevent unbounded growth
        _MAX_PENDING_DECISIONS = 500

        # Neural watchdog — monitors brain state for pathological patterns
        self._watchdog = NeuralWatchdog(WatchdogConfig(
            weight_oscillation_min_amplitude=float(
                os.environ.get("NEURO_WATCHDOG_MIN_AMPLITUDE", "0.0")
            ),
            weight_oscillation_window=int(
                os.environ.get("NEURO_WATCHDOG_OSC_WINDOW", "6")
            ),
            weight_oscillation_threshold=int(
                os.environ.get("NEURO_WATCHDOG_OSC_THRESHOLD", "4")
            ),
        ))
        # Cooldown: prevent watchdog escalation spam (min steps between escalations)
        self._last_escalation_step: int = -10000
        # Persistent governor corrections — applied EVERY step until next check
        # clears them.  Without persistence, a single-step correction is useless
        # because the driving current resumes immediately.
        self._persistent_governor: dict[str, float] = {}

    async def _setup(self) -> None:
        """Initialize network, load state, subscribe to NATS, start loops."""
        # Create persistence
        self._persistence = NeuromorphicPersistence(self._config.sqlite_path)
        await self._persistence.open()

        # Create network
        self._network = NeuromorphicNetwork(self._config)

        # Load saved state if available
        if await self._persistence.has_saved_state():
            state = await self._persistence.load_state()
            if state:
                self._network.set_state(state)
                # Restore AuditorySTM state if saved
                stm_state = state.get("auditory_stm")
                if stm_state:
                    self._auditory_stm.set_state(stm_state)
                    self.logger.info(
                        f"Restored auditory STM ({self._auditory_stm.n_frames} frames)"
                    )
                # Restore watchdog state (silence counters, weight history, etc.)
                wd_state = state.get("watchdog")
                if wd_state:
                    self._watchdog.set_state(wd_state)
                    self.logger.info("Restored watchdog state")
                phase = self._network.neuromodulation.phase
                self.logger.info(
                    f"Resumed from step {self._network.step_count} "
                    f"(phase: {phase})"
                )
        else:
            self.logger.info("No saved state — starting as newborn infant")

        # Subscribe to NATS subjects
        await self.event_bus.subscribe(
            "observation.>", self._handle_observation,
            pending_msgs_limit=256 * 1024,
            pending_bytes_limit=256 * 1024 * 1024,
        )
        await self.event_bus.subscribe("neuromorphic.status", self._handle_status)
        await self.event_bus.subscribe("neuromorphic.drives.event", self._handle_drive_event)
        await self.event_bus.subscribe("neuromorphic.teach", self._handle_teach)
        await self.event_bus.subscribe("neuromorphic.train_bulk", self._handle_train_bulk)
        await self.event_bus.subscribe("neuromorphic.concept.probe", self._handle_concept_probe)
        await self.event_bus.subscribe("cognitive.response.validated", self._handle_cognitive_response)
        # Motor feedback loop — subscribe to outcome signals from actuators,
        # sensors (IMU, camera, joints), simulators, or human teachers.
        if self._config.motor_feedback.enabled:
            await self.event_bus.subscribe("motor.outcome.>", self._handle_motor_outcome)
            # Start motor feedback adapter (MuJoCo virtual body + real actuator detection)
            from neuromorphic.motor_feedback_adapter import MotorFeedbackAdapter
            self._motor_adapter = MotorFeedbackAdapter(self._config.motor_feedback, self.event_bus)
            await self._motor_adapter.start()
            # Teach mode DA boost — adapter publishes neuromod.teach.da when
            # a teach sequence fires, and we register a motor echo so the DA
            # boost sustains eligibility traces through the feedback window.
            await self.event_bus.subscribe("neuromod.teach.da", self._handle_teach_da)
            await self.event_bus.subscribe("body.posture", self._handle_body_posture)
            self.logger.info("Motor feedback loop enabled — MuJoCo body + actuator detection active")
        # Safety gate — subscribe to Kernel decisions so DENY/TRANSFORM can
        # feed back as negative/corrective learning signals.
        if self._config.safety_gate.enabled:
            await self.event_bus.subscribe("decision.>", self._handle_kernel_decision)
            await self.event_bus.subscribe("approval.response.>", self._handle_approval_response)
            self.logger.info("Safety gate enabled — motor commands routed through Kernel")

        # Start background tasks
        self._running = True
        self._sim_task = asyncio.create_task(self._simulation_loop())
        self._drive_task = asyncio.create_task(self._drive_update_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        self._save_task = asyncio.create_task(self._save_loop())

        self.logger.info("Neuromorphic service started — cognitive core active")

    async def _cleanup(self) -> None:
        """Save state and stop all loops."""
        self._running = False

        # Cancel background tasks
        for task in [self._sim_task, self._drive_task, self._metrics_task, self._save_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cancel any in-flight safety decision tasks
        for task in list(self._decision_tasks):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._decision_tasks.clear()
        self._pending_decisions.clear()
        self._pending_proposals.clear()

        # Shut down the step executor
        self._step_executor.shutdown(wait=True, cancel_futures=True)

        # Wait for any in-flight background save to finish
        if self._persistence:
            await self._persistence.wait_bg_save()

        # Final synchronous save (acquire lock to get consistent state)
        if self._network and self._persistence:
            async with self._net_lock:
                state = self._network.get_state()
                state["auditory_stm"] = self._auditory_stm.get_state()
                state["watchdog"] = self._watchdog.get_state()
            await self._persistence.save_state(state)
            self.logger.info("Final state saved")

        if self._motor_adapter:
            await self._motor_adapter.stop()

        if self._persistence:
            await self._persistence.close()

    # ===== Message Handlers =====

    async def _handle_observation(self, data: dict[str, Any]) -> None:
        """Store the latest observation per modality.

        Overwrites any previous observation for the same modality — we only
        care about the most recent data.  The simulation loop atomically
        swaps the dict before each step, so this is lock-free.

        Audio observations with temporal frames are routed through the
        AuditorySTM buffer for cross-step temporal accumulation. The STM
        produces a richer feature vector that captures audio context across
        multiple seconds (echoic memory).
        """
        provenance = data.get("provenance", "")
        modality = _resolve_modality(provenance)

        if modality == "auditory":
            raw_data = data.get("data", data)
            if isinstance(raw_data, dict) and raw_data.get("type") == "temporal_audio_frame":
                # Feed the temporal frame into the STM for cross-step accumulation
                self._auditory_stm.update(raw_data)
                # Get enriched features from the STM
                stm_features = self._auditory_stm.get_features()
                if stm_features is not None:
                    # Replace the observation data with STM-enriched features
                    data = {
                        "provenance": provenance or "sensor.audiofile",
                        "data": stm_features.tolist(),
                    }
            # If it's a plain MFCC list (legacy/non-aggregated), pass through as-is

        self._latest_obs[modality] = data

        # Track pain intensity for DA penalty in next step
        if provenance == "observation.pain":
            raw = data.get("data", [])
            if isinstance(raw, list) and raw:
                self._latest_pain_max = max(raw)

    async def _handle_status(self, data: dict[str, Any]) -> None:
        """Return network metrics."""
        if self._network:
            async with self._net_lock:
                metrics = self._network.get_metrics()
            await self.event_bus.publish("neuromorphic.metrics", metrics)

    async def _handle_drive_event(self, data: dict[str, Any]) -> None:
        """Handle external drive events (feed, damage, rest, etc.)."""
        if self._network:
            async with self._net_lock:
                self._network.drives.handle_event(data)

    async def _handle_teach(self, data: dict[str, Any]) -> None:
        """
        Handle teaching input — synchronized multi-modal demonstration.

        Expected format:
        {
            "inputs": {"sensor.camera": [...], "sensor.audio": [...]},
            "description": "human-readable description",
            "repetitions": 5
        }
        """
        if not self._network:
            return

        inputs = data.get("inputs", {})
        repetitions = min(int(data.get("repetitions", 1)), self.MAX_TEACH_REPETITIONS)
        description = data.get("description", "teaching event")

        self.logger.info(f"Teaching: {description} ({repetitions} reps)")

        step_count = 0
        for i in range(repetitions):
            async with self._net_lock:
                sensory_current = self._network.inject_multimodal(inputs)
                self._network.step(sensory_current)
                step_count = self._network.step_count
            # Yield every 10 reps to avoid blocking the event loop
            if i % 10 == 9:
                await asyncio.sleep(0)

        # Store teaching event in memory
        try:
            await self.event_bus.publish("memory.store", {
                "trace_id": generate_trace_id(),
                "type": "teaching",
                "description": description,
                "modalities": list(inputs.keys()),
                "step": step_count,
            })
        except Exception as e:
            self.logger.warning(f"Failed to publish teaching event: {e}")

    async def _handle_train_bulk(self, data: dict[str, Any]) -> None:
        """
        Handle bulk training — run a corpus through the network at high speed.

        Expected format:
        {
            "corpus": ["text1", "text2", ...],
            "repetitions_per_item": 5,
            "stdp_interval_override": 3
        }
        """
        if not self._network:
            return

        corpus = data.get("corpus", [])
        if not corpus:
            return

        # Cap corpus and reps
        corpus = corpus[:self.MAX_BULK_CORPUS]
        reps = min(int(data.get("repetitions_per_item", 5)), self.MAX_BULK_REPS)
        stdp_override = data.get("stdp_interval_override")

        self.logger.info(
            f"Bulk training: {len(corpus)} items × {reps} reps "
            f"= {len(corpus) * reps} steps"
        )

        # Optionally override STDP interval for faster training
        # Use the lock to safely swap the interval, avoiding mutation races
        if stdp_override is not None:
            async with self._net_lock:
                original_interval = self._network.config.stdp_update_interval
                self._network.config.stdp_update_interval = max(1, int(stdp_override))
        else:
            original_interval = None

        total_steps = 0
        t0 = time.time()

        try:
            for rep in range(reps):
                for text in corpus:
                    async with self._net_lock:
                        sensory_current = self._network.inject_observation(
                            text, provenance="train_bulk"
                        )
                        self._network.step(sensory_current)
                        total_steps += 1

                    # Yield every 50 steps to avoid blocking the event loop
                    if total_steps % 50 == 0:
                        await asyncio.sleep(0)

            elapsed = time.time() - t0
            rate = total_steps / elapsed if elapsed > 0 else 0
            self.logger.info(
                f"Bulk training done: {total_steps} steps in {elapsed:.1f}s "
                f"({rate:.1f} steps/sec)"
            )

            # Publish completion event
            try:
                async with self._net_lock:
                    step_count = self._network.step_count
                await self.event_bus.publish("memory.store", {
                    "trace_id": generate_trace_id(),
                    "type": "bulk_training",
                    "description": f"Bulk training: {len(corpus)} items × {reps} reps",
                    "total_steps": total_steps,
                    "rate": round(rate, 1),
                    "step": step_count,
                })
            except Exception as e:
                self.logger.warning(f"Failed to publish bulk training event: {e}")

        finally:
            # Restore original STDP interval under lock
            if original_interval is not None:
                async with self._net_lock:
                    self._network.config.stdp_update_interval = original_interval

    async def _handle_concept_probe(self, data: dict[str, Any]) -> None:
        """Probe concept layer response to a controlled stimulus.

        Injects a stimulus, runs propagation steps, and captures which
        concept neurons fire. Used to demonstrate learned representations.

        Expected format:
        {
            "stimulus": {"provenance": "observation.text", "data": "dog"},
            "propagation_steps": 10,
            "label": "dog_1"
        }
        """
        import numpy as np

        if not self._network:
            return

        concept = self._network.concept
        if concept is None:
            self.logger.warning("Concept probe: no concept layer (NEURO_CONCEPT_N=0)")
            await self.event_bus.publish("neuromorphic.concept.result", {
                "error": "no concept layer",
            })
            return

        stimulus = data.get("stimulus", {})
        propagation_steps = min(int(data.get("propagation_steps", 10)), 50)
        label = data.get("label", "probe")

        provenance = stimulus.get("provenance", "observation.text")
        stim_data = stimulus.get("data", "")

        # Accumulate concept spikes across propagation window
        concept_n = concept.n
        spike_counts = np.zeros(concept_n, dtype=np.int32)
        step_start = 0
        step_end = 0
        region_spike_counts: dict[str, int] = defaultdict(int)

        async with self._net_lock:
            step_start = self._network.step_count

            for i in range(propagation_steps):
                if i == 0:
                    # Inject stimulus on first step
                    sensory_current = self._network.inject_multimodal(
                        {provenance: stim_data}
                    )
                    self._network.step(sensory_current)
                else:
                    # Let activity propagate without new input
                    self._network.step()

                # Record concept spikes
                spike_counts += concept.spikes.astype(np.int32)

                # Record all region firing rates
                for rname, region in self._network.regions.items():
                    region_spike_counts[rname] += int(region.spikes.sum())

            step_end = self._network.step_count

        # Build result
        active_mask = spike_counts > 0
        active_indices = np.nonzero(active_mask)[0]
        rates = spike_counts.astype(np.float32) / propagation_steps

        region_rates = {
            rname: round(count / (propagation_steps * self._network.regions[rname].n), 4)
            for rname, count in region_spike_counts.items()
            if rname in self._network.regions
        }

        result = {
            "label": label,
            "active_indices": active_indices.tolist(),
            "active_count": int(active_mask.sum()),
            "spike_counts": spike_counts[active_mask].tolist(),
            "rates": rates[active_mask].tolist(),
            "total_concept_neurons": concept_n,
            "propagation_steps": propagation_steps,
            "step_range": [step_start, step_end],
            "region_rates": region_rates,
        }

        self.logger.info(
            f"Concept probe '{label}': {result['active_count']}/{concept_n} "
            f"neurons fired over {propagation_steps} steps"
        )

        await self.event_bus.publish("neuromorphic.concept.result", result)

    async def _handle_cognitive_response(self, data: dict[str, Any]) -> None:
        """Re-inject Kernel-validated LLM response as sensory input with boosted gain.

        Receives from cognitive.response.validated (Kernel has already checked
        for prompt injection, length, and profile capability).

        The response text is encoded as auditory input (language modality)
        with a gain multiplier so the brain pays extra attention to it.
        This creates the STDP teaching signal — neurons that fired the
        query now get reinforced by the correlated response.
        """
        if not self._network:
            return

        response_text = data.get("response_text", "")
        if not isinstance(response_text, str) or not response_text:
            return

        try:
            gain = self._config.cognitive_action.response_injection_gain
            async with self._net_lock:
                sensory_current = self._network.inject_observation(
                    response_text, provenance="cognitive.response.validated"
                )
                if sensory_current is not None:
                    sensory_current *= gain
                    self._network.step(sensory_current)

            self.logger.info(
                f"Injected validated cognitive response ({len(response_text)} chars, "
                f"gain={gain}x, step={data.get('query_step', '?')})"
            )
        except Exception as e:
            self.logger.error(f"Failed to inject cognitive response: {e}")

    async def _handle_motor_outcome(self, data: dict[str, Any]) -> None:
        """Re-inject motor outcome as proprioceptive feedback with gain.

        This closes the sensorimotor loop: the brain fires a motor command,
        the actuator/sensor/simulator reports what happened, and this handler
        injects the consequence as proprioceptive input so STDP can learn
        which motor patterns produce desired outcomes.

        Expected payload:
            {
                "channel": "locomotion" | "manipulation" | "head",
                "success": true/false,
                "confidence": 0.0-1.0,
                "proprioceptive_state": [float, ...],  # optional: raw sensor data
                "error_magnitude": float,               # optional: how far from target
            }

        Feedback sources:
            - IMU/gyro: balance state after locomotion (motor.outcome.locomotion)
            - Camera: visual change confirming manipulation (motor.outcome.manipulation)
            - Joint encoders: target position reached (motor.outcome.manipulation)
            - Force sensors: grip success (motor.outcome.manipulation)
            - Simulator: ground truth (motor.outcome.{channel})
            - Human teacher: explicit good/bad (motor.outcome.{channel})
        """
        if not self._network:
            return

        channel = data.get("channel", "")
        if not channel:
            return

        _VALID_MOTOR_CHANNELS = {
            "locomotion", "manipulation", "head",
            "expression", "speech", "cognitive",
        }
        if channel not in _VALID_MOTOR_CHANNELS:
            self.logger.warning("Invalid motor outcome channel: %s", channel)
            return

        success = data.get("success", True)
        confidence = data.get("confidence", 0.5)

        mfb = self._config.motor_feedback
        gain = mfb.success_gain if success else mfb.failure_gain
        # Scale gain by confidence — uncertain outcomes have weaker effect
        gain *= confidence

        # Use raw proprioceptive state if provided, otherwise encode outcome
        # as a simple scalar signal
        proprio_data = data.get("proprioceptive_state")
        if proprio_data is None:
            # Encode success/failure as a scalar: 1.0 = full success, 0.0 = failure
            proprio_data = [1.0 if success else 0.0]

        provenance = f"motor.outcome.{channel}"
        # Queue the outcome — the simulation loop will drain it on the next
        # iteration while holding _net_lock.  This avoids blocking here for
        # the duration of a full sim step (~1.6s at 1M neurons).
        self._pending_motor_outcomes.append({
            "proprio_data": proprio_data,
            "provenance": provenance,
            "gain": gain,
            "channel": channel,
            "success": success,
        })

        # Outcome-contingent DA: success -> DA burst, failure -> DA dip.
        # This makes three-factor learning (STDP * eligibility * DA)
        # selectively strengthen motor patterns that produced good outcomes.
        if success:
            self._outcome_da_multiplier = mfb.outcome_da_success
        else:
            self._outcome_da_multiplier = mfb.outcome_da_failure
        self._outcome_da_steps_remaining = mfb.outcome_da_steps

        # Log at DEBUG for task-generated outcomes (high frequency, ~5 Hz),
        # INFO for user/actuator-initiated outcomes
        is_task = data.get("task_name") is not None
        log_fn = self.logger.debug if is_task else self.logger.info
        log_fn(
            f"Motor outcome queued: {channel} success={success} "
            f"confidence={confidence:.2f} gain={gain:.2f}x"
        )

    async def _handle_teach_da(self, data: dict[str, Any]) -> None:
        """Handle DA boost from teach mode.

        The motor feedback adapter publishes ``neuromod.teach.da`` during each
        teach repetition.  We record a synthetic motor echo so the existing
        motor-echo DA boost logic in ``_step_in_thread`` keeps eligibility
        traces alive for the full echo window (~100 steps).  This ensures
        three-factor learning (STDP * eligibility * DA) fires correctly.
        """
        if not self._network:
            return

        channel = data.get("channel", "locomotion")
        step_count = self._network.step_count

        # Record motor echo — same mechanism as normal motor fire.
        # The _step_in_thread loop checks this dict and applies DA boost
        # while any echo is within the echo_window_steps.
        self._motor_echo_steps[channel] = step_count
        self._motor_echo_ts[channel] = time.monotonic()

        self.logger.info(
            "Teach DA boost: channel=%s at step %d (echo window=%d steps)",
            channel, step_count,
            self._config.motor_feedback.echo_window_steps,
        )

    async def _handle_body_posture(self, data: dict[str, Any]) -> None:
        """Cache latest body posture for continuous height DA and standing pattern.

        Published at proprio rate (~5 Hz) by the motor feedback adapter.
        The brain step loop reads these cached values for:
        - Continuous DA modulation proportional to height (C2)
        - Standing pattern injection during supported phase (C6)
        - Motor homeostasis boost during supported phase (C7)
        """
        self._body_height_ratio = data.get("height_ratio", 0.0)
        self._body_task_name = data.get("task_name", "")
        self._body_support_active = data.get("support_active", False)
        self._body_standing_torques = data.get("standing_torques", {})

    # ===== Background Loops =====

    def _step_in_thread(
        self,
        batch: dict[str, Any],
        pain_max_snapshot: float = 0.0,
        posture_snapshot: tuple[float, str, bool, dict[str, float]] | None = None,
    ) -> tuple[dict[str, Any], int, dict[str, Any]]:
        """Run inject + network.step() synchronously (called from thread executor).

        This is the CPU-heavy work (~1.6s at 1M neurons).  Running it in a
        thread keeps the asyncio event loop free so the NATS client can
        continuously drain incoming messages instead of buffering them until
        the step completes.

        Returns (result_dict, step_count, drives_state).
        Watchdog status is embedded in result_dict["_watchdog_status"] when triggered.
        """
        network = self._network
        sensory_current = None
        if batch:
            if len(batch) == 1:
                prov, data = next(iter(batch.items()))
                sensory_current = network.inject_observation(data, prov)
            else:
                sensory_current = network.inject_multimodal(batch)

        # --- Merge queued motor outcomes into sensory current ---
        # Outcomes are merged additively into the main step's current
        # instead of triggering extra network.step() calls.  At 1M neurons
        # each step costs ~1.35s — extra steps would halve training speed.
        pending = self._pending_motor_outcomes
        self._pending_motor_outcomes = []
        for outcome in pending:
            sc = network.inject_observation(
                outcome["proprio_data"], provenance=outcome["provenance"]
            )
            sc *= outcome["gain"]
            if sensory_current is None:
                sensory_current = sc
            else:
                sensory_current = sensory_current + sc
            # Direct motor cortex teaching: fast corollary-discharge pathway.
            # Outcome reaches motor cortex in 1 step (not 4+ hops through
            # sensory -> association -> WM -> motor).
            channel = outcome.get("channel", "")
            if channel:
                network.inject_motor_teaching(
                    channel, outcome.get("success", True), outcome["gain"],
                )

        # Compute combined DA multiplier for this step.
        # All DA modulations are combined into a single multiplier passed to
        # network.step(), which applies it AFTER neuromodulation.update()
        # (which resets DA to baseline).  This ensures the modulation actually
        # affects plasticity_multiplier and eligibility trace gating.
        mfb = self._config.motor_feedback
        da_multiplier = 1.0

        # Pain-induced DA penalty (nociceptive feedback).
        # When joints are near limits, reduce DA to weaken the motor pattern.
        if mfb.enabled and mfb.pain_enabled and pain_max_snapshot > 0.1:
            pain = pain_max_snapshot
            # Scale DA: penalty=0.3 means DA is multiplied by 0.3 at max pain
            da_multiplier *= 1.0 - pain * (1.0 - mfb.pain_da_penalty)

        # Motor echo DA boost: when a motor command recently fired, boost DA
        # so eligibility traces on motor pathways stay alive during the
        # feedback delay (until proprioceptive feedback arrives).
        # Two modes: step-based (default, for MuJoCo) or wall-clock (for real hardware).
        use_wallclock = mfb.echo_window_ms > 0
        if mfb.enabled and self._motor_echo_steps:
            # Snapshot to avoid RuntimeError from concurrent dict mutation
            # (_handle_teach_da can write from the event loop while we iterate)
            if use_wallclock:
                now_mono = time.monotonic()
                window_s = mfb.echo_window_ms / 1000.0
                ts_snapshot = dict(self._motor_echo_ts)
                any_active = any(
                    (now_mono - ts) < window_s for ts in ts_snapshot.values()
                )
            else:
                step_now = network.step_count
                echo_snapshot = dict(self._motor_echo_steps)
                any_active = any(
                    (step_now - fire_step) < mfb.echo_window_steps
                    for fire_step in echo_snapshot.values()
                )
            if any_active:
                da_multiplier *= mfb.echo_da_boost

        # Outcome-contingent DA: modulate DA based on motor success/failure.
        # Multiplicative with echo and pain.  Decays to 1.0 after N steps.
        if self._outcome_da_steps_remaining > 0:
            da_multiplier *= self._outcome_da_multiplier
            self._outcome_da_steps_remaining -= 1

        # Unpack posture snapshot (taken on event loop, race-free).
        _p_height, _p_task, _p_support, _p_torques = posture_snapshot or (0.0, "", False, {})

        # Continuous height-proportional DA (C2): every step, DA reflects how
        # upright the body is.  Provides gradient signal for motor learning
        # even when sparse task outcomes aren't firing.
        if mfb.enabled and mfb.continuous_height_da and _p_height > 0:
            # Range: [0.5, 2.0].  Upright (1.0) -> 2.0x, fallen (0.0) -> 0.5x
            height_da = 0.5 + 1.5 * min(_p_height, 1.0)
            da_multiplier *= height_da

        # Standing pattern injection (C6): during supported phase, inject
        # PD-computed ideal torques as weak motor cortex current.
        if mfb.enabled and _p_support and _p_torques:
            network.inject_standing_pattern(_p_torques, mfb.standing_pattern_gain)

        # Motor homeostasis boost (C7): during supported phase, scale motor
        # synapse weights more aggressively to break saturation.
        if mfb.enabled and _p_task == "supported_stand":
            network._motor_scaling_boost = mfb.motor_homeostasis_boost

        # Persistent governor: inject inhibitory current into over-firing
        # regions EVERY step (not just on watchdog check step).  Without
        # persistence, single-step corrections are immediately overwhelmed
        # by the driving current on the next step.
        if self._persistent_governor:
            self._apply_governor_corrections(self._persistent_governor)

        _sensory_gap = network.step_count - self._last_obs_step
        result = network.step(sensory_current, da_multiplier=da_multiplier,
                              sensory_gap=_sensory_gap)
        # Clear expired echo channels
        if mfb.enabled and self._motor_echo_steps:
            if use_wallclock:
                now_mono = time.monotonic()
                window_s = mfb.echo_window_ms / 1000.0
                ts_snapshot = dict(self._motor_echo_ts)
                expired = [ch for ch, ts in ts_snapshot.items() if (now_mono - ts) > window_s]
            else:
                step_now = network.step_count
                echo_snapshot = dict(self._motor_echo_steps)
                expired = [
                    ch for ch, fire_step in echo_snapshot.items()
                    if (step_now - fire_step) > mfb.echo_window_steps
                ]
            for ch in expired:
                self._motor_echo_steps.pop(ch, None)
                self._motor_echo_ts.pop(ch, None)

        # Watchdog: check inside the thread for consistent state reads.
        step_count = network.step_count
        if step_count % self._watchdog.check_interval == 0:
            sensory_gap = step_count - self._last_obs_step
            wd_status = self._watchdog.check(network, step_count, sensory_gap)
            result["_watchdog_status"] = wd_status
            if wd_status.level >= AlertLevel.WARNING:
                self._persistent_governor = self._watchdog.get_governor_corrections(network)
            else:
                self._persistent_governor = {}

        return result, step_count, network.drives.get_state()

    async def _simulation_loop(self) -> None:
        """Main simulation loop — processes observations through the network.

        Each iteration:
        1. Atomically swap _latest_obs to get the most recent observation
           per modality (written by _handle_observation callbacks).
        2. Run inject + step in a thread executor so the event loop stays
           responsive and NATS can drain continuously.
        3. Emit motor proposals, cognitive queries, etc.
        """
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                try:
                    # Atomically grab the latest observations.  New observations
                    # that arrive during the step will go into the fresh dict.
                    snapshot = self._latest_obs
                    self._latest_obs = {}

                    # Build batch: provenance → data (same format as before)
                    batch: dict[str, Any] = {}
                    for modality, obs in snapshot.items():
                        provenance = obs.get("provenance", "")
                        data = obs.get("data", obs)
                        batch[provenance] = data

                    # Zero-observation watchdog: warn if no sensory input for
                    # extended periods.  All 6 patent learning mechanisms need
                    # sensory-driven spikes — running on noise wastes compute.
                    step_now = self._network.step_count
                    if batch:
                        if self._obs_warning_logged:
                            silent_gap = step_now - self._last_obs_step
                            logger.info(
                                "Sensory input restored at step %d "
                                "(was silent for %d steps)",
                                step_now, silent_gap,
                            )
                            self._obs_warning_logged = False
                            self._gateway_restart_requested = False
                        self._last_obs_step = step_now
                    elif step_now - self._last_obs_step >= 100 and not self._obs_warning_logged:
                        logger.warning(
                            "No sensory observations for %d steps -- "
                            "check gateway/NATS pipeline",
                            step_now - self._last_obs_step,
                        )
                        self._obs_warning_logged = True

                    # After prolonged starvation, ask the gateway to restart.
                    # Published once per starvation episode (reset when data returns).
                    starvation_gap = step_now - self._last_obs_step
                    if (starvation_gap >= self._gateway_restart_threshold
                            and not self._gateway_restart_requested):
                        self._gateway_restart_requested = True
                        logger.warning(
                            "Sensory starvation for %d steps -- "
                            "publishing gateway.restart.request",
                            starvation_gap,
                        )
                        try:
                            await self.event_bus.publish(
                                "gateway.restart.request",
                                {"step": step_now, "gap": starvation_gap},
                            )
                        except Exception as e:
                            logger.error("Failed to publish gateway.restart.request: %s", e)

                    # Run the heavy compute in a thread — event loop stays free
                    # for NATS message draining during the entire ~1.6s step.
                    # Snapshot pain and posture on the event loop to avoid race
                    # with NATS callbacks writing these from async handlers.
                    pain_snap = self._latest_pain_max
                    self._latest_pain_max = 0.0
                    posture_snap = (
                        self._body_height_ratio,
                        self._body_task_name,
                        self._body_support_active,
                        self._body_standing_torques,
                    )
                    async with self._net_lock:
                        result, step_count, drives_state = await loop.run_in_executor(
                            self._step_executor,
                            self._step_in_thread,
                            batch,
                            pain_snap,
                            posture_snap,
                        )

                    # Process outputs (no lock needed — using captured snapshots)
                    for cmd in result.get("motor_commands", []):
                        await self._emit_motor_proposal(cmd, step_count)
                        # Motor echo: record per-channel fire time.  The echo
                        # tracker sustains a DA boost so eligibility traces on
                        # motor pathways stay alive until proprioceptive
                        # feedback arrives (50-500ms later).
                        if self._config.motor_feedback.enabled:
                            ch = cmd.get("channel", "unknown")
                            self._motor_echo_steps[ch] = step_count
                            self._motor_echo_ts[ch] = time.monotonic()

                    for reflex in result.get("reflex_responses", []):
                        await self._emit_reflex_proposal(reflex)

                    for cog_cmd in result.get("cognitive_commands", []):
                        await self._emit_cognitive_query(cog_cmd, step_count, drives_state)

                    for speech_cmd in result.get("speech_commands", []):
                        await self._emit_speech_output(speech_cmd, step_count)

                    pred_error = result.get("prediction_error", 0.0)
                    if pred_error > 0.7:
                        await self._emit_knowledge_gap(pred_error, step_count, drives_state)

                    # Neural watchdog: status computed inside _step_in_thread
                    # for consistent reads.  Just handle NATS publishing here.
                    wd_status = result.pop("_watchdog_status", None)
                    if wd_status is not None:
                        if wd_status.level >= AlertLevel.WARNING:
                            await self._publish_watchdog_status(wd_status)
                        if wd_status.level >= AlertLevel.CRITICAL:
                            await self._escalate_watchdog(wd_status)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(f"Simulation step error: {e}", exc_info=True)
                    await asyncio.sleep(0.1)  # brief pause on error

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Simulation loop fatal error: {e}", exc_info=True)

    async def _drive_update_loop(self) -> None:
        """Update homeostatic drives periodically."""
        try:
            while self._running:
                try:
                    async with self._net_lock:
                        self._network.drives.update(self._config.drive_update_interval_s)
                        is_critical = self._network.drives.is_critical()
                        summary = self._network.drives.summary() if is_critical else ""
                        step = self._network.step_count

                    if is_critical:
                        await self.event_bus.publish("beliefs.add_node", {
                            "id": f"drive-state-{step}",
                            "type": "fact",
                            "content": f"Critical drive state: {summary}",
                            "confidence": 0.9,
                            "source": "neuromorphic.drives",
                        })
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(f"Drive update error: {e}", exc_info=True)

                await asyncio.sleep(self._config.drive_update_interval_s)
        except asyncio.CancelledError:
            pass

    async def _metrics_loop(self) -> None:
        """Publish network metrics periodically."""
        consecutive_errors = 0
        self.logger.info("Metrics loop started (interval=%ss)", self._config.metrics_interval_s)
        try:
            while self._running:
                try:
                    if self._network:
                        async with self._net_lock:
                            metrics = self._network.get_metrics()
                        await self.event_bus.publish("neuromorphic.metrics", metrics)
                        # Log step timing breakdown every 5th metrics emission (~50s)
                        timing = metrics.get("step_timing_ms")
                        if timing and consecutive_errors == 0:
                            self._timing_log_counter += 1
                            if self._timing_log_counter % 5 == 0:
                                total = timing.get("total", 0)
                                parts = " | ".join(
                                    f"{k}={v:.0f}ms" for k, v in timing.items() if k != "total"
                                )
                                logger.info("Step timing: total=%.0fms | %s", total, parts)
                                # Log per-region firing rates alongside timing
                                fr = metrics.get("firing_rates")
                                if fr:
                                    fr_parts = " | ".join(
                                        f"{r}={v:.1%}" for r, v in sorted(fr.items())
                                    )
                                    logger.info("Firing rates: %s", fr_parts)
                        consecutive_errors = 0
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= 3 or consecutive_errors % 30 == 0:
                        self.logger.error(
                            "Metrics loop error (#%d): %s", consecutive_errors, e,
                            exc_info=True,
                        )
                    if consecutive_errors > 3:
                        await asyncio.sleep(10.0)
                        continue
                await asyncio.sleep(self._config.metrics_interval_s)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error("Metrics loop FATAL: %s", e, exc_info=True)

    async def _save_loop(self) -> None:
        """Periodically save network state using fork-based background save.

        The fork happens *while* ``_net_lock`` is held so the child
        inherits a frozen snapshot of the network arrays via OS
        copy-on-write — no explicit ``.copy()`` needed.  The parent
        releases the lock immediately after ``fork()`` returns, so
        training pauses for only ~1 ms (the cost of the fork syscall).

        The child process writes .npy files and SQLite metadata, then
        exits via ``os._exit()``.
        """
        try:
            while self._running:
                await asyncio.sleep(self._config.save_interval_s)
                try:
                    if self._network and self._persistence:
                        async with self._net_lock:
                            # Build state dict with zero-copy refs to live arrays.
                            # Safe because fork() is called while the lock is held,
                            # so no other coroutine can mutate the arrays between
                            # get_state_zerocopy() and fork().
                            state = self._network.get_state_zerocopy()
                            # Include service-level state (tiny JSON, no arrays)
                            state["auditory_stm"] = self._auditory_stm.get_state()
                            state["watchdog"] = self._watchdog.get_state()
                            # Fork while locked — child gets frozen COW snapshot
                            self._persistence.save_background(state)
                        # Lock released — parent resumes training immediately.
                        # Child process has its own COW copy of all arrays.
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(f"Save loop error: {e}", exc_info=True)
        except asyncio.CancelledError:
            pass

    # ===== Output Emitters =====

    async def _emit_motor_proposal(self, cmd: dict[str, Any], step_count: int) -> None:
        """Convert a motor command into a Kernel-gated ActionProposal.

        When the safety gate is enabled, the proposal is published to
        ``proposal.new`` and the Kernel's decision is awaited asynchronously.
        DENY and TRANSFORM decisions inject proprioceptive feedback so the
        brain learns safe motor patterns through STDP.

        When the safety gate is disabled, proposals are published fire-and-forget
        (backward compatible — original behavior).
        """
        trace_id = generate_trace_id()
        proposal = {
            "trace_id": trace_id,
            "provenance": "neuromorphic.motor",
            "action": {
                "type": "motor_command",
                "channel": cmd["channel"],
                "intensity": cmd["intensity"],
            },
            "priority": 5,
            "metadata": {
                "source": "neuromorphic",
                "step": step_count,
            },
        }
        try:
            if not self._config.safety_gate.enabled:
                # Fire-and-forget (original behavior)
                await self.event_bus.publish("proposal.new", proposal)
                # Route to virtual body / real actuator for feedback
                if self._motor_adapter:
                    await self._motor_adapter.handle_motor_command(
                        channel=cmd["channel"],
                        intensity=cmd["intensity"],
                        trace_id=trace_id,
                        actuator_intensities=cmd.get("actuator_intensities"),
                    )
                return

            # Prevent unbounded growth if Kernel is unresponsive
            if len(self._pending_decisions) >= 500:
                self.logger.warning("Pending decisions at capacity (500) — dropping proposal")
                return

            # Publish first; only register Future if publish succeeds.
            # This avoids cleanup races in the exception handler.
            await self.event_bus.publish("proposal.new", proposal)

            # Safety gate ON — register a Future so the decision handler can
            # resolve it.  The sim loop does NOT block; a background task
            # handles the decision asynchronously.
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._pending_decisions[trace_id] = fut
            self._pending_proposals[trace_id] = proposal

            # Track the task so _cleanup can cancel it on shutdown
            task = asyncio.create_task(self._await_safety_decision(trace_id, fut, cmd))
            self._decision_tasks.add(task)
            task.add_done_callback(self._decision_tasks.discard)

        except Exception as e:
            self.logger.warning(f"Failed to publish motor proposal: {e}")

    async def _await_safety_decision(
        self,
        trace_id: str,
        fut: asyncio.Future,
        cmd: dict[str, Any],
    ) -> None:
        """Await a Kernel decision and inject feedback for DENY/TRANSFORM.

        Runs as a background task — does NOT block the simulation loop.
        """
        sg = self._config.safety_gate
        try:
            decision = await asyncio.wait_for(fut, timeout=sg.decision_timeout)
        except asyncio.TimeoutError:
            if sg.fail_open:
                self.logger.debug(f"Safety gate timeout for {trace_id} — fail-open, allowing")
            else:
                self.logger.warning(f"Safety gate timeout for {trace_id} — fail-closed, injecting deny")
                await self._inject_safety_feedback(cmd["channel"], success=False, confidence=0.5)
            return
        except asyncio.CancelledError:
            return
        finally:
            self._pending_decisions.pop(trace_id, None)
            self._pending_proposals.pop(trace_id, None)

        decision_type = decision.get("type", "ALLOW")

        if decision_type == "ALLOW":
            self.logger.debug(f"Safety ALLOW: {cmd['channel']} intensity={cmd['intensity']:.2f}")
            # Route to virtual body / real actuator for feedback
            if self._motor_adapter:
                await self._motor_adapter.handle_motor_command(
                    channel=cmd["channel"],
                    intensity=cmd["intensity"],
                    trace_id=trace_id,
                    actuator_intensities=cmd.get("actuator_intensities"),
                )

        elif decision_type == "DENY" and sg.deny_feedback:
            # Inject NEGATIVE feedback — brain learns this pattern is unsafe
            self.logger.info(
                f"Safety DENY: {cmd['channel']} intensity={cmd['intensity']:.2f} "
                f"reason={decision.get('reason', 'unknown')}"
            )
            await self._inject_safety_feedback(cmd["channel"], success=False, confidence=1.0)

        elif decision_type == "TRANSFORM" and sg.transform_feedback:
            # Inject CORRECTED feedback — brain learns the safe output range
            transformations = decision.get("transformations") or []
            corrected_intensity = cmd["intensity"]
            for t in transformations:
                if isinstance(t, dict) and "action" in t:
                    corrected_intensity = t["action"].get("intensity", corrected_intensity)
            self.logger.info(
                f"Safety TRANSFORM: {cmd['channel']} {cmd['intensity']:.2f} → {corrected_intensity:.2f}"
            )
            await self._inject_safety_feedback(
                cmd["channel"],
                success=True,
                confidence=0.7,
                proprio_data=[corrected_intensity],
            )

        elif decision_type == "DEFER":
            # Publish approval request for the dashboard human-in-the-loop UI.
            # A human operator will ALLOW or DENY via approval.response.{trace_id}.
            self.logger.info(f"Safety DEFER: {cmd['channel']} — requesting human approval")
            try:
                await self.event_bus.publish("approval.request", {
                    "trace_id": trace_id,
                    "channel": cmd["channel"],
                    "intensity": cmd["intensity"],
                    "reason": decision.get("reason", "Elevated risk — requires human approval"),
                    "risk_score": decision.get("risk_score", 0.0),
                })
            except Exception as e:
                self.logger.warning(f"Failed to publish approval request: {e}")

    async def _inject_safety_feedback(
        self,
        channel: str,
        success: bool,
        confidence: float = 1.0,
        proprio_data: list[float] | None = None,
    ) -> None:
        """Inject a safety decision as proprioceptive motor feedback.

        This reuses the existing motor feedback loop — DENY becomes a negative
        outcome that R-STDP uses to weaken the synapses that produced the
        unsafe motor command.
        """
        if not self._config.motor_feedback.enabled:
            if self._config.safety_gate.enabled:
                self.logger.warning(
                    "Safety gate DENY/TRANSFORM feedback lost — motor_feedback is disabled. "
                    "Enable NEURO_MOTOR_FEEDBACK=1 for safety learning to work."
                )
            return

        outcome = {
            "channel": channel,
            "success": success,
            "confidence": confidence,
            "error_magnitude": 0.0 if success else 1.0,
            "source": "safety_gate",
        }
        if proprio_data is not None:
            outcome["proprioceptive_state"] = proprio_data

        # Queue via the existing motor outcome handler
        self._pending_motor_outcomes.append({
            "proprio_data": proprio_data if proprio_data is not None else [1.0 if success else 0.0],
            "provenance": f"motor.outcome.{channel}",
            "gain": (
                self._config.motor_feedback.success_gain if success
                else self._config.motor_feedback.failure_gain
            ) * confidence,
            "channel": channel,
            "success": success,
        })

    async def _handle_kernel_decision(self, data: dict[str, Any]) -> None:
        """Handle Kernel decisions arriving on decision.{trace_id}."""
        trace_id = data.get("trace_id", "")
        fut = self._pending_decisions.get(trace_id)
        if fut is None:
            # Stale or already-timed-out decision — harmless, just ignore
            return
        if fut.done():
            self.logger.debug(f"Decision for already-completed trace {trace_id} — ignoring")
            return
        fut.set_result(data)

    async def _handle_approval_response(self, data: dict[str, Any]) -> None:
        """Handle human approval decisions from the dashboard.

        Arrives on approval.response.{trace_id}.  The human clicks
        ALLOW or DENY in the dashboard; we inject the corresponding
        proprioceptive feedback so the brain learns from the decision.
        """
        trace_id = data.get("trace_id", "")
        channel = data.get("channel", "")
        approved = data.get("approved", False)

        if not trace_id or not channel:
            self.logger.warning(f"Approval response missing trace_id or channel: {data}")
            return

        if approved:
            self.logger.info(f"Human APPROVED deferred action: {channel} (trace={trace_id})")
            # No negative feedback — the action will proceed and produce real outcome
        else:
            self.logger.info(f"Human DENIED deferred action: {channel} (trace={trace_id})")
            if self._config.safety_gate.deny_feedback:
                await self._inject_safety_feedback(channel, success=False, confidence=0.8)

    async def _emit_reflex_proposal(self, reflex: dict[str, Any]) -> None:
        """Emit a high-priority reflex action proposal."""
        try:
            await self.event_bus.publish("proposal.new", {
                "trace_id": generate_trace_id(),
                "provenance": "neuromorphic.reflex",
                "action": reflex["action"],
                "priority": reflex["priority"],
                "metadata": {
                    "source": "neuromorphic.reflex",
                    "reflex": reflex["name"],
                },
            })
        except Exception as e:
            self.logger.warning(f"Failed to publish reflex proposal: {e}")

    async def _emit_cognitive_query(self, cmd: dict[str, Any], step_count: int, drives_state: dict) -> None:
        """Emit a cognitive query — the brain is asking for help.

        All cognitive queries route through the Kernel via proposal.new.
        The Kernel forwards ALLOWED queries to cognitive.execute, which
        the CognitiveBridgeService subscribes to.
        """
        action_type = cmd.get("type", "query_llm")
        try:
            if action_type == "query_llm":
                await self.event_bus.publish("proposal.new", {
                    "trace_id": generate_trace_id(),
                    "provenance": "neuromorphic.cognitive",
                    "action": {
                        "type": "cognitive_query",
                        "query_type": action_type,
                        "intensity": cmd.get("intensity", 0.0),
                        "prediction_error": cmd.get("prediction_error", 0.0),
                    },
                    "priority": 3,
                    "metadata": {
                        "source": "neuromorphic.cognitive",
                        "step": step_count,
                        "query_number": cmd.get("query_number", 0),
                        "drives": drives_state,
                    },
                })
            elif action_type == "save_memory":
                await self.event_bus.publish("memory.store", {
                    "trace_id": generate_trace_id(),
                    "type": "cognitive_save",
                    "description": "Brain-initiated memory consolidation",
                    "step": step_count,
                })
        except Exception as e:
            self.logger.warning(f"Failed to emit cognitive query: {e}")

    async def _emit_speech_output(self, cmd: dict[str, Any], step_count: int) -> None:
        """Emit a speech output event — brain-native language production.

        Routes through the Kernel via proposal.new. On ALLOW, the Kernel
        forwards to speech.execute for downstream consumers (TTS actuator,
        dashboard display, logging).
        """
        try:
            await self.event_bus.publish("proposal.new", {
                "trace_id": generate_trace_id(),
                "provenance": "neuromorphic.speech",
                "action": {
                    "type": "speech_output",
                    "channel": "speech",
                    "token_idx": cmd.get("token_idx", 0),
                    "confidence": cmd.get("confidence", 0.0),
                    "intensity": cmd.get("intensity", 0.0),
                    "output_number": cmd.get("output_number", 0),
                },
                "metadata": {
                    "source": "neuromorphic.speech",
                    "step": step_count,
                },
            })
        except Exception as e:
            self.logger.warning(f"Failed to publish speech proposal: {e}")

    async def _emit_knowledge_gap(self, prediction_error: float, step_count: int, drives_state: dict) -> None:
        """Signal a knowledge gap when prediction error is high."""
        try:
            await self.event_bus.publish("knowledge.gap", {
                "trace_id": generate_trace_id(),
                "description": f"High prediction error ({prediction_error:.2f}) — world model surprised",
                "context": {
                    "prediction_error": prediction_error,
                    "step": step_count,
                    "drives": drives_state,
                },
                "confidence": 1.0 - prediction_error,
                "source": "neuromorphic.predictive",
            })
        except Exception as e:
            self.logger.warning(f"Failed to publish knowledge gap: {e}")

    async def _publish_watchdog_status(self, status: WatchdogStatus) -> None:
        """Publish watchdog alerts to NATS for dashboard / cloud monitoring."""
        try:
            level_name = status.level.name
            status_dict = status.to_dict()
            await self.event_bus.publish("safety.watchdog.status", status_dict)

            # Log locally based on severity
            if status.level >= AlertLevel.CRITICAL:
                for alert in status.alerts:
                    self.logger.error(f"WATCHDOG {alert.level.name}: {alert.message}")
            elif status.level >= AlertLevel.WARNING:
                for alert in status.alerts:
                    self.logger.warning(f"WATCHDOG {alert.level.name}: {alert.message}")
        except Exception as e:
            self.logger.warning(f"Failed to publish watchdog status: {e}")

    def _apply_governor_corrections(self, corrections: dict[str, float]) -> None:
        """Inject inhibitory current into over-firing regions.

        This is homeostatic regulation (Patent Claim 1e) — the watchdog
        detects over-activity and injects negative current to stabilize
        firing rates. It does NOT modify weights (Claim 6 safe).
        """
        if not self._network:
            return
        import numpy as np
        for region_name, gain in corrections.items():
            region = self._network.regions.get(region_name)
            if region is not None:
                # Inject uniform inhibitory current across all neurons
                inhibitory = np.full(region.n, gain, dtype=np.float32)
                region.inject_current(inhibitory)
                self.logger.debug(
                    f"Governor: injected {gain:.2f} inhibitory to '{region_name}'"
                )

    async def _escalate_watchdog(self, status: WatchdogStatus) -> None:
        """Automated response to CRITICAL/EMERGENCY watchdog alerts.

        CRITICAL: Request motor intensity reduction via Kernel policy.
        EMERGENCY: Request full motor halt via Kernel policy.

        Cooldown: escalation fires at most once per 1000 steps to prevent
        policy thrashing when alerts persist across multiple watchdog checks.

        This is homeostatic regulation (Claim 1e) — the brain
        self-regulates through the Kernel, not by modifying its own weights.
        """
        # Cooldown: skip if we escalated recently (prevents NATS flood)
        escalation_cooldown = 1000  # steps
        if status.step - self._last_escalation_step < escalation_cooldown:
            return
        self._last_escalation_step = status.step

        try:
            has_motor_spam = any(
                a.check_name == "motor_spam" for a in status.alerts
            )
            if status.level >= AlertLevel.EMERGENCY:
                # Request full motor halt — all channels to 0
                await self.event_bus.publish("policy.restrict", {
                    "motor_limits": {
                        ch: {"max_intensity": 0.0}
                        for ch in ("locomotion", "manipulation", "head", "speech")
                    },
                    "reason": f"EMERGENCY watchdog escalation at step {status.step}",
                    "operator_id": "system:watchdog",
                })
                self.logger.error(
                    f"WATCHDOG EMERGENCY: requested motor halt at step {status.step}"
                )
            elif has_motor_spam:
                # Motor spam: halve motor intensity
                await self.event_bus.publish("policy.restrict", {
                    "motor_limits": {
                        "locomotion": {"max_intensity": 0.5},
                        "manipulation": {"max_intensity": 0.5},
                    },
                    "reason": f"Motor spam detected at step {status.step}",
                    "operator_id": "system:watchdog",
                })
                self.logger.warning(
                    f"WATCHDOG CRITICAL: motor spam — requested intensity reduction"
                )
        except Exception as e:
            self.logger.warning(f"Failed to escalate watchdog: {e}")


async def main() -> None:
    """Main entry point."""
    service = NeuromorphicService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
