"""
Sensory Gateway — discovers sensors and streams data to the brain via NATS.

Runs on the host Mac (not Docker) because it needs hardware access.
Auto-discovers cameras, microphones, and serial devices, then starts
SensorPlugin instances that publish observations to NATS.

Also supports video file training: dashboard can submit YouTube URLs or
local paths via NATS commands. The gateway downloads, creates
VideoFileSensor + AudioFileSensor + optional TranscriptSensor, and
tracks training session progress.

Usage:
    python gateway.py                       # discover and start all sensors
    python gateway.py --no-camera           # skip camera
    python gateway.py --no-mic              # skip microphone
    python gateway.py --stt                 # enable speech-to-text (requires faster-whisper)
    python gateway.py --nats nats://host:4222  # custom NATS URL
    python gateway.py --list                # list available sensors and exit
    python gateway.py --video baby_words.mp4 --video-loop  # play video file
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
import uuid
from pathlib import Path

# Limit ffmpeg decoder threads inside OpenCV VideoCapture.
# Default: each capture creates ~79 OS threads (one per CPU core for decoding).
# With 442 video sensors that's 34K+ threads, hitting vm.max_map_count.
# Setting to "1" makes ffmpeg decode on the calling thread — zero extra threads.
# At 2 fps per sensor, single-thread decode (~10-50ms) is well within budget.
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")

import cv2  # noqa: E402 — must import after setting env vars above

cv2.setNumThreads(2)  # Limit OpenCV parallel_for pool (default = CPU count = 64)

from activelearning.nats_client import EventBus  # noqa: E402
from activelearning.plugins import register_sensor  # noqa: E402

from aggregating_bus import AggregatingEventBus  # noqa: E402
from dataset_manifest import DatasetManifest  # noqa: E402
from discovery import KNOWN_DEVICE_TYPES, DiscoveredDevice, discover_all  # noqa: E402
from queue_state import QueueStateManager  # noqa: E402
from video_quality import VideoQualityGate  # noqa: E402

logger = logging.getLogger("sensory-gateway")

# NATS subjects for dashboard <-> gateway communication
SUBJECT_COMMAND = "sensory.gateway.command"
SUBJECT_STATUS = "sensory.gateway.status"
SUBJECT_VIDEO_STATUS = "video.training.status"
SUBJECT_NEURO_METRICS = "neuromorphic.metrics"
SUBJECT_DEVICE_UNKNOWN = "device.unknown"
SUBJECT_DRIVER_READY = "device.driver.ready"

# Key plastic synapse groups that indicate learning from sensory input
_LEARNING_SYNAPSE_GROUPS = {
    "sensory_association",
    "sensory_motor",
    "sensory_cerebellum",
    "association_lateral",
    "sensory_feature",
    "feature_association",
    "association_concept",
}


class ConvergenceTracker:
    """Tracks STDP weight change magnitude to detect when learning has stabilized.

    Monitors the mean |dw| across key synapse groups. When it drops below
    a threshold for N consecutive metric windows, the content is considered
    'learned' and video training can auto-stop.
    """

    def __init__(
        self,
        threshold: float = 0.0005,
        window: int = 5,
        min_steps: int = 500,
    ):
        self.threshold = threshold  # mean |dw| below this = stable
        self.window = window  # consecutive stable checks needed
        self.min_steps = min_steps  # don't check before this many steps
        self._stable_count: int = 0
        self._last_mean_delta: float = 1.0
        self._step_at_start: int = 0
        self._converged: bool = False

    def reset(self, current_step: int = 0) -> None:
        """Reset tracker for a new training session."""
        self._stable_count = 0
        self._last_mean_delta = 1.0
        self._step_at_start = current_step
        self._converged = False

    def update(self, metrics: dict) -> bool:
        """Update with neuromorphic metrics. Returns True if converged."""
        if self._converged:
            return True

        step_count = metrics.get("step_count", 0)
        if step_count - self._step_at_start < self.min_steps:
            return False

        stdp_deltas = metrics.get("stdp_deltas", {})
        if not stdp_deltas:
            return False

        # Average delta across key learning connections only
        relevant = [v for k, v in stdp_deltas.items() if k in _LEARNING_SYNAPSE_GROUPS and v > 0]
        if not relevant:
            return False

        self._last_mean_delta = sum(relevant) / len(relevant)

        if self._last_mean_delta < self.threshold:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if self._stable_count >= self.window:
            self._converged = True
            logger.info(
                f"Convergence detected: mean STDP delta {self._last_mean_delta:.6f} "
                f"< {self.threshold} for {self.window} consecutive checks"
            )
            return True

        return False

    @property
    def status(self) -> dict:
        return {
            "converged": self._converged,
            "mean_delta": round(self._last_mean_delta, 6),
            "stable_count": self._stable_count,
            "window": self.window,
            "threshold": self.threshold,
        }


class VideoTrainingSession:
    """Tracks a video training session (video + audio + optional transcript sensors)."""

    def __init__(
        self,
        session_id: str,
        filepath: str,
        title: str,
        duration_s: float,
        fps: float,
        loop: bool,
        transcript: bool,
        url: str | None = None,
        target_loops: int = 0,
        category: str = "",
    ):
        self.session_id = session_id
        self.filepath = filepath
        self.title = title
        self.duration_s = duration_s
        self.fps = fps
        self.loop = loop
        self.transcript = transcript
        self.url = url
        self.target_loops = target_loops  # 0 = infinite
        self.category = category
        self.status = (
            "pending"  # pending, downloading, starting, running, completed, error, stopped, queued
        )
        self.error: str | None = None
        self.video_sensor = None
        self.audio_sensor = None
        self.transcript_sensor = None
        self.created_at = time.time()
        self.started_at: float = 0.0
        self.completed_at: float = 0.0
        self.convergence = ConvergenceTracker()
        # Per-video learning metrics
        self.learning_score: float = 0.0  # cumulative STDP delta during this video
        self.stdp_samples: int = 0  # how many metric windows we've seen
        self._quality_result: dict | None = None

    def update_learning_metrics(self, metrics: dict) -> None:
        """Update per-video learning score from neuromorphic metrics."""
        stdp_deltas = metrics.get("stdp_deltas", {})
        relevant = [v for k, v in stdp_deltas.items() if k in _LEARNING_SYNAPSE_GROUPS and v > 0]
        if relevant:
            mean_delta = sum(relevant) / len(relevant)
            self.learning_score += mean_delta
            self.stdp_samples += 1

    @property
    def avg_learning_rate(self) -> float:
        """Average STDP delta per metric window for this video."""
        return self.learning_score / self.stdp_samples if self.stdp_samples > 0 else 0.0

    def get_status(self) -> dict:
        """Build status dict for NATS broadcast."""
        result = {
            "session_id": self.session_id,
            "title": self.title,
            "filepath": self.filepath,
            "url": self.url,
            "duration_s": round(self.duration_s, 1),
            "fps": self.fps,
            "loop": self.loop,
            "transcript": self.transcript,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "elapsed_s": round(time.time() - self.created_at, 1),
            "target_loops": self.target_loops,
            "category": self.category,
            "learning_score": round(self.learning_score, 6),
            "avg_learning_rate": round(self.avg_learning_rate, 6),
            "stdp_samples": self.stdp_samples,
        }
        if self.video_sensor and hasattr(self.video_sensor, "get_training_status"):
            vs = self.video_sensor.get_training_status()
            result["current_pos_s"] = vs["current_pos_s"]
            result["progress"] = vs["progress"]
            result["loop_count"] = vs["loop_count"]
            result["frames_emitted"] = vs["frames_emitted"]
        else:
            result["current_pos_s"] = 0
            result["progress"] = 0
            result["loop_count"] = 0
            result["frames_emitted"] = 0
        result["convergence"] = self.convergence.status
        return result


def create_sensor(device: DiscoveredDevice, args: argparse.Namespace):
    """Map a discovered device to its SensorPlugin implementation."""
    if device.device_type == "camera":
        if args.no_camera:
            return None
        from sensors.camera import CameraSensor

        return CameraSensor(
            device_index=device.metadata["index"],
            fps=args.camera_fps,
            cnn=args.cnn,
        )

    elif device.device_type == "mic":
        if args.no_mic:
            return None
        from sensors.microphone import MicrophoneSensor

        return MicrophoneSensor(
            device_index=device.metadata["index"],
            hz=args.audio_hz,
        )

    elif device.device_type == "serial":
        from sensors.serial_device import SerialSensor

        return SerialSensor(
            port=device.metadata["port"],
        )

    elif device.device_type == "imu":
        from sensors.imu import IMUSensor

        return IMUSensor(
            port=device.metadata["port"],
        )

    return None


async def run(args: argparse.Namespace) -> None:
    """Main gateway loop: discover, connect, start, stream."""
    # Discover available hardware (honor --no-camera/--no-mic to skip slow probes)
    devices = discover_all(
        skip_cameras=getattr(args, "no_camera", False),
        skip_microphones=getattr(args, "no_mic", False),
    )

    if args.list:
        print(f"\nDiscovered {len(devices)} sensor(s):\n")
        for d in devices:
            print(f"  [{d.device_type:8s}] {d.device_id:20s} — {d.name}")
            for k, v in d.metadata.items():
                print(f"             {k}: {v}")
        return

    has_video = bool(getattr(args, "video", None))
    if not devices and not has_video:
        logger.warning("No sensors discovered and no video files. Nothing to do.")
        return

    # Connect to NATS (use aggregating bus to reduce observation message rate)
    flush_hz = getattr(args, "aggregate_hz", 2.0)
    if getattr(args, "no_aggregate", False):
        bus = EventBus(nats_url=args.nats)
    else:
        bus = AggregatingEventBus(nats_url=args.nats, flush_hz=flush_hz)
    await bus.connect()
    logger.info(f"Connected to NATS at {args.nats}")

    # Create and start sensors
    active_sensors = []
    sensor_state: dict[str, dict] = {}
    unknown_published: set[str] = set()  # device_ids already published as unknown

    for device in devices:
        if device.device_type not in KNOWN_DEVICE_TYPES:
            # Unknown device — publish for coordinator/meta-programmer pipeline
            if device.device_id not in unknown_published:
                await bus.publish(
                    SUBJECT_DEVICE_UNKNOWN,
                    {
                        "device_type": device.device_type,
                        "device_id": device.device_id,
                        "name": device.name,
                        "metadata": device.metadata,
                    },
                )
                unknown_published.add(device.device_id)
                logger.info(f"Unknown device published: {device.device_id} ({device.name})")
            continue

        sensor = create_sensor(device, args)
        if sensor is None:
            continue
        try:
            register_sensor(sensor)
            await sensor.start(bus)
            active_sensors.append(sensor)
            sensor_state[sensor.sensor_id] = {"sensor": sensor, "running": True}
            logger.info(f"Started: {sensor.sensor_id} ({sensor.name})")
        except Exception as e:
            logger.error(f"Failed to start {device.device_id}: {e}")

    # Optionally start speech-to-text (uses the same mic, separate plugin)
    if args.stt:
        try:
            from sensors.speech import SpeechSensor

            speech = SpeechSensor(model_size=args.stt_model)
            register_sensor(speech)
            await speech.start(bus)
            active_sensors.append(speech)
            sensor_state[speech.sensor_id] = {"sensor": speech, "running": True}
            logger.info(f"Started: {speech.sensor_id} (Whisper {args.stt_model})")
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install sensory-gateway[stt]")
        except Exception as e:
            logger.error(f"Failed to start speech sensor: {e}")

    # ── Video training queue + sessions ──
    video_sessions: dict[str, VideoTrainingSession] = {}
    video_queue: list[str] = []  # ordered list of session_ids waiting to play
    _queue_active_id: str | None = None  # session_id currently playing via queue
    _queue_runner_task: asyncio.Task | None = None
    _queue_advance_event = asyncio.Event()
    _queue_state = QueueStateManager()
    _quality_gate = VideoQualityGate()
    _manifest = DatasetManifest()
    _manifest.start_run(f"run_{int(time.time())}")

    def _save_queue_state() -> None:
        """Persist current queue state to disk."""
        _queue_state.save(video_sessions, video_queue, _queue_active_id)

    # Resume from saved state if available
    _resumable = _queue_state.has_resumable()
    if _resumable and has_video:
        logger.warning(
            f"Found {len(_resumable)} resumable session(s) but --video was passed. "
            f"Saved queue state ignored. Delete {_queue_state.path} to suppress this warning."
        )
    if _resumable and not has_video:
        logger.info(f"Found {len(_resumable)} resumable video session(s) from previous run")
        for entry in _resumable:
            sid = entry.get("session_id", uuid.uuid4().hex[:8])
            filepath = entry.get("filepath", "")
            if not filepath or (
                not Path(filepath).exists() and not (entry.get("url") or "").startswith("http")
            ):
                logger.warning(f"Skipping resume of {entry.get('title', sid)}: file not found")
                continue
            session = VideoTrainingSession(
                session_id=sid,
                filepath=filepath,
                title=entry.get("title", Path(filepath).stem),
                duration_s=entry.get("duration_s", 0.0),
                fps=entry.get("fps", args.video_fps),
                loop=True,
                transcript=entry.get("transcript", False),
                url=entry.get("url"),
                target_loops=entry.get("target_loops", 5),
                category=entry.get("category", ""),
            )
            session.status = "queued"
            video_sessions[sid] = session
            video_queue.append(sid)
            logger.info(f"  Resumed: {session.title} ({session.target_loops} loops)")
        if video_queue:
            _queue_advance_event.set()

    async def _start_video_session(
        filepath: str,
        title: str,
        duration_s: float,
        fps: float,
        loop: bool,
        transcript: bool,
        url: str | None = None,
        target_loops: int = 0,
        category: str = "",
    ) -> VideoTrainingSession:
        """Create and start a video training session with its sensors."""
        session_id = uuid.uuid4().hex[:8]
        session = VideoTrainingSession(
            session_id=session_id,
            filepath=filepath,
            title=title,
            duration_s=duration_s,
            fps=fps,
            loop=loop,
            transcript=transcript,
            url=url,
            target_loops=target_loops,
            category=category,
        )
        video_sessions[session_id] = session
        session.status = "starting"
        session.started_at = time.time()

        try:
            from sensors.audio_file import AudioFileSensor
            from sensors.video_file import VideoFileSensor

            # Video sensor
            vsensor = VideoFileSensor(
                filepath,
                fps=fps,
                loop=loop,
                session_id=session_id,
                cnn=args.cnn,
                turbo=args.turbo,
                sparse_threshold=args.sparse_threshold,
            )
            register_sensor(vsensor)
            await vsensor.start(bus)
            active_sensors.append(vsensor)
            sensor_state[vsensor.sensor_id] = {"sensor": vsensor, "running": True}
            session.video_sensor = vsensor
            logger.info(f"Video session {session_id}: started {vsensor.sensor_id}")

            # Audio sensor (optional — needs ffmpeg + sounddevice). If it can't
            # start, keep the visual feed running rather than failing the session.
            try:
                if AudioFileSensor is None:
                    raise ImportError("AudioFileSensor unavailable (sounddevice not installed)")
                asensor = AudioFileSensor(
                    filepath, hz=10.0, loop=loop, session_id=session_id, turbo=args.turbo
                )
                register_sensor(asensor)
                await asensor.start(bus)
                active_sensors.append(asensor)
                sensor_state[asensor.sensor_id] = {"sensor": asensor, "running": True}
                session.audio_sensor = asensor
                logger.info(f"Video session {session_id}: started {asensor.sensor_id}")
            except Exception as e:
                logger.warning(
                    f"Video session {session_id}: audio sensor disabled ({e}). "
                    "Visual input only — install ffmpeg + sounddevice for audio."
                )

            # Optional transcript sensor
            if transcript:
                try:
                    from sensors.transcript import TranscriptSensor

                    tsensor = TranscriptSensor(
                        filepath,
                        model_size=args.stt_model,
                        loop=loop,
                        session_id=session_id,
                    )
                    register_sensor(tsensor)
                    await tsensor.start(bus)
                    active_sensors.append(tsensor)
                    sensor_state[tsensor.sensor_id] = {"sensor": tsensor, "running": True}
                    session.transcript_sensor = tsensor
                    logger.info(f"Video session {session_id}: started {tsensor.sensor_id}")
                except ImportError:
                    logger.warning("faster-whisper not installed, skipping transcript")
                except Exception as e:
                    logger.warning(f"Transcript sensor failed: {e}")

            session.status = "running"
            return session

        except Exception as e:
            session.status = "error"
            session.error = str(e)
            logger.error(f"Video session {session_id} failed: {e}")
            return session

    async def _stop_video_session(session_id: str) -> None:
        """Stop all sensors in a video training session and clean up."""
        session = video_sessions.get(session_id)
        if not session:
            return

        for sensor in [session.video_sensor, session.audio_sensor, session.transcript_sensor]:
            if sensor is not None:
                sid = sensor.sensor_id
                state = sensor_state.get(sid)
                if state and state["running"]:
                    try:
                        await sensor.stop()
                    except Exception as e:
                        logger.error(f"Error stopping {sid}: {e}")
                # Remove from tracking lists
                sensor_state.pop(sid, None)
                try:
                    active_sensors.remove(sensor)
                except ValueError:
                    pass

        session.status = "stopped"
        logger.info(f"Video session {session_id} stopped")

    async def _download_for_queue(entry: VideoTrainingSession) -> bool:
        """Download video for a queued entry. Returns True on success."""
        if not entry.url or not entry.url.startswith("http"):
            return True  # local file, nothing to download
        entry.status = "downloading"
        await bus.publish(SUBJECT_VIDEO_STATUS, entry.get_status())
        try:
            from scripts.download_video import download_video_async

            logger.info(f"Queue: downloading {entry.url}")
            info = await download_video_async(entry.url)
            entry.filepath = info["filepath"]
            entry.title = info.get("title", entry.title)
            entry.duration_s = info.get("duration", 0)
            return True
        except Exception as e:
            entry.status = "error"
            entry.error = str(e)
            logger.error(f"Queue: download failed for {entry.url}: {e}")
            await bus.publish(SUBJECT_VIDEO_STATUS, entry.get_status())
            return False

    def _get_queue_status() -> dict:
        """Build queue state for NATS broadcast."""
        queue_items = []
        for sid in video_queue:
            s = video_sessions.get(sid)
            if s:
                queue_items.append(s.get_status())
        active = None
        if _queue_active_id:
            s = video_sessions.get(_queue_active_id)
            if s:
                active = s.get_status()
        completed = []
        for s in video_sessions.values():
            if s.status in ("completed", "stopped", "error") and s.session_id not in video_queue:
                completed.append(s.get_status())
        completed.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return {
            "queue": queue_items,
            "active": active,
            "completed": completed[:20],
            "queue_length": len(video_queue),
        }

    async def _queue_runner():
        """Main queue loop: play one video at a time, advance when target loops reached."""
        nonlocal _queue_active_id
        while True:
            # Wait for something in the queue
            while not video_queue:
                _queue_active_id = None
                await bus.publish(
                    SUBJECT_VIDEO_STATUS,
                    {
                        "type": "queue_update",
                        **_get_queue_status(),
                    },
                )
                _queue_advance_event.clear()
                await _queue_advance_event.wait()

            session_id = video_queue[0]
            session = video_sessions.get(session_id)
            if not session:
                video_queue.pop(0)
                continue

            _queue_active_id = session_id

            # Download if needed
            if session.url and session.url.startswith("http") and session.status == "queued":
                ok = await _download_for_queue(session)
                if not ok:
                    video_queue.pop(0)
                    _queue_active_id = None
                    continue

            # Run quality gate on the video file (in executor — OpenCV I/O is blocking)
            loop = asyncio.get_event_loop()
            qr = await loop.run_in_executor(None, _quality_gate.validate, session.filepath)
            session._quality_result = qr
            if not qr["valid"]:
                logger.warning(
                    f"Queue: skipping {session.title} — quality check failed: "
                    + "; ".join(qr["issues"])
                )
                session.status = "error"
                session.error = "Quality check: " + "; ".join(qr["issues"])
                await bus.publish(SUBJECT_VIDEO_STATUS, session.get_status())
                video_queue.pop(0)
                _queue_active_id = None
                _save_queue_state()
                continue
            if qr["metadata"].get("is_duplicate"):
                logger.info(f"Queue: {session.title} is a duplicate (proceeding anyway)")
            # Pass pre-computed hash to avoid re-opening the video file
            _quality_gate.register_hash(
                session.filepath,
                session.session_id,
                video_hash=qr.get("metadata", {}).get("hash"),
            )

            # Start the session (loop=True so it keeps going, we monitor loop_count)
            try:
                started = await _start_video_session(
                    filepath=session.filepath,
                    title=session.title,
                    duration_s=session.duration_s,
                    fps=session.fps,
                    loop=True,  # always loop, queue controls when to stop
                    transcript=session.transcript,
                    url=session.url,
                    target_loops=session.target_loops,
                    category=session.category,
                )
                # The _start_video_session creates a new session object; update our tracking
                video_queue[0] = started.session_id
                _queue_active_id = started.session_id
                # Carry over quality result from the placeholder session
                started._quality_result = session._quality_result
                # Remove the placeholder and replace with the real session
                video_sessions.pop(session_id, None)

                await bus.publish(
                    SUBJECT_VIDEO_STATUS,
                    {
                        "type": "queue_update",
                        **_get_queue_status(),
                    },
                )
            except Exception as e:
                logger.error(f"Queue: failed to start {session.title}: {e}")
                session.status = "error"
                session.error = str(e)
                video_queue.pop(0)
                _queue_active_id = None
                continue

            # Reset convergence tracker with current step count
            step_now = _latest_neuro_metrics.get("step_count", 0)
            started.convergence.reset(step_now)

            # Monitor loop count + convergence
            target = started.target_loops
            poll_interval = 0.5 if args.turbo else 2.0
            while True:
                await asyncio.sleep(poll_interval)

                # Check if session was stopped externally or skipped
                if started.status in ("stopped", "error"):
                    break
                if _queue_active_id != started.session_id:
                    break  # was skipped

                # Check if target loops reached
                if target > 0 and started.video_sensor:
                    current_loops = started.video_sensor.loop_count
                    if current_loops >= target:
                        logger.info(
                            f"Queue: {started.title} reached {current_loops}/{target} loops, advancing"
                        )
                        started.status = "completed"
                        started.completed_at = time.time()
                        await _stop_video_session(started.session_id)
                        break

                # Convergence auto-advance: when brain has learned this video, move on
                if started.convergence._converged and not getattr(
                    started, "_convergence_announced", False
                ):
                    loop_count = started.video_sensor.loop_count if started.video_sensor else 0
                    logger.info(
                        f"Convergence auto-advance: {started.title} after {loop_count} loops "
                        f"(mean delta: {started.convergence._last_mean_delta:.6f}, "
                        f"learning score: {started.learning_score:.6f})"
                    )
                    await bus.publish(
                        SUBJECT_VIDEO_STATUS,
                        {
                            "type": "converged",
                            "session_id": started.session_id,
                            "title": started.title,
                            "loop_count": loop_count,
                            "learning_score": round(started.learning_score, 6),
                            **started.convergence.status,
                        },
                    )
                    started._convergence_announced = True
                    # Auto-advance: stop this video and move to next
                    started.status = "completed"
                    started.completed_at = time.time()
                    await _stop_video_session(started.session_id)
                    break

            # Record to dataset manifest
            loop_count = started.video_sensor.loop_count if started.video_sensor else 0
            frames = started.video_sensor.frames_emitted if started.video_sensor else 0
            video_hash = ""
            qr = getattr(started, "_quality_result", None)
            if qr and qr.get("metadata"):
                video_hash = qr["metadata"].get("hash", "")
            _manifest.record_video(
                session_id=started.session_id,
                filepath=started.filepath,
                title=started.title,
                url=started.url,
                category=started.category,
                target_loops=started.target_loops,
                completed_loops=loop_count,
                frames_emitted=frames,
                learning_score=started.learning_score,
                avg_learning_rate=started.avg_learning_rate,
                quality_result=qr,
                video_hash=video_hash,
                status=started.status,
                duration_s=started.duration_s,
                elapsed_s=round(time.time() - started.created_at, 1),
            )

            # Remove from front of queue and advance
            if video_queue and video_queue[0] == started.session_id:
                video_queue.pop(0)
            _queue_active_id = None
            _save_queue_state()
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "queue_update",
                    **_get_queue_status(),
                },
            )

    # Start video files from CLI args (batched to avoid thundering herd)
    if args.video:
        batch_size = args.batch_size
        batch_delay = args.batch_delay
        total = len(args.video)
        logger.info(
            f"Starting {total} video sensor(s) in batches of {batch_size} "
            f"({batch_delay}s delay between batches)"
        )
        for batch_start in range(0, total, batch_size):
            batch = args.video[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            logger.info(f"Batch {batch_num}/{total_batches}: starting {len(batch)} sensor(s)")
            for path in batch:
                filepath = path
                title = Path(path).stem if not path.startswith("http") else path
                duration_s = 0.0

                # Download if URL
                if path.startswith("http"):
                    try:
                        from scripts.download_video import download_video_async

                        logger.info(f"Downloading: {path}")
                        info = await download_video_async(path)
                        filepath = info["filepath"]
                        title = info.get("title", title)
                        duration_s = info.get("duration", 0)
                    except Exception as e:
                        logger.error(f"Failed to download {path}: {e}")
                        continue

                await _start_video_session(
                    filepath=filepath,
                    title=title,
                    duration_s=duration_s,
                    fps=args.video_fps,
                    loop=args.video_loop,
                    transcript=args.video_transcript,
                    url=path if path.startswith("http") else None,
                )
            # Let the batch settle before starting next
            if batch_start + batch_size < total:
                logger.info(f"Batch {batch_num} done, waiting {batch_delay}s before next batch...")
                await asyncio.sleep(batch_delay)

    if not active_sensors:
        logger.warning("No sensors started. Nothing to do.")
        await bus.close()
        return

    # Log summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Sensory Gateway running with {len(active_sensors)} sensor(s):")
    for s in active_sensors:
        meta = s.describe()
        logger.info(f"  {meta.sensor_id:20s} @ {meta.rate_limit_hz:.0f} Hz — {meta.name}")
    logger.info(f"Publishing to NATS at {args.nats}")
    logger.info(f"{'='*50}\n")

    # ── Status publisher: broadcast gateway state every 3 seconds ──
    async def _build_status() -> dict:
        sensors_info = []
        for sid, state in sensor_state.items():
            s = state["sensor"]
            meta = s.describe()
            sensors_info.append(
                {
                    "sensor_id": sid,
                    "name": meta.name,
                    "type": meta.description,
                    "hz": meta.rate_limit_hz,
                    "running": state["running"],
                }
            )
        return {
            "gateway": "running",
            "timestamp": time.time(),
            "sensors": sensors_info,
        }

    async def _publish_status_loop():
        while True:
            try:
                status = await _build_status()
                await bus.publish(SUBJECT_STATUS, status)
            except Exception as e:
                logger.debug(f"Status publish error: {e}")
            await asyncio.sleep(3)

    # ── Command handler: respond to dashboard commands ──
    async def _handle_command(data: dict) -> None:
        action = data.get("action")
        sensor_id = data.get("sensor_id")
        logger.info(f"Command received: {action} {sensor_id or '(all)'}")

        if action == "stop" and sensor_id:
            state = sensor_state.get(sensor_id)
            if state and state["running"]:
                try:
                    await state["sensor"].stop()
                    state["running"] = False
                    logger.info(f"Stopped sensor: {sensor_id}")
                except Exception as e:
                    logger.error(f"Error stopping {sensor_id}: {e}")

        elif action == "start" and sensor_id:
            state = sensor_state.get(sensor_id)
            if state and not state["running"]:
                try:
                    await state["sensor"].start(bus)
                    state["running"] = True
                    logger.info(f"Restarted sensor: {sensor_id}")
                except Exception as e:
                    logger.error(f"Error restarting {sensor_id}: {e}")

        elif action == "stop_all":
            for sid, state in sensor_state.items():
                if state["running"]:
                    try:
                        await state["sensor"].stop()
                        state["running"] = False
                    except Exception as e:
                        logger.error(f"Error stopping {sid}: {e}")
            logger.info("All sensors stopped")

        elif action == "start_all":
            for sid, state in sensor_state.items():
                if not state["running"]:
                    try:
                        await state["sensor"].start(bus)
                        state["running"] = True
                    except Exception as e:
                        logger.error(f"Error starting {sid}: {e}")
            logger.info("All sensors started")

        elif action == "queue_video":
            # Add video to the training queue (sequential playback)
            url_or_path = data.get("url", "")
            fps = data.get("fps", args.video_fps)
            transcript = data.get("transcript", False)
            target_loops = int(data.get("target_loops", 5))
            category = data.get("category", "")

            if not url_or_path:
                return

            # Check for duplicate in queue or active
            for sid in video_queue:
                s = video_sessions.get(sid)
                if s and (s.url or s.filepath) == url_or_path:
                    logger.warning(f"Video already in queue: {url_or_path}")
                    return

            # Check blacklist for local files (run in executor — compute_hash does I/O)
            if not url_or_path.startswith("http"):
                _loop = asyncio.get_event_loop()
                if await _loop.run_in_executor(None, _quality_gate.is_blacklisted, url_or_path):
                    logger.warning(f"Video is blacklisted: {url_or_path}")
                    await bus.publish(
                        SUBJECT_VIDEO_STATUS,
                        {
                            "type": "queue_rejected",
                            "url": url_or_path,
                            "reason": "Video is blacklisted",
                        },
                    )
                    return

            # Create a placeholder session in "queued" state
            session_id = uuid.uuid4().hex[:8]
            title = url_or_path
            if not url_or_path.startswith("http"):
                title = Path(url_or_path).stem
            session = VideoTrainingSession(
                session_id=session_id,
                filepath=url_or_path,
                title=title,
                duration_s=0.0,
                fps=fps,
                loop=True,
                transcript=transcript,
                url=url_or_path if url_or_path.startswith("http") else None,
                target_loops=target_loops,
                category=category,
            )
            session.status = "queued"
            video_sessions[session_id] = session
            video_queue.append(session_id)
            logger.info(f"Queued video: {title} ({target_loops} loops)")

            # Wake up the queue runner
            _queue_advance_event.set()
            _save_queue_state()

            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "queue_update",
                    **_get_queue_status(),
                },
            )

        elif action == "skip_video":
            # Skip the currently playing video, advance to next in queue
            if _queue_active_id:
                session = video_sessions.get(_queue_active_id)
                if session and session.status == "running":
                    session.status = "stopped"
                    await _stop_video_session(_queue_active_id)
                    logger.info(f"Skipped video: {session.title}")
            _save_queue_state()
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "queue_update",
                    **_get_queue_status(),
                },
            )

        elif action == "remove_queued":
            # Remove a specific video from the queue (not the active one)
            session_id = data.get("session_id", "")
            if session_id in video_queue and session_id != _queue_active_id:
                video_queue.remove(session_id)
                s = video_sessions.get(session_id)
                if s:
                    s.status = "stopped"
                logger.info(f"Removed from queue: {session_id}")
            _save_queue_state()
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "queue_update",
                    **_get_queue_status(),
                },
            )

        elif action == "clear_queue":
            # Clear all queued (not active) videos
            kept = []
            for sid in video_queue:
                if sid == _queue_active_id:
                    kept.append(sid)
                else:
                    s = video_sessions.get(sid)
                    if s:
                        s.status = "stopped"
            video_queue.clear()
            video_queue.extend(kept)
            logger.info("Queue cleared")
            _save_queue_state()
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "queue_update",
                    **_get_queue_status(),
                },
            )

        elif action == "reorder_queue":
            # Reorder the queue: move session_id to position
            session_id = data.get("session_id", "")
            position = int(data.get("position", 0))
            if session_id in video_queue:
                video_queue.remove(session_id)
                video_queue.insert(min(position, len(video_queue)), session_id)
            _save_queue_state()
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "queue_update",
                    **_get_queue_status(),
                },
            )

        elif action == "blacklist_video":
            # Blacklist a video — stop if active, remove from queue, add to blacklist
            target_sid = data.get("session_id", "")
            reason = data.get("reason", "Blacklisted by user")
            # If the requested session_id isn't found (old placeholder was swapped),
            # fall back to the currently active session
            session = video_sessions.get(target_sid)
            if not session and _queue_active_id:
                session = video_sessions.get(_queue_active_id)
                if session:
                    target_sid = _queue_active_id
            if session:
                _quality_gate.blacklist(session.filepath, reason)
                if target_sid == _queue_active_id:
                    session.status = "stopped"
                    await _stop_video_session(target_sid)
                if target_sid in video_queue:
                    video_queue.remove(target_sid)
                session.status = "error"
                session.error = f"Blacklisted: {reason}"
                logger.info(f"Blacklisted video: {session.title} — {reason}")
            else:
                # Blacklist by filepath/url directly
                filepath = data.get("filepath", "")
                if filepath:
                    _quality_gate.blacklist(filepath, reason)
            _save_queue_state()
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "queue_update",
                    **_get_queue_status(),
                },
            )

        elif action == "unblacklist_video":
            video_hash = data.get("hash", "")
            if video_hash:
                _quality_gate.unblacklist(video_hash)
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "blacklist_update",
                    "blacklist": _quality_gate.get_blacklist(),
                },
            )

        elif action == "get_blacklist":
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "blacklist_update",
                    "blacklist": _quality_gate.get_blacklist(),
                },
            )

        elif action == "get_manifest":
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "manifest_update",
                    "runs": _manifest.list_runs(),
                },
            )

        elif action == "save_manifest":
            path = _manifest.save()
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "manifest_saved",
                    "path": str(path),
                },
            )

        elif action == "add_video":
            # Direct play (bypass queue) — kept for backwards compat
            url_or_path = data.get("url", "")
            fps = data.get("fps", args.video_fps)
            loop = data.get("loop", True)
            transcript = data.get("transcript", False)

            # Check for duplicate
            for existing in video_sessions.values():
                if existing.status in ("running", "starting", "downloading"):
                    existing_key = existing.url or existing.filepath
                    if existing_key == url_or_path:
                        logger.warning(f"Video already active: {url_or_path}")
                        return

            async def _add_video_bg():
                filepath = url_or_path
                title = url_or_path
                duration_s = 0.0

                if url_or_path.startswith("http"):
                    try:
                        from scripts.download_video import download_video_async

                        logger.info(f"Downloading video: {url_or_path}")
                        await bus.publish(
                            SUBJECT_VIDEO_STATUS,
                            {
                                "session_id": "pending",
                                "status": "downloading",
                                "title": url_or_path,
                                "url": url_or_path,
                            },
                        )
                        info = await download_video_async(url_or_path)
                        filepath = info["filepath"]
                        title = info.get("title", title)
                        duration_s = info.get("duration", 0)
                    except Exception as e:
                        logger.error(f"Failed to download {url_or_path}: {e}")
                        await bus.publish(
                            SUBJECT_VIDEO_STATUS,
                            {
                                "session_id": "download_error",
                                "status": "error",
                                "error": str(e),
                                "url": url_or_path,
                            },
                        )
                        return
                else:
                    title = Path(filepath).stem

                session = await _start_video_session(
                    filepath=filepath,
                    title=title,
                    duration_s=duration_s,
                    fps=fps,
                    loop=loop,
                    transcript=transcript,
                    url=url_or_path if url_or_path.startswith("http") else None,
                )
                await bus.publish(SUBJECT_VIDEO_STATUS, session.get_status())

            asyncio.create_task(_add_video_bg())

        elif action == "stop_video":
            session_id = data.get("session_id", "")
            await _stop_video_session(session_id)
            session = video_sessions.get(session_id)
            if session:
                await bus.publish(SUBJECT_VIDEO_STATUS, session.get_status())

        elif action == "get_queue":
            # Dashboard requests full queue state
            await bus.publish(
                SUBJECT_VIDEO_STATUS,
                {
                    "type": "queue_update",
                    **_get_queue_status(),
                },
            )

        elif action == "status":
            pass  # Just triggers an immediate status publish below

        # Publish updated status immediately after any command
        try:
            status = await _build_status()
            await bus.publish(SUBJECT_STATUS, status)
        except Exception as e:
            logger.debug(f"Status publish error: {e}")

    # ── Video training status publisher ──
    async def _publish_video_status_loop():
        while True:
            try:
                # Publish individual running session status
                for session in video_sessions.values():
                    if session.status in ("running", "starting"):
                        await bus.publish(SUBJECT_VIDEO_STATUS, session.get_status())
                # Publish queue state
                if video_queue or _queue_active_id:
                    await bus.publish(
                        SUBJECT_VIDEO_STATUS,
                        {
                            "type": "queue_update",
                            **_get_queue_status(),
                        },
                    )
            except Exception as e:
                logger.debug(f"Video status publish error: {e}")
            await asyncio.sleep(3)

    # ── Neuromorphic metrics handler: feed convergence tracker ──
    _latest_neuro_metrics: dict = {}

    async def _handle_neuro_metrics(data: dict) -> None:
        nonlocal _latest_neuro_metrics
        _latest_neuro_metrics = data
        # Feed metrics to all running video sessions' convergence + learning trackers
        for session in video_sessions.values():
            if session.status == "running":
                session.convergence.update(data)
                session.update_learning_metrics(data)

    # ── Hot-load handler: meta-programmer delivers new driver plugin ──
    async def _handle_driver_ready(data: dict) -> None:
        """Hot-load a newly generated sensor/actuator plugin."""
        target_path = data.get("target_path", "")
        device_id = data.get("device_id", "")
        plugin_type = data.get("plugin_type", "sensor")  # "sensor" or "actuator"
        init_kwargs = data.get("init_kwargs", {})

        if not target_path or not Path(target_path).exists():
            logger.error(f"Driver ready but file missing: {target_path}")
            return

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_hotloaded_plugin", target_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Find the Plugin subclass in the module
            from activelearning.plugins import ActuatorPlugin, SensorPlugin

            plugin_cls = None
            base = SensorPlugin if plugin_type == "sensor" else ActuatorPlugin
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and issubclass(attr, base) and attr is not base:
                    plugin_cls = attr
                    break

            if plugin_cls is None:
                logger.error(f"No {plugin_type} plugin class found in {target_path}")
                return

            plugin = plugin_cls(**init_kwargs)
            register_sensor(plugin)
            await plugin.start(bus)
            active_sensors.append(plugin)
            sid = getattr(plugin, "sensor_id", None) or getattr(plugin, "actuator_id", device_id)
            sensor_state[sid] = {"sensor": plugin, "running": True}
            logger.info(f"Hot-loaded plugin: {sid} from {target_path}")

        except Exception as e:
            logger.error(f"Failed to hot-load driver {target_path}: {e}")

    await bus.subscribe(SUBJECT_NEURO_METRICS, _handle_neuro_metrics)
    await bus.subscribe(SUBJECT_COMMAND, _handle_command)
    await bus.subscribe(SUBJECT_DRIVER_READY, _handle_driver_ready)
    status_task = asyncio.create_task(_publish_status_loop())
    video_status_task = asyncio.create_task(_publish_video_status_loop())
    _queue_runner_task = asyncio.create_task(_queue_runner())

    # Publish initial status
    try:
        status = await _build_status()
        await bus.publish(SUBJECT_STATUS, status)
    except Exception:
        pass

    # Run until interrupted
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutting down...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows: add_signal_handler unsupported; Ctrl+C still works

    try:
        await stop_event.wait()
    finally:
        status_task.cancel()
        video_status_task.cancel()
        if _queue_runner_task:
            _queue_runner_task.cancel()
        # Always clean up, even on unexpected exceptions
        for sid, state in sensor_state.items():
            if state["running"]:
                try:
                    await state["sensor"].stop()
                except Exception as e:
                    logger.error(f"Error stopping {sid}: {e}")

        # Save or clear queue state on shutdown
        if video_queue or _queue_active_id:
            _save_queue_state()
            logger.info("Queue state saved for resume on next startup")
        else:
            _queue_state.clear()

        # Save dataset manifest
        if _manifest._entries:
            manifest_path = _manifest.save()
            logger.info(f"Dataset manifest saved: {manifest_path}")

        await bus.close()
        logger.info("Sensory Gateway stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensory Gateway — stream sensor data to the neuromorphic brain",
    )
    parser.add_argument(
        "--nats",
        default="nats://localhost:4222",
        help="NATS server URL (default: nats://localhost:4222)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered sensors and exit",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip camera sensors",
    )
    parser.add_argument(
        "--no-mic",
        action="store_true",
        help="Skip microphone sensors",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=5.0,
        help="Camera capture rate in FPS (default: 5)",
    )
    parser.add_argument(
        "--audio-hz",
        type=float,
        default=30.0,
        help="Audio MFCC extraction rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        help="Enable speech-to-text via Whisper",
    )
    parser.add_argument(
        "--stt-model",
        default="tiny",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: tiny)",
    )
    _default_video = os.environ.get("SENSORY_VIDEO")
    parser.add_argument(
        "--video",
        nargs="+",
        default=[_default_video] if _default_video else None,
        help="Video file path(s) or YouTube URL(s) to play as training input "
        "(default: $SENSORY_VIDEO)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=2.0,
        help="Video playback FPS (default: 2)",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=0.05,
        help="Sparse encoding threshold — skip frames with mean pixel diff below this (default: 0.05)",
    )
    parser.add_argument(
        "--video-loop",
        action="store_true",
        help="Loop video playback",
    )
    parser.add_argument(
        "--video-transcript",
        action="store_true",
        help="Enable Whisper transcription of video audio",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        default=False,
        help="Turbo training mode: remove sleep between frames for max throughput",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of video sensors to start per batch (default: 25)",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between sensor startup batches (default: 2.0)",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        default=False,
        help="Enable CNN feature extraction (MobileNetV3) for visual sensors",
    )
    parser.add_argument(
        "--no-cnn",
        action="store_true",
        default=False,
        help="Disable CNN feature extraction (use raw 64x64 pixels)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        default=False,
        help="Disable observation aggregation (send every sensor message individually)",
    )
    parser.add_argument(
        "--aggregate-hz",
        type=float,
        default=2.0,
        help="Aggregation flush rate in Hz (default: 2.0 — matches brain step rate)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main_sync():
    """Sync entry point for console_scripts."""
    args = parse_args()
    # --no-cnn overrides --cnn
    if args.no_cnn:
        args.cnn = False
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main_sync()
