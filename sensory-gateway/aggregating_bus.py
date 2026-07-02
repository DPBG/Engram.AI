"""
AggregatingEventBus — reduces NATS message rate by batching sensor observations.

Problem: 403 video sensors × 2 fps + audio = ~1,400 msgs/sec, but the brain
processes ~0.6 steps/sec.  NATS buffers overflow and disconnect the brain.

Solution: Intercept observation publishes, keep only the latest per modality,
and flush a single fused message per modality on a timer.  This drops
1,400 msgs/sec → ~4 msgs/sec (1 visual + 1 auditory per flush @ 2 Hz).

Audio gets special treatment: instead of keeping only the LAST 32ms MFCC
chunk per flush window (which destroys temporal structure), we accumulate
ALL audio chunks and compute a temporal audio frame on flush:
  - mean MFCC (spectral centroid)
  - delta MFCC (first-order derivative — encodes transitions)
  - energy contour (amplitude envelope — word/syllable boundaries)
  - raw MFCC stack (complete temporal sequence)

This preserves the temporal information that STDP needs to learn phonemes,
words, and prosody — a single 32ms snapshot cannot represent these.

Non-observation publishes (status, commands, etc.) pass through immediately.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import is_dataclass
from typing import Any

import numpy as np
from activelearning.nats_client import EventBus

logger = logging.getLogger("sensory-gateway.aggregator")

# Map observation subject prefixes to modality buckets
_MODALITY_MAP = {
    "observation.videofile.": "visual",
    "observation.camera.": "visual",
    "observation.cam": "visual",
    "observation.audiofile.": "auditory",
    "observation.mic": "auditory",
    "observation.transcript": "auditory",
    "observation.text": "auditory",
    "observation.serial": "proprioceptive",
    "observation.imu": "proprioceptive",
}


def _classify_observation(subject: str) -> str | None:
    """Map a NATS subject to a modality bucket, or None if not an observation."""
    for prefix, modality in _MODALITY_MAP.items():
        if subject.startswith(prefix):
            return modality
    # Fallback: any observation.* → visual
    if subject.startswith("observation."):
        return "visual"
    return None


def compute_temporal_audio_frame(chunks: list[list[float]]) -> dict[str, Any]:
    """Compute a temporal audio frame from accumulated MFCC chunks.

    Given N chunks of 13 MFCC coefficients each, computes:
      - mean_mfcc:  [13] mean spectral profile across the window
      - delta_mfcc: [13] first-order derivative (transitions between phonemes)
      - energy:     [N]  per-chunk energy (amplitude envelope for word boundaries)
      - raw_stack:  [N×13] complete MFCC sequence (full temporal info)

    Returns a dict with these arrays as lists and metadata.
    """
    n = len(chunks)
    arr = np.array(chunks, dtype=np.float32)  # (N, 13)

    # Mean MFCC — spectral centroid of the window
    mean_mfcc = arr.mean(axis=0)  # (13,)

    # Delta MFCC — first-order derivative (encodes transitions)
    # For single-chunk windows, delta is zero
    if n >= 2:
        # Central differences for interior, forward/backward for edges
        delta = np.zeros_like(arr)
        delta[0] = arr[1] - arr[0]
        delta[-1] = arr[-1] - arr[-2]
        if n > 2:
            delta[1:-1] = (arr[2:] - arr[:-2]) / 2.0
        delta_mfcc = delta.mean(axis=0)  # (13,) average rate of change
    else:
        delta_mfcc = np.zeros(arr.shape[1], dtype=np.float32)

    # Energy contour — RMS of each chunk's MFCC (proxy for loudness)
    energy = np.sqrt((arr**2).mean(axis=1))  # (N,)

    return {
        "type": "temporal_audio_frame",
        "n_chunks": n,
        "mean_mfcc": mean_mfcc.tolist(),
        "delta_mfcc": delta_mfcc.tolist(),
        "energy": energy.tolist(),
        "raw_stack": arr.tolist(),
    }


class AggregatingEventBus(EventBus):
    """EventBus wrapper that batches observation publishes.

    Sensors call ``publish("observation.videofile.XXX", data)`` as before.
    Instead of sending each one to NATS, the aggregator stores the latest
    per modality and flushes at ``flush_hz`` (default 2 Hz).

    Audio modality uses temporal aggregation: all MFCC chunks within a
    flush window are accumulated and fused into a temporal audio frame
    with mean, delta, energy, and raw stack features.

    This is transparent to sensors (they don't know) and backward-compatible
    with the brain (temporal frames include a "type" field for detection).
    """

    def __init__(
        self,
        nats_url: str | None = None,
        name: str = "activelearning",
        flush_hz: float = 2.0,
        publish_watchdog_timeout: float = 30.0,
        heartbeat_interval: float = 10.0,
        heartbeat_timeout: float = 5.0,
    ):
        super().__init__(nats_url=nats_url, name=name)
        self._flush_interval = 1.0 / flush_hz if flush_hz > 0 else 0.5
        # Latest observation per modality (written by publish, read by flush)
        self._latest: dict[str, dict[str, Any]] = {}  # modality → data
        self._latest_subject: dict[str, str] = {}  # modality → subject
        # Audio temporal buffer — accumulates MFCC chunks between flushes.
        # Keyed per-subject so we don't mix audio from different sensors.
        # On flush, we pick the sensor with the most chunks in the window
        # (most temporal context), similar to how old code kept "latest".
        self._audio_buffers: dict[str, list[list[float]]] = {}  # subject → chunks
        self._audio_provenance: dict[str, str] = {}  # subject → provenance
        self._flush_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._stats_total = 0
        self._stats_flushed = 0
        self._stats_audio_chunks = 0
        self._stats_last_log = 0.0
        # Publish watchdog: if no successful publish in this many seconds
        # and we had data to send, trigger force_reconnect.
        self._publish_watchdog_timeout = publish_watchdog_timeout
        self._last_successful_publish = time.monotonic()
        self._reconnecting = False  # guard against concurrent reconnects
        # Heartbeat liveness: NATS PING/PONG roundtrip check
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._consecutive_heartbeat_failures = 0
        # Reconnect cooldown: minimum seconds between force_reconnect attempts
        # to prevent log spam and connection churn during sustained NATS outages
        self._reconnect_cooldown = 15.0
        self._last_reconnect_attempt = 0.0

    async def connect(self) -> None:
        await super().connect()
        self._last_successful_publish = time.monotonic()
        self._consecutive_heartbeat_failures = 0
        self._reconnecting = False
        # Only create tasks if they're not already running (prevents duplicates
        # when connect() is called after force_reconnect within a running loop)
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_loop())
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            f"Aggregating bus active: flush every {self._flush_interval:.1f}s "
            f"({1/self._flush_interval:.0f} Hz), temporal audio enabled, "
            f"heartbeat every {self._heartbeat_interval:.0f}s, "
            f"publish watchdog {self._publish_watchdog_timeout:.0f}s"
        )

    async def close(self) -> None:
        for task in (self._flush_task, self._heartbeat_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Final flush before close
        await self._flush_once()
        await super().close()

    async def publish(self, subject: str, data: Any) -> None:
        """Intercept observation publishes; pass-through everything else."""
        modality = _classify_observation(subject)
        if modality is None:
            # Non-observation — pass through immediately
            await super().publish(subject, data)
            return

        self._stats_total += 1

        if modality == "auditory":
            # Accumulate MFCC audio chunks for temporal frame computation.
            # Non-MFCC auditory data (transcripts, text) falls through to
            # the latest-only path so it's not silently dropped.
            #
            # `data` may be an Observation dataclass (from SensorPlugin.emit)
            # or a plain dict (from other sources). Extract the payload.
            if is_dataclass(data) and not isinstance(data, type):
                raw = data.data  # Observation.data
                provenance = getattr(data, "provenance", "")
            elif isinstance(data, dict):
                raw = data.get("data", data)
                provenance = data.get("provenance", "") if isinstance(data, dict) else ""
            else:
                raw = data
                provenance = ""

            if isinstance(raw, (list, tuple)) and len(raw) == 13:
                # Exactly 13 floats = MFCC chunk — accumulate per-sensor
                if subject not in self._audio_buffers:
                    self._audio_buffers[subject] = []
                self._audio_buffers[subject].append(list(raw))
                self._stats_audio_chunks += 1
                self._audio_provenance[subject] = provenance
            else:
                # Non-MFCC auditory (transcript, text, etc.) — keep latest only
                self._latest[modality] = data
                self._latest_subject[modality] = subject
        else:
            # Non-audio: keep only the latest (video, proprioceptive, etc.)
            self._latest[modality] = data
            self._latest_subject[modality] = subject

    async def _flush_once(self) -> None:
        """Publish the latest observation for each modality.

        Audio gets special treatment: accumulated MFCC chunks are fused
        into a temporal audio frame with mean/delta/energy/raw features.
        """
        has_non_audio = bool(self._latest)
        has_audio = bool(self._audio_buffers)

        if not has_non_audio and not has_audio:
            return

        # Guard: verify NATS is still connected before flushing.
        # Silent publish failures caused 148K empty brain steps (2026-03-13).
        if not self.is_connected:
            # Throttle to once every 15s — a sustained outage flushes every
            # _flush_interval (e.g. 0.5s) and would otherwise flood the console.
            now = time.monotonic()
            if now - getattr(self, "_last_disc_log_ts", 0.0) >= 15.0:
                self._last_disc_log_ts = now
                logger.error(
                    "Aggregator flush skipped — NATS disconnected. Buffered "
                    "observations are being dropped until NATS reconnects "
                    "(throttled; this repeats every 15s while disconnected)."
                )
            # Clear buffers to avoid unbounded growth while disconnected
            self._latest.clear()
            self._latest_subject.clear()
            self._audio_buffers.clear()
            self._audio_provenance.clear()
            return

        # Atomically swap out non-audio buffer
        to_send = self._latest
        subjects = self._latest_subject
        self._latest = {}
        self._latest_subject = {}

        # Atomically swap out audio buffers (per-sensor)
        audio_buffers = self._audio_buffers
        audio_provenance = self._audio_provenance
        self._audio_buffers = {}
        self._audio_provenance = {}

        # Pick the sensor with the most chunks (best temporal context)
        audio_chunks: list[list[float]] = []
        audio_subject: str = ""
        best_provenance: str = ""
        if audio_buffers:
            best_subject = max(audio_buffers, key=lambda s: len(audio_buffers[s]))
            audio_chunks = audio_buffers[best_subject]
            audio_subject = best_subject
            best_provenance = audio_provenance.get(best_subject, "")

        # Flush non-audio modalities (latest-only, as before)
        published_any = False
        for modality, data in to_send.items():
            subject = subjects.get(modality, f"observation.{modality}")
            try:
                await super().publish(subject, data)
                self._stats_flushed += 1
                published_any = True
            except Exception as e:
                logger.error(f"Flush error for {modality}: {e}")

        # Flush audio as temporal frame
        if audio_chunks:
            temporal_frame = compute_temporal_audio_frame(audio_chunks)
            # Wrap in the standard observation format
            audio_data = {
                "provenance": best_provenance or "sensor.audiofile",
                "data": temporal_frame,
            }
            subject = audio_subject or "observation.auditory"
            try:
                await super().publish(subject, audio_data)
                self._stats_flushed += 1
                published_any = True
            except Exception as e:
                logger.error(f"Flush error for auditory temporal frame: {e}")

        if published_any:
            self._last_successful_publish = time.monotonic()

    async def _try_force_reconnect(self, reason: str) -> None:
        """Attempt force_reconnect with guard against concurrent calls.

        Important: We do NOT call self.force_reconnect() because that calls
        self.close() which cancels our flush/heartbeat tasks (we're running
        INSIDE one of them). Instead, we manually tear down just the NATS
        connection and re-create it, preserving our running tasks.
        """
        if self._reconnecting:
            return
        # Cooldown: don't hammer NATS during sustained outages
        now = time.monotonic()
        if now - self._last_reconnect_attempt < self._reconnect_cooldown:
            return
        self._last_reconnect_attempt = now
        self._reconnecting = True
        try:
            logger.warning(f"Triggering force_reconnect: {reason}")
            # Save handler registry (force_reconnect logic, inlined)
            saved_handlers = dict(self._handlers)
            # Tear down NATS connection only (not our tasks)
            if self._nc is not None:
                try:
                    await asyncio.wait_for(self._nc.close(), timeout=5.0)
                except Exception as e:
                    logger.warning(f"NATS close during reconnect failed (expected): {e}")
                self._nc = None
                self._js = None
                self._connected.clear()
                self._subscriptions.clear()
            # Re-create connection (calls EventBus.connect, not our override)
            await EventBus.connect(self)
            # Re-subscribe saved handlers
            for subject, handler in saved_handlers.items():
                try:
                    await self.subscribe(subject, handler)
                except Exception as e:
                    logger.error(f"Re-subscribe failed for {subject}: {e}")
            self._last_successful_publish = time.monotonic()
            self._consecutive_heartbeat_failures = 0
            logger.info(f"Reconnect complete, {len(self._subscriptions)} subs restored")
        except Exception as e:
            logger.error(f"force_reconnect failed: {e}")
        finally:
            self._reconnecting = False

    async def _flush_loop(self) -> None:
        """Periodic flush of buffered observations."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                await self._flush_once()

                # Publish watchdog: if we had data but haven't successfully
                # published in too long, the connection is silently dead.
                now = time.monotonic()
                stale = now - self._last_successful_publish
                has_data = self._stats_total > 0
                if has_data and stale > self._publish_watchdog_timeout:
                    await self._try_force_reconnect(
                        f"publish watchdog: {stale:.0f}s since last successful publish"
                    )

                # Log stats every 60 seconds
                if now - self._stats_last_log > 60.0:
                    if self._stats_total > 0:
                        reduction = 1.0 - (self._stats_flushed / self._stats_total)
                        logger.info(
                            f"Aggregator: {self._stats_total} sensor obs -> "
                            f"{self._stats_flushed} NATS msgs "
                            f"({reduction:.1%} reduction), "
                            f"audio: {self._stats_audio_chunks} chunks aggregated"
                        )
                    self._stats_total = 0
                    self._stats_flushed = 0
                    self._stats_audio_chunks = 0
                    self._stats_last_log = now
        except asyncio.CancelledError:
            pass

    async def _heartbeat_loop(self) -> None:
        """Periodic NATS PING/PONG liveness check.

        nats-py's flush() sends a PING to the server and waits for PONG.
        If the connection is silently dead, this times out, revealing the
        problem before 30s of stale data accumulates.
        """
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                if self._reconnecting:
                    continue
                # If _nc is None, a prior reconnect failed and left no connection.
                # Retry reconnection instead of silently skipping forever.
                if self._nc is None:
                    await self._try_force_reconnect("heartbeat: _nc is None, retrying reconnection")
                    continue
                try:
                    await asyncio.wait_for(self._nc.flush(), timeout=self._heartbeat_timeout)
                    self._consecutive_heartbeat_failures = 0
                except (TimeoutError, Exception) as e:
                    self._consecutive_heartbeat_failures += 1
                    logger.warning(
                        f"Heartbeat failed ({self._consecutive_heartbeat_failures}x): {e}"
                    )
                    # Two consecutive failures = connection is dead
                    if self._consecutive_heartbeat_failures >= 2:
                        await self._try_force_reconnect(
                            f"heartbeat failed {self._consecutive_heartbeat_failures}x consecutively"
                        )
        except asyncio.CancelledError:
            pass
