"""
Transcript sensor — pre-transcribes video audio via Whisper, then emits
timestamped text segments in sync with video playback.

Published to observation.{sensor_id} with provenance "sensor.transcript".
The neuromorphic encoding pipeline routes text to the auditory modality
(encoded as character ordinals, same as observation.text).

MFCC features capture sound texture (voice pitch, music, environmental sounds).
Transcript captures semantic content (words). Both go to auditory cortex but
provide different information. The brain gets both raw audio features AND
word labels simultaneously.

Requires the [stt] optional dependency:
    pip install sensory-gateway[stt]
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np

from activelearning.plugins import SensorPlugin, PluginCapability, RiskClass

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class TranscriptSensor(SensorPlugin[str]):
    """Pre-transcribed text segments from video -> NATS observation."""

    def __init__(
        self,
        filepath: str,
        model_size: str = "tiny",
        loop: bool = True,
        session_id: str | None = None,
    ):
        stem = Path(filepath).stem[:20]
        sensor_id = f"transcript.{stem}"
        super().__init__(
            sensor_id=sensor_id,
            name=f"Transcript: {Path(filepath).name}",
            description=f"Whisper {model_size} video transcription",
            rate_limit_hz=2.0,  # check twice per second for segment timing
            risk_class=RiskClass.LOW,
        )
        self._filepath = filepath
        self._model_size = model_size
        self._loop = loop
        self._session_id = session_id
        self._segments: list[dict] = []  # {start, end, text}
        self._segment_idx: int = 0
        self._playback_start: float = 0.0
        self._loop_count: int = 0
        self._segments_emitted: int = 0
        self._audio_duration_s: float = 0.0

        self.add_capability(PluginCapability(
            name="video_transcription",
            description="Transcribes video audio and emits text segments",
            parameters={"model": model_size, "mode": "offline"},
        ))

    def _cache_path(self) -> Path:
        """Path to cached transcript JSON alongside the video file."""
        return Path(self._filepath + f".transcript.{self._model_size}.json")

    def _load_cache(self) -> bool:
        """Try to load cached transcript. Returns True on success."""
        cache = self._cache_path()
        if not cache.exists():
            return False
        try:
            data = json.loads(cache.read_text())
            self._segments = data.get("segments", [])
            self._audio_duration_s = data.get("audio_duration_s", 0.0)
            if not self._segments or self._audio_duration_s <= 0:
                return False
            logger.info(
                f"Loaded cached transcript: {len(self._segments)} segments, "
                f"{self._audio_duration_s:.1f}s ({cache.name})"
            )
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Cache corrupt, will re-transcribe: {e}")
            return False

    def _save_cache(self) -> None:
        """Save transcript to cache file for instant loading next time."""
        cache = self._cache_path()
        try:
            cache.write_text(json.dumps({
                "model": self._model_size,
                "audio_duration_s": self._audio_duration_s,
                "segments": self._segments,
            }, ensure_ascii=False))
            logger.info(f"Cached transcript: {cache.name}")
        except OSError as e:
            logger.warning(f"Failed to cache transcript: {e}")

    async def start(self, bus=None) -> None:
        """Load cached transcript or run Whisper, then start emit loop."""
        if self._running:
            logger.warning(f"TranscriptSensor {self.sensor_id} already running, skipping start")
            return

        # Try cached transcript first (instant load)
        if not self._load_cache():
            from sensors.audio_file import _extract_audio

            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, _extract_audio, self._filepath)
            self._audio_duration_s = len(audio) / SAMPLE_RATE

            # Run Whisper transcription
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError(
                    "faster-whisper not installed. "
                    "Install with: pip install sensory-gateway[stt]"
                )

            logger.info(f"Transcribing {Path(self._filepath).name} with Whisper {self._model_size}...")
            model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(self._model_size, device="cpu", compute_type="int8"),
            )
            segments_iter, info = await loop.run_in_executor(
                None,
                lambda: model.transcribe(audio, beam_size=1, vad_filter=True),
            )
            # Materialize segments (generator -> list)
            self._segments = []
            for seg in segments_iter:
                text = seg.text.strip()
                if text:
                    self._segments.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": text,
                    })

            logger.info(
                f"Transcription complete: {len(self._segments)} segments, "
                f"{self._audio_duration_s:.1f}s audio"
            )
            if self._segments:
                preview = " | ".join(s["text"][:30] for s in self._segments[:5])
                logger.info(f"Preview: {preview}")

            # Cache for next startup
            self._save_cache()

        self._segment_idx = 0
        self._loop_count = 0
        self._segments_emitted = 0
        self._playback_start = time.time()

        await super().start(bus)

    async def stop(self) -> None:
        """Stop the sensor."""
        await super().stop()
        self._segments = []

    async def capture(self) -> str | None:
        """Emit the text segment matching current playback time.

        Returns text string or None if between segments.
        """
        if not self._segments:
            return None

        # Guard against zero-duration audio
        if self._audio_duration_s <= 0:
            return None

        # Snapshot elapsed once for consistent timing within this call
        elapsed = time.time() - self._playback_start
        video_time = elapsed % self._audio_duration_s

        # Detect loop transition
        current_loop = int(elapsed / self._audio_duration_s)
        if current_loop > self._loop_count:
            if not self._loop:
                return None
            self._loop_count = current_loop
            self._segment_idx = 0
            logger.info(f"Transcript loop #{self._loop_count}: {Path(self._filepath).name}")

        # Find segment that should be playing now
        if self._segment_idx >= len(self._segments):
            return None

        seg = self._segments[self._segment_idx]
        if video_time >= seg["start"] and video_time < seg["end"]:
            # We're in this segment's time window — emit it once
            self._segment_idx += 1
            self._segments_emitted += 1
            return seg["text"]

        # Advance past segments we missed (if playback jumped)
        while self._segment_idx < len(self._segments) and video_time >= self._segments[self._segment_idx]["end"]:
            self._segment_idx += 1

        return None

    async def _emit_loop(self) -> None:
        """Override emit loop to skip None (between segments)."""
        interval = 1.0 / self.rate_limit_hz if self.rate_limit_hz > 0 else 0
        while self._running:
            try:
                data = await self.capture()
                if data is not None and data.strip():
                    await self.emit(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in transcript sensor {self.sensor_id}: {e}")
            if interval > 0:
                await asyncio.sleep(interval)
