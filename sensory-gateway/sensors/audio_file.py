"""
Audio file sensor — extracts MFCC features from a video/audio file.

Produces 13 MFCC coefficients per chunk (same as MicrophoneSensor),
published to observation.{sensor_id} with provenance "sensor.audiofile".
The neuromorphic encoding pipeline routes this to the auditory modality.

Extracts audio via ffmpeg at init, then reads through the WAV buffer
at a configurable rate, staying roughly in sync with VideoFileSensor
via a shared playback start time.

Requires ffmpeg to be installed on the system.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin

logger = logging.getLogger(__name__)

# Limit concurrent ffmpeg processes to avoid CPU/memory spikes at startup
_ffmpeg_semaphore = threading.Semaphore(4)

# Reuse MFCC parameters from microphone sensor
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.032  # 32ms
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 512 samples
N_MFCC = 13
SILENCE_FLOOR = -23.0
SILENCE_THRESHOLD = 0.5


def _compute_mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Compute MFCC features from a short audio chunk.

    Reuses the same logic as microphone.py for consistency.
    """
    if len(audio) < CHUNK_SAMPLES:
        return None

    try:
        from python_speech_features import mfcc

        features = mfcc(
            audio,
            samplerate=sample_rate,
            winlen=CHUNK_DURATION,
            winstep=CHUNK_DURATION,
            numcep=N_MFCC,
            nfilt=26,
            nfft=512,
        )
        return features.mean(axis=0).astype(np.float32) if len(features) > 0 else None
    except ImportError:
        pass

    # Minimal fallback
    windowed = audio[:512] * np.hanning(min(len(audio), 512))
    spectrum = np.abs(np.fft.rfft(windowed, n=512)) ** 2
    log_spectrum = np.log(spectrum + 1e-10)
    step = max(1, len(log_spectrum) // N_MFCC)
    features = log_spectrum[: N_MFCC * step : step][:N_MFCC]
    return features.astype(np.float32)


def _extract_audio(video_path: str) -> np.ndarray:
    """Extract audio from video file to 16kHz mono WAV via ffmpeg.

    Returns the full audio as a float32 numpy array.
    Uses a semaphore to limit concurrent ffmpeg processes (max 4).
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Install it: brew install ffmpeg (macOS) "
            "or apt install ffmpeg (Linux)"
        )

    # Use delete=False so the file exists when ffmpeg writes to it
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        with _ffmpeg_semaphore:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vn",  # no video
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",  # mono
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")

        # Read WAV file
        import wave

        with wave.open(tmp_path, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) == 0:
            raise RuntimeError(f"No audio extracted from {Path(video_path).name} (0 samples)")

        logger.info(f"Extracted audio: {len(audio)/SAMPLE_RATE:.1f}s from {Path(video_path).name}")
        return audio
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class AudioFileSensor(SensorPlugin[list]):
    """Audio track from video file -> MFCC features -> NATS observation."""

    def __init__(
        self,
        filepath: str,
        hz: float = 10.0,
        loop: bool = True,
        session_id: str | None = None,
        turbo: bool = False,
    ):
        stem = Path(filepath).stem[:20]
        sensor_id = f"audiofile.{stem}"
        super().__init__(
            sensor_id=sensor_id,
            name=f"Audio: {Path(filepath).name}",
            description="Video audio MFCC extraction",
            rate_limit_hz=hz,
            risk_class=RiskClass.LOW,
        )
        self._filepath = filepath
        self._loop = loop
        self._session_id = session_id
        self._turbo = turbo
        self._audio: np.ndarray | None = None
        self._audio_duration_s: float = 0.0
        self._position: int = 0  # current sample position
        self._chunk_samples: int = int(SAMPLE_RATE / hz)  # samples per emission
        self._loop_count: int = 0
        self._chunks_emitted: int = 0
        self._start_time: float = 0.0
        self._silent_frames: int = 0

        self.add_capability(
            PluginCapability(
                name="audio_file_mfcc",
                description="Extracts MFCC features from video audio track",
                parameters={"sample_rate": "16000", "mfcc_coefficients": "13"},
            )
        )

    @property
    def current_pos_s(self) -> float:
        if self._audio is None:
            return 0.0
        return self._position / SAMPLE_RATE

    @property
    def progress(self) -> float:
        if self._audio_duration_s <= 0:
            return 0.0
        return min(1.0, self.current_pos_s / self._audio_duration_s)

    async def start(self, bus=None) -> None:
        """Extract audio from video file, then start emit loop."""
        if self._running:
            logger.warning(f"AudioFileSensor {self.sensor_id} already running, skipping start")
            return
        loop = asyncio.get_event_loop()
        self._audio = await loop.run_in_executor(None, _extract_audio, self._filepath)
        self._audio_duration_s = len(self._audio) / SAMPLE_RATE
        self._position = 0
        self._loop_count = 0
        self._chunks_emitted = 0
        self._start_time = time.time()
        self._silent_frames = 0
        logger.info(f"Audio sensor ready: {self._audio_duration_s:.1f}s of audio")
        await super().start(bus)

    async def stop(self) -> None:
        """Release audio buffer."""
        await super().stop()
        self._audio = None

    async def capture(self) -> list | None:
        """Read next audio chunk, compute MFCC, return 13 floats.

        Returns None to skip (silence squelch) or when audio ends without loop.
        """
        if self._audio is None:
            return None

        end = self._position + self._chunk_samples
        if end > len(self._audio):
            # End of audio
            if self._loop:
                self._position = 0
                self._loop_count += 1
                self._silent_frames = 0
                logger.info(f"Audio loop #{self._loop_count}: {Path(self._filepath).name}")
                end = self._chunk_samples
            else:
                return None

        chunk = self._audio[self._position : end]
        self._position = end

        features = _compute_mfcc(chunk)
        if features is None:
            return None

        # Silence squelch (same as MicrophoneSensor)
        if np.all(np.abs(features - SILENCE_FLOOR) < SILENCE_THRESHOLD):
            self._silent_frames += 1
            if self._silent_frames > 10:
                return None
            return features.tolist()

        self._silent_frames = 0
        self._chunks_emitted += 1
        return features.tolist()

    async def _emit_loop(self) -> None:
        """Override emit loop to skip None and stop on non-looping end."""
        interval = 1.0 / self.rate_limit_hz if self.rate_limit_hz > 0 else 0
        while self._running:
            try:
                data = await self.capture()
                if data is not None:
                    await self.emit(data)
                elif (
                    not self._loop
                    and self._audio is not None
                    and self._position >= len(self._audio)
                ):
                    logger.info(f"Audio sensor {self.sensor_id} finished (no loop)")
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audio sensor {self.sensor_id}: {e}")
            if self._turbo:
                await asyncio.sleep(0)  # yield only, no delay
            elif interval > 0:
                await asyncio.sleep(interval)
