"""
Microphone sensor — captures audio and extracts MFCC features.

Produces 13 MFCC coefficients per chunk (32ms windows),
published to observation.{sensor_id} with provenance "sensor.audio".
The neuromorphic encoding pipeline routes this to the auditory modality.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque

import numpy as np
import sounddevice as sd
from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin

logger = logging.getLogger(__name__)

# MFCC parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.032  # 32ms per frame
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 512 samples
N_MFCC = 13
N_FFT = 512
N_MELS = 26

# Silence detection: MFCC floor value is ~-23.025 when no sound
SILENCE_FLOOR = -23.0
SILENCE_THRESHOLD = 0.5  # max deviation from floor to count as silence
SILENCE_SQUELCH_FRAMES = 10  # suppress after this many consecutive silent frames


def _compute_mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Compute MFCC features from a short audio chunk.

    Uses python_speech_features if available, falls back to a minimal
    DCT-of-mel-filterbank implementation.
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
            nfilt=N_MELS,
            nfft=N_FFT,
        )
        # Average across frames if multiple
        return features.mean(axis=0).astype(np.float32) if len(features) > 0 else None
    except ImportError:
        pass

    # Minimal fallback: power spectrum -> log -> rough features
    windowed = audio[:N_FFT] * np.hanning(min(len(audio), N_FFT))
    spectrum = np.abs(np.fft.rfft(windowed, n=N_FFT)) ** 2
    log_spectrum = np.log(spectrum + 1e-10)
    # Take first N_MFCC components as rough features
    step = max(1, len(log_spectrum) // N_MFCC)
    features = log_spectrum[: N_MFCC * step : step][:N_MFCC]
    return features.astype(np.float32)


class MicrophoneSensor(SensorPlugin[list]):
    """Mac microphone -> MFCC features -> NATS observation."""

    def __init__(self, device_index: int | None = None, hz: float = 30.0):
        name = "Default Microphone" if device_index is None else f"Microphone {device_index}"
        sensor_id = "mic.default" if device_index is None else f"mic.{device_index}"

        super().__init__(
            sensor_id=sensor_id,
            name=name,
            description="Audio capture with MFCC feature extraction",
            rate_limit_hz=hz,
            risk_class=RiskClass.LOW,
        )
        self._device_index = device_index
        self._stream: sd.InputStream | None = None
        self._buffer: deque[np.ndarray] = deque(maxlen=100)
        self._silent_frames: int = 0

        self.add_capability(
            PluginCapability(
                name="audio_capture",
                description="Captures audio and extracts MFCC features",
                parameters={"sample_rate": "16000", "mfcc_coefficients": "13"},
            )
        )

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio callback — buffer incoming audio chunks."""
        if status:
            logger.debug(f"Audio status: {status}")
        self._buffer.append(indata[:, 0].copy())

    async def start(self, bus=None) -> None:
        """Open the audio stream, then start the emit loop."""
        self._stream = sd.InputStream(
            device=self._device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SAMPLES,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info(f"Microphone stream started (device={self._device_index})")
        try:
            await super().start(bus)
        except Exception:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            raise

    async def stop(self) -> None:
        """Close the audio stream."""
        await super().stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._buffer.clear()

    async def capture(self) -> list | None:
        """Consume buffered audio chunks, compute MFCC, return as list of floats.

        Returns 13 MFCC coefficients, or None if the mic is silent
        (squelched after SILENCE_SQUELCH_FRAMES consecutive silent frames).
        """
        # Wait for audio data if buffer is empty
        while not self._buffer:
            await asyncio.sleep(0.005)

        # When deeply squelched, drain buffer without computing MFCC.
        # Check raw audio RMS to detect when sound returns.
        if self._silent_frames > SILENCE_SQUELCH_FRAMES:
            chunks = []
            while self._buffer:
                try:
                    chunks.append(self._buffer.popleft())
                except IndexError:
                    break
            audio = np.concatenate(chunks)
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.005:  # still silent — skip MFCC entirely
                return None
            # Sound detected — exit squelch, fall through to full processing
            logger.info(f"Audio detected on {self.sensor_id} after silence")
            self._silent_frames = 0
            features = _compute_mfcc(audio)
            return features.tolist() if features is not None else [0.0] * N_MFCC

        # Drain buffer atomically — popleft() is GIL-atomic, so no data
        # is lost between reads even though the PortAudio callback appends
        # from a separate thread.
        chunks = []
        while self._buffer:
            try:
                chunks.append(self._buffer.popleft())
            except IndexError:
                break  # drained between check and pop
        audio = np.concatenate(chunks)

        features = _compute_mfcc(audio)
        if features is None:
            self._silent_frames += 1
            if self._silent_frames > SILENCE_SQUELCH_FRAMES:
                return None
            return [0.0] * N_MFCC

        # Check if MFCC is at silence floor (all coefficients near -23.025)
        if np.all(np.abs(features - SILENCE_FLOOR) < SILENCE_THRESHOLD):
            self._silent_frames += 1
            if self._silent_frames > SILENCE_SQUELCH_FRAMES:
                return None
            return features.tolist()

        # Real audio detected — reset silence counter
        self._silent_frames = 0
        return features.tolist()

    async def _emit_loop(self) -> None:
        """Override emit loop to skip None (squelched silence)."""
        interval = 1.0 / self.rate_limit_hz if self.rate_limit_hz > 0 else 0
        while self._running:
            try:
                data = await self.capture()
                if data is not None:
                    await self.emit(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sensor {self.sensor_id}: {e}")
            if interval > 0:
                await asyncio.sleep(interval)
