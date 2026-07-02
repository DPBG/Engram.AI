"""
Speech-to-text sensor — transcribes audio via Whisper.

Buffers 2-3 seconds of audio, runs faster-whisper for transcription,
publishes text to observation.{sensor_id} with provenance "sensor.voice".
The neuromorphic encoding pipeline routes this to the auditory modality
(text encoded as character ordinals).

Requires the [stt] optional dependency:
    pip install sensory-gateway[stt]
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque

import numpy as np
import sounddevice as sd
from activelearning.plugins import PluginCapability, RiskClass, SensorPlugin

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BUFFER_SECONDS = 3.0
SILENCE_THRESHOLD = 0.01  # RMS below this = silence


class SpeechSensor(SensorPlugin[str]):
    """Mac microphone -> Whisper STT -> text -> NATS observation."""

    def __init__(
        self,
        device_index: int | None = None,
        model_size: str = "tiny",
        buffer_seconds: float = BUFFER_SECONDS,
    ):
        sensor_id = "voice.default" if device_index is None else f"voice.{device_index}"

        super().__init__(
            sensor_id=sensor_id,
            name="Speech-to-Text",
            description=f"Whisper {model_size} speech recognition",
            rate_limit_hz=0.3,  # ~1 transcription per 3 seconds
            risk_class=RiskClass.LOW,
        )
        self._device_index = device_index
        self._model_size = model_size
        self._buffer_seconds = buffer_seconds
        self._model = None
        self._stream: sd.InputStream | None = None
        self._audio_buffer: deque[np.ndarray] = deque(
            maxlen=int(SAMPLE_RATE * buffer_seconds / 512) + 1
        )

        self.add_capability(
            PluginCapability(
                name="speech_recognition",
                description="Transcribes speech to text via Whisper",
                parameters={"model": model_size, "language": "auto"},
            )
        )

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio callback — buffer incoming audio."""
        if status:
            logger.debug(f"Audio status: {status}")
        self._audio_buffer.append(indata[:, 0].copy())

    async def start(self, bus=None) -> None:
        """Load Whisper model and start audio capture."""
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info(f"Whisper model '{self._model_size}' loaded")
        except ImportError:
            logger.error(
                "faster-whisper not installed. " "Install with: pip install sensory-gateway[stt]"
            )
            raise

        self._stream = sd.InputStream(
            device=self._device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=512,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Speech sensor audio stream started")
        await super().start(bus)

    async def stop(self) -> None:
        """Stop audio capture and release model."""
        await super().stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._audio_buffer.clear()
        self._model = None

    async def capture(self) -> str:
        """Buffer audio, detect speech, transcribe.

        Returns transcribed text string. Returns empty string if silence.
        The neuromorphic encoder converts text to character ordinals and
        routes to the auditory pathway.
        """
        # Wait for enough audio to accumulate
        target_chunks = int(SAMPLE_RATE * self._buffer_seconds / 512)
        while len(self._audio_buffer) < target_chunks:
            await asyncio.sleep(0.1)

        # Drain buffer atomically — popleft() is GIL-atomic, avoids race
        # with the PortAudio callback thread that appends.
        chunks = []
        while self._audio_buffer:
            try:
                chunks.append(self._audio_buffer.popleft())
            except IndexError:
                break
        audio = np.concatenate(chunks)

        # Skip if silence (no point running Whisper on nothing)
        rms = np.sqrt(np.mean(audio**2))
        if rms < SILENCE_THRESHOLD:
            return ""

        # Run Whisper in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._transcribe, audio)
        if text:
            logger.info(f"Transcribed: {text[:80]}")
        return text

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run Whisper transcription (blocking, called via executor)."""
        if self._model is None:
            return ""

        segments, _ = self._model.transcribe(
            audio,
            beam_size=1,
            language=None,  # auto-detect
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
