"""
Human Verifier - Multi-modal human presence detection.

Uses camera (face detection + liveness) and microphone (voice activity)
to verify that a real human is present and issuing the override.
"""

import logging
import time
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)


class HumanVerifier:
    """
    Multi-modal human presence verification.

    Prevents computer programs from issuing overrides by requiring
    proof of human presence via camera and/or microphone.
    """

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self._camera_available = False
        self._microphone_available = False
        self._physical_button_available = False

        # Try to detect available sensors
        self._detect_sensors()

    def _detect_sensors(self) -> None:
        """Detect which verification sensors are available."""
        # Try camera
        try:
            import cv2
            self._camera = cv2.VideoCapture(0)
            if self._camera.isOpened():
                self._camera_available = True
                logger.info("Camera available for human verification")
            else:
                self._camera_available = False
                logger.warning("Camera not available")
        except Exception as e:
            logger.warning(f"Camera detection failed: {e}")
            self._camera_available = False

        # Try microphone
        try:
            import pyaudio
            self._audio = pyaudio.PyAudio()
            # Check if any input devices are available
            if self._audio.get_device_count() > 0:
                self._microphone_available = True
                logger.info("Microphone available for human verification")
            else:
                self._microphone_available = False
                logger.warning("Microphone not available")
        except Exception as e:
            logger.warning(f"Microphone detection failed: {e}")
            self._microphone_available = False

        # TODO: Detect physical button (GPIO or USB HID)
        self._physical_button_available = False

    async def verify_human(self) -> dict:
        """
        Verify that a human is present.

        Returns:
            dict with keys:
                - verified: bool indicating if human was verified
                - confidence: float (0.0 to 1.0)
                - method: str indicating which method was used
                - error: optional error message
        """
        if not any([self._camera_available, self._microphone_available, self._physical_button_available]):
            logger.error("No verification methods available")
            return {
                "verified": False,
                "confidence": 0.0,
                "method": "none",
                "error": "No verification sensors available. Install camera or microphone.",
            }

        # Try camera-based verification first (strongest signal)
        if self._camera_available:
            camera_result = await self._verify_camera()
            if camera_result["verified"] and camera_result["confidence"] >= self.confidence_threshold:
                return camera_result

        # Try microphone-based verification
        if self._microphone_available:
            mic_result = await self._verify_microphone()
            if mic_result["verified"] and mic_result["confidence"] >= self.confidence_threshold:
                return mic_result

        # Fallback to physical button
        if self._physical_button_available:
            button_result = await self._verify_button()
            if button_result["verified"]:
                return button_result

        # No verification succeeded
        return {
            "verified": False,
            "confidence": 0.0,
            "method": "none",
            "error": "Human verification failed. No valid signal detected.",
        }

    async def _verify_camera(self) -> dict:
        """
        Camera-based human verification.

        Uses face detection and simple liveness check (movement between frames).
        """
        try:
            import cv2
            import numpy as np

            # Load face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Capture multiple frames for liveness check
            frames = []
            for _ in range(10):  # 10 frames over ~0.5 seconds
                ret, frame = self._camera.read()
                if ret:
                    frames.append(frame)
                await asyncio.sleep(0.05)

            if len(frames) < 5:
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "method": "camera",
                    "error": "Could not capture enough frames",
                }

            # Detect faces in each frame
            face_detections = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_detections.append(len(faces) > 0)

            # Check if face was detected in most frames
            face_detection_rate = sum(face_detections) / len(face_detections)

            if face_detection_rate < 0.6:
                return {
                    "verified": False,
                    "confidence": face_detection_rate,
                    "method": "camera",
                    "error": "No face detected",
                }

            # Simple liveness check: detect movement between frames
            movement_scores = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                movement = np.mean(diff)
                movement_scores.append(movement)

            avg_movement = np.mean(movement_scores)

            # If there's some movement (not a static photo), increase confidence
            liveness_confidence = min(avg_movement / 10.0, 1.0)  # Scale to 0-1
            final_confidence = (face_detection_rate * 0.7) + (liveness_confidence * 0.3)

            return {
                "verified": final_confidence >= self.confidence_threshold,
                "confidence": final_confidence,
                "method": "camera",
                "error": None if final_confidence >= self.confidence_threshold else "Liveness check failed",
            }

        except Exception as e:
            logger.error(f"Camera verification error: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "method": "camera",
                "error": str(e),
            }

    async def _verify_microphone(self) -> dict:
        """
        Microphone-based human verification.

        Uses voice activity detection to confirm human speech.
        """
        try:
            import pyaudio
            import numpy as np

            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            RECORD_SECONDS = 2

            stream = self._audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            logger.info("Listening for voice...")

            # Record audio
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

            # Simple voice activity detection based on energy
            energy = np.abs(audio_data).mean()
            noise_floor = 500  # Typical background noise level

            # Calculate confidence based on energy above noise floor
            if energy > noise_floor * 3:
                # Strong signal
                confidence = min((energy - noise_floor) / (noise_floor * 10), 1.0)
            else:
                confidence = 0.0

            return {
                "verified": confidence >= self.confidence_threshold,
                "confidence": confidence,
                "method": "microphone",
                "error": None if confidence >= self.confidence_threshold else "No voice detected",
            }

        except Exception as e:
            logger.error(f"Microphone verification error: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "method": "microphone",
                "error": str(e),
            }

    async def _verify_button(self) -> dict:
        """
        Physical button verification (fallback).

        Waits for physical button press as proof of human presence.
        """
        # TODO: Implement GPIO button or USB HID button detection
        logger.warning("Physical button verification not implemented")
        return {
            "verified": False,
            "confidence": 0.0,
            "method": "button",
            "error": "Physical button not implemented",
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_camera') and self._camera_available:
            self._camera.release()
        if hasattr(self, '_audio') and self._microphone_available:
            self._audio.terminate()
