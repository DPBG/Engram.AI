"""
CNN Retina — lightweight pretrained feature extraction for visual sensors.

Uses MobileNetV3-Small via ONNX Runtime to extract 576 semantic features
from 64x64 grayscale frames. Acts as a biological retina/LGN:
raw pixels are preprocessed into meaningful features before reaching cortex.

Runs on the gateway side (host), NOT in the neuromorphic container.
This keeps the brain code clean — it receives feature vectors the same
way it receives raw pixels, through the standard encoding pipeline.

Usage:
    preprocessor = get_shared_preprocessor()  # singleton ONNX session
    features = preprocessor.extract(gray_64x64)  # -> 576 floats
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization stats
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "mobilenetv3_small_features.onnx"


class CNNPreprocessor:
    """MobileNetV3-Small feature extractor via ONNX Runtime."""

    def __init__(self, model_path: str | Path | None = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install sensory-gateway[cnn]"
            )

        path = Path(model_path) if model_path else _MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {path}. "
                f"Run: python scripts/export_mobilenet.py"
            )

        # Limit ONNX internal thread pool to avoid CPU thrashing.
        # Default (0) uses ALL cores — with 403 sensors calling extract()
        # concurrently, this causes 403 x 64 = 25K+ threads competing.
        # 2 intra-op threads is actually faster per-call due to less contention
        # (3.2ms vs 10.1ms) and uses ~3 cores total instead of 60.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        actual_providers = self._session.get_providers()
        logger.info(f"ONNX Runtime providers: {actual_providers}")

        # Verify output shape
        out_shape = self._session.get_outputs()[0].shape
        if out_shape and len(out_shape) > 0 and out_shape[-1] is not None:
            self._feature_dim = out_shape[-1]
        else:
            self._feature_dim = 576
        logger.info(
            f"CNN preprocessor loaded: {path.name} "
            f"({self._feature_dim} features)"
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract(self, gray_64x64: np.ndarray) -> list[float]:
        """Extract features from a 64x64 grayscale frame.

        Args:
            gray_64x64: 2D (64, 64) or 1D (4096,) float32 array, values in [0, 1].

        Returns:
            List of feature floats (576 for MobileNetV3-Small).
        """
        # Validate and reshape
        if gray_64x64.ndim == 1:
            if len(gray_64x64) != 4096:
                logger.warning(f"Expected 4096 elements, got {len(gray_64x64)}, truncating/padding")
                padded = np.zeros(4096, dtype=np.float32)
                n = min(len(gray_64x64), 4096)
                padded[:n] = gray_64x64[:n]
                gray_64x64 = padded
            img = gray_64x64.reshape(64, 64)
        elif gray_64x64.ndim == 2:
            img = gray_64x64
        else:
            logger.warning(f"Unexpected ndim={gray_64x64.ndim}, flattening")
            img = gray_64x64.flatten()[:4096]
            padded = np.zeros(4096, dtype=np.float32)
            padded[:len(img)] = img
            img = padded.reshape(64, 64)

        # Sanitize NaN/inf
        if not np.isfinite(img).all():
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        # Resize to 224x224 (MobileNetV3 expected input)
        img_224 = cv2.resize(img.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)

        # Grayscale -> 3-channel (stack)
        img_3ch = np.stack([img_224, img_224, img_224], axis=-1)  # (224, 224, 3)

        # Normalize per ImageNet stats
        img_3ch = (img_3ch - _IMAGENET_MEAN) / _IMAGENET_STD

        # NCHW format: (1, 3, 224, 224)
        tensor = img_3ch.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        # Run inference
        result = self._session.run([self._output_name], {self._input_name: tensor})
        features = result[0].flatten()

        return features.tolist()


# ── Singleton shared instance ──
_shared_instance: CNNPreprocessor | None = None
_singleton_lock = threading.Lock()


def get_shared_preprocessor(model_path: str | Path | None = None) -> CNNPreprocessor:
    """Return a shared CNNPreprocessor instance (singleton).

    All video sensors share one ONNX session (~50MB) instead of each
    creating their own (~50MB x N sensors = massive RAM waste).
    The extract() method is stateless, so sharing is safe.

    Thread-safe via double-checked locking — prevents duplicate ONNX
    sessions when multiple sensors call this concurrently at startup.
    """
    global _shared_instance
    if _shared_instance is None:
        with _singleton_lock:
            if _shared_instance is None:
                _shared_instance = CNNPreprocessor(model_path)
                logger.info("Created shared CNN preprocessor singleton")
    return _shared_instance
