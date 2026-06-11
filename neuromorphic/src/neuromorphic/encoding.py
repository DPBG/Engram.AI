"""Sensor data → spike trains (flexible input encoding)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.regions import SensoryCortex

logger = logging.getLogger(__name__)


class DynamicSensoryAllocator:
    """Dynamically allocates sensory cortex subranges based on active modalities.

    Only modalities that have actually received data get cortex space.
    When a modality goes inactive (no data for `inactivity_timeout` steps),
    its space is redistributed to the remaining active modalities.
    On startup with no data yet, nothing is pre-allocated — the first
    observation triggers the first allocation.

    **Frozen ranges (safety):** When ``freeze_existing_ranges()`` is called
    (typically after loading saved state), existing modality ranges are locked.
    New modalities are appended to the *end* of the sensory cortex using
    unallocated neurons — existing ranges never move.  This prevents
    destabilising learned weights when a new modality (e.g. proprioceptive)
    appears mid-training.
    """

    _MODALITY_ORDER = ["visual", "auditory", "tactile", "proprioceptive", "body_visual"]

    def __init__(self, config: NeuromorphicConfig):
        self._weights = config.encoding.modality_weights
        self._timeout = config.encoding.inactivity_timeout
        self._n_sensory = config.populations.sensory_cortex
        self._last_active: dict[str, int] = {}  # modality → last active step
        self._current_ranges: dict[str, tuple[int, int]] = {}
        self._frozen: bool = False  # set True after loading saved state
        self._frozen_modalities: set[str] = set()  # modalities locked in place
        self._pending_freeze: bool = False  # auto-freeze after first recompute
        # Start with empty ranges — no pre-allocation until data arrives
        for mod in self._MODALITY_ORDER:
            self._current_ranges[mod] = (0, 0)

    def freeze_existing_ranges(self) -> None:
        """Lock current non-empty ranges so they never move.

        New modalities will be appended after the last frozen range
        instead of triggering a full recompute.  Call this after
        ``set_state()`` to protect learned representations.
        """
        self._frozen = True
        self._frozen_modalities = {
            mod for mod, (s, e) in self._current_ranges.items() if e > s
        }
        if self._frozen_modalities:
            logger.info(
                "Sensory allocator frozen — locked modalities: %s",
                ", ".join(sorted(self._frozen_modalities)),
            )

    def report_active(self, modality: str, step: int) -> None:
        """Mark a modality as active at the given step."""
        prev = self._last_active.get(modality)
        was_active = prev is not None and (step - prev) < self._timeout
        self._last_active[modality] = step
        if not was_active:
            self._recompute(step)

    def get_range(self, modality: str) -> tuple[int, int]:
        """Get (start, end) indices for a modality. Returns empty range if unknown."""
        return self._current_ranges.get(modality, (0, 0))

    @property
    def current_ranges(self) -> dict[str, tuple[int, int]]:
        """Current subrange assignments for all modalities."""
        return dict(self._current_ranges)

    def _recompute(self, current_step: int = 0) -> None:
        """Recompute subrange boundaries based on currently active modalities.

        When frozen, existing modality ranges are preserved and new
        modalities are appended to the end of the sensory cortex.
        """
        active = {}
        for mod, weight in self._weights.items():
            if mod in self._last_active and (current_step - self._last_active[mod]) < self._timeout:
                active[mod] = weight

        if not active:
            return

        # --- Frozen mode: preserve existing, append new ---
        if self._frozen and self._frozen_modalities:
            # Find the end of the last frozen range
            frozen_end = 0
            for mod in self._frozen_modalities:
                _, e = self._current_ranges.get(mod, (0, 0))
                frozen_end = max(frozen_end, e)

            # Only allocate new (unfrozen) modalities into remaining space
            new_mods = {m: w for m, w in active.items() if m not in self._frozen_modalities}
            if not new_mods:
                return  # nothing new to allocate

            remaining = self._n_sensory - frozen_end
            if remaining <= 0:
                logger.warning("No sensory cortex space for new modality — all neurons frozen")
                return

            # Use ALL unfrozen modality weights (not just active) to
            # pre-reserve space for modalities that haven't arrived yet.
            all_unfrozen = {m: w for m, w in self._weights.items()
                           if m not in self._frozen_modalities}
            total_unfrozen_weight = sum(all_unfrozen.values()) or 1.0
            start = frozen_end
            for mod in self._MODALITY_ORDER:
                if mod in new_mods:
                    size = int(remaining * all_unfrozen.get(mod, 0) / total_unfrozen_weight)
                    self._current_ranges[mod] = (start, start + size)
                    start += size
                elif mod not in self._frozen_modalities:
                    self._current_ranges[mod] = (0, 0)

            # Lock newly allocated modalities so they don't shift
            # when yet another modality arrives later.
            self._frozen_modalities.update(new_mods.keys())

            logger.info(
                "Appended new modalities after frozen end (%d): %s",
                frozen_end,
                {m: self._current_ranges[m] for m in new_mods},
            )
            return

        # --- Normal mode (no frozen ranges) ---
        if self._pending_freeze:
            # Pre-allocate ALL modalities at proportional sizes regardless of
            # which modality arrives first.  This prevents race conditions
            # (e.g. MuJoCo proprioceptive arriving before gateway visual)
            # from stealing another modality's position.
            total_weight = sum(self._weights.values())
            start = 0
            for mod in self._MODALITY_ORDER:
                size = int(self._n_sensory * self._weights.get(mod, 0) / total_weight)
                self._current_ranges[mod] = (start, start + size)
                start += size
        else:
            total_weight = sum(active.values())
            start = 0
            for mod in self._MODALITY_ORDER:
                if mod in active:
                    size = int(self._n_sensory * self._weights.get(mod, 0) / total_weight)
                    self._current_ranges[mod] = (start, start + size)
                    start += size
                else:
                    self._current_ranges[mod] = (0, 0)  # no space for inactive

        # Fix rounding — extend last active modality to end.
        # Skip when pending_freeze so that unallocated neurons remain
        # available for new modalities that arrive after the freeze.
        if not self._pending_freeze:
            last_active = [m for m in self._MODALITY_ORDER if m in active]
            if last_active:
                mod = last_active[-1]
                s, _ = self._current_ranges[mod]
                self._current_ranges[mod] = (s, self._n_sensory)

        # Auto-freeze after first recompute when loading trained state.
        # This locks visual+auditory ranges so that new modalities
        # (proprioceptive) are appended, not reshuffled.
        if self._pending_freeze and any(e > s for s, e in self._current_ranges.values()):
            self._pending_freeze = False
            self.freeze_existing_ranges()

# Provenance → sensory sub-range mapping
_PROVENANCE_MAP: dict[str, str] = {
    # Body camera — dedicated modality (must be before generic sensor.video)
    "observation.visual.body": "body_visual",
    "sensor.video.body": "body_visual",
    "sensor.camera": "visual",
    "sensor.video": "visual",
    "sensor.image": "visual",
    "sensor.mic": "auditory",
    "sensor.microphone": "auditory",
    "sensor.audio": "auditory",
    "sensor.voice": "auditory",
    "observation.pain": "tactile",  # nociceptive — somatosensory
    "sensor.touch": "tactile",
    "sensor.pressure": "tactile",
    "sensor.tactile": "tactile",
    "sensor.serial": "proprioceptive",
    "sensor.imu": "proprioceptive",
    "sensor.gyro": "proprioceptive",
    "sensor.joint": "proprioceptive",
    "sensor.motor": "proprioceptive",
    "sensor.videofile": "visual",
    "sensor.audiofile": "auditory",
    "sensor.transcript": "auditory",
    "sensor.text": "auditory",
    "observation.text": "auditory",
    # Cognitive bridge responses — LLM output routed as auditory (language)
    "cognitive.response": "auditory",
    "cognitive.llm": "auditory",
    # Motor outcome feedback — routed as proprioceptive (body sense)
    "motor.outcome": "proprioceptive",
    "motor.feedback": "proprioceptive",
    # MuJoCo continuous physics loop proprioceptive emission
    "observation.proprioceptive": "proprioceptive",
}


def _resolve_modality(provenance: str) -> str:
    """Map provenance string to sensory modality."""
    # Exact match first
    if provenance in _PROVENANCE_MAP:
        return _PROVENANCE_MAP[provenance]
    # Prefix match
    for prefix, modality in _PROVENANCE_MAP.items():
        if provenance.startswith(prefix):
            return modality
    # Fallback: distribute across all modalities
    return "visual"


class SpikeEncoder:
    """
    Converts raw observation data into spike currents for the sensory cortex.

    Supports:
    - Rate coding (default): normalize → scale by gain → add noise
    - Provenance-based routing to sensory sub-ranges
    - Pre-processed feature vectors
    - Graceful handling of unknown formats
    """

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        self.config = config
        self._rng = rng or np.random.default_rng()
        self._enc = config.encoding

    def encode(
        self,
        sensory_cortex: SensoryCortex,
        data: Any,
        provenance: str = "",
        allocator: DynamicSensoryAllocator | None = None,
    ) -> np.ndarray:
        """
        Encode observation data into current injection for sensory cortex.

        Args:
            sensory_cortex: The sensory cortex region (for sub-range info).
            data: Raw observation payload — dict, list, or numeric array.
            provenance: Source provenance string for routing.
            allocator: Optional dynamic allocator for sensory subrange redistribution.

        Returns:
            float32 current array of shape (sensory_cortex.n,)
        """
        n = sensory_cortex.n
        current = np.zeros(n, dtype=np.float32)

        # Determine target modality
        modality = _resolve_modality(provenance)

        # Use allocator for dynamic subrange, or fall back to static subranges
        if allocator is not None:
            allocator.report_active(modality, sensory_cortex.time)
            start, end = allocator.get_range(modality)
            sr_slice = slice(start, end)
            sr_size = end - start
        else:
            sr = sensory_cortex.get_subrange(modality)
            if sr is None:
                logger.warning(f"Unknown modality {modality}, using full range")
                sr_slice = slice(0, n)
                sr_size = n
            else:
                sr_slice = sr.slice()
                sr_size = sr.size

        if sr_size <= 0:
            return current

        # Extract feature vector from data
        features = self._extract_features(data, sr_size)

        # Rate coding: normalize → scale → noise
        if features is not None and len(features) > 0:
            features = self._rate_encode(features, sr_size)
            current[sr_slice] = features

        return current

    def encode_multimodal(
        self,
        sensory_cortex: SensoryCortex,
        inputs: dict[str, Any],
        allocator: DynamicSensoryAllocator | None = None,
    ) -> np.ndarray:
        """
        Encode multiple simultaneous sensory inputs.

        Args:
            inputs: dict mapping provenance → data for each modality.
            allocator: Optional dynamic allocator for sensory subrange redistribution.

        Returns:
            Combined current array.
        """
        n = sensory_cortex.n
        current = np.zeros(n, dtype=np.float32)
        for provenance, data in inputs.items():
            current += self.encode(sensory_cortex, data, provenance, allocator=allocator)
        return current

    def _extract_features(self, data: Any, target_size: int) -> np.ndarray | None:
        """Extract a numeric feature vector from observation data."""
        if isinstance(data, np.ndarray):
            return data.astype(np.float32).flatten()

        if isinstance(data, (list, tuple)):
            try:
                return np.array(data, dtype=np.float32).flatten()
            except (ValueError, TypeError):
                pass

        if isinstance(data, dict):
            # Temporal audio frame from AuditorySTM or aggregator
            if data.get("type") == "temporal_audio_frame":
                return self._extract_temporal_audio(data)
            # Try 'features', 'data', 'values' keys
            for key in ("features", "data", "values", "vector"):
                if key in data:
                    return self._extract_features(data[key], target_size)
            # Fall back to numeric values in dict
            vals = [v for v in data.values() if isinstance(v, (int, float))]
            if vals:
                return np.array(vals, dtype=np.float32)

        if isinstance(data, str):
            if not data:
                return None
            # Convert text to character ordinals normalized to [0, 1], clamped for unicode
            return np.array([min(ord(c) / 127.0, 1.0) for c in data[:target_size]], dtype=np.float32)

        if isinstance(data, (int, float)):
            return np.array([float(data)], dtype=np.float32)

        return None

    def _extract_temporal_audio(self, frame: dict) -> np.ndarray | None:
        """Extract features from a temporal audio frame (fallback path).

        Concatenates mean_mfcc + delta_mfcc + energy into a single vector.
        This is a lightweight fallback for when temporal frames bypass the
        AuditorySTM (e.g., direct NATS injection, first frame on startup).
        Normal pipeline: STM produces richer features before encoding.
        """
        parts = []
        mean = frame.get("mean_mfcc")
        if mean is not None:
            parts.append(np.array(mean, dtype=np.float32))

        delta = frame.get("delta_mfcc")
        if delta is not None:
            parts.append(np.array(delta, dtype=np.float32))

        energy = frame.get("energy")
        if energy is not None:
            parts.append(np.array(energy, dtype=np.float32))

        if not parts:
            return None
        return np.concatenate(parts)

    def _rate_encode(self, features: np.ndarray, target_size: int) -> np.ndarray:
        """Apply rate coding: normalize, scale, add noise, resize to target."""
        if len(features) == 0:
            return np.zeros(target_size, dtype=np.float32)
        # Guard against NaN/inf
        features = np.where(np.isfinite(features), features, np.float32(0.0))

        # Normalize to [0, 1]
        fmin, fmax = features.min(), features.max()
        if fmax > fmin:
            features = (features - fmin) / (fmax - fmin)
        else:
            features = np.zeros_like(features) + 0.5

        # Scale by gain
        features = features * self._enc.rate_gain

        # Add noise and clamp to non-negative
        if self._enc.noise_fraction > 0:
            noise = self._rng.normal(0, self._enc.noise_fraction * self._enc.rate_gain, size=len(features))
            features = features + noise.astype(np.float32)
        np.maximum(features, np.float32(0.0), out=features)

        # Resize to target_size (repeat or truncate)
        if len(features) < target_size:
            # Tile to fill
            repeats = (target_size // len(features)) + 1
            features = np.tile(features, repeats)[:target_size]
        elif len(features) > target_size:
            features = features[:target_size]

        return features.astype(np.float32)
