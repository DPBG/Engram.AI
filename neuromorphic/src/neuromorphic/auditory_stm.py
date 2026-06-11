"""Auditory Short-Term Memory — echoic memory buffer for temporal audio context.

Biologically inspired by the auditory cortex's echoic memory (~3-4s), this
module accumulates consecutive temporal audio frames across multiple brain
steps and presents a rich temporal feature vector to the sensory cortex.

The STM provides cross-step temporal accumulation: even though each brain step
receives one temporal audio frame from the aggregator (covering ~500ms of audio),
the STM maintains a sliding window across multiple steps so that the brain has
access to ~2-4 seconds of audio history.

Patent alignment:
  - Claim 4: Cross-modal binding correlates visual frames with temporally-rich
    auditory representations (not just single-frame snapshots)
  - Claim 5: STM buffer maps to apical dendritic compartment for temporal context
  - Claim 6: Continual learning benefits from richer, more distinct spike patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AuditorySTMConfig:
    """Configuration for the auditory short-term memory buffer."""

    # Number of temporal audio frames to retain (each frame = one aggregator flush)
    window_steps: int = 8  # At 2 Hz flush → 4s of audio history
    # Exponential decay applied to older frames (recent frames weighted more)
    decay_rate: float = 0.85  # per-step decay for older frames
    # Whether to include delta features in the output
    include_delta: bool = True
    # Whether to include energy contour in the output
    include_energy: bool = True
    # Whether to include the raw MFCC stack from the most recent frame
    include_recent_raw: bool = True
    # Max raw stack chunks to include from the most recent frame
    max_recent_raw_chunks: int = 5
    # Feature vector normalization
    normalize_output: bool = True


class AuditorySTM:
    """Echoic memory buffer that accumulates temporal audio frames across brain steps.

    Each brain step, the aggregator delivers a temporal audio frame containing:
      - mean_mfcc: [13] spectral centroid
      - delta_mfcc: [13] temporal derivative
      - energy: [N] amplitude envelope
      - raw_stack: [N×13] complete MFCC sequence

    The STM stores the last ``window_steps`` frames and produces a unified
    feature vector that captures temporal context across multiple seconds.
    """

    N_MFCC = 13  # Number of MFCC coefficients

    def __init__(self, config: AuditorySTMConfig | None = None):
        self.config = config or AuditorySTMConfig()
        self._mean_buffer: list[np.ndarray] = []  # circular buffer of mean_mfcc
        self._delta_buffer: list[np.ndarray] = []  # circular buffer of delta_mfcc
        self._energy_buffer: list[np.ndarray] = []  # circular buffer of energy contours
        self._raw_stack: np.ndarray | None = None  # most recent raw MFCC stack
        self._step_count = 0

    @property
    def n_frames(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._mean_buffer)

    def update(self, temporal_frame: dict) -> None:
        """Ingest a temporal audio frame from the aggregator.

        Args:
            temporal_frame: dict with keys 'mean_mfcc', 'delta_mfcc',
                           'energy', 'raw_stack', 'n_chunks', 'type'.
        """
        if temporal_frame.get("type") != "temporal_audio_frame":
            return

        mean = np.array(temporal_frame["mean_mfcc"], dtype=np.float32)
        delta = np.array(temporal_frame["delta_mfcc"], dtype=np.float32)
        energy = np.array(temporal_frame["energy"], dtype=np.float32)
        raw = temporal_frame.get("raw_stack")

        self._mean_buffer.append(mean)
        self._delta_buffer.append(delta)
        self._energy_buffer.append(energy)

        if raw is not None:
            self._raw_stack = np.array(raw, dtype=np.float32)

        # Trim to window size
        w = self.config.window_steps
        if len(self._mean_buffer) > w:
            self._mean_buffer = self._mean_buffer[-w:]
            self._delta_buffer = self._delta_buffer[-w:]
            self._energy_buffer = self._energy_buffer[-w:]

        self._step_count += 1

    def get_features(self) -> np.ndarray | None:
        """Produce a unified feature vector from the STM buffer.

        Returns a 1D float32 array combining:
          1. Decay-weighted mean MFCC across buffer (13 floats)
          2. Decay-weighted delta MFCC across buffer (13 floats) [optional]
          3. Energy contour summary: mean, std, slope (3 floats) [optional]
          4. Temporal gradient: mean MFCC difference between first and last half (13 floats)
          5. Recent raw stack flattened (N×13 floats) [optional]

        Returns None if the buffer is empty.
        """
        if not self._mean_buffer:
            return None

        n = len(self._mean_buffer)
        cfg = self.config

        # Compute decay weights: most recent = 1.0, oldest = decay^(n-1)
        weights = np.array(
            [cfg.decay_rate ** (n - 1 - i) for i in range(n)],
            dtype=np.float32,
        )
        weights /= weights.sum()  # normalize to sum=1

        # 1. Decay-weighted mean MFCC (13 floats)
        mean_stack = np.stack(self._mean_buffer)  # (n, 13)
        weighted_mean = (mean_stack * weights[:, None]).sum(axis=0)  # (13,)
        parts = [weighted_mean]

        # 2. Decay-weighted delta MFCC (13 floats)
        if cfg.include_delta and self._delta_buffer:
            delta_stack = np.stack(self._delta_buffer)  # (n, 13)
            weighted_delta = (delta_stack * weights[:, None]).sum(axis=0)
            parts.append(weighted_delta)

        # 3. Energy contour summary (3 floats: mean, std, slope)
        if cfg.include_energy and self._energy_buffer:
            all_energy = np.concatenate(self._energy_buffer)
            e_mean = all_energy.mean()
            e_std = all_energy.std()
            # Slope: linear trend of energy across the window
            if len(all_energy) >= 2:
                x = np.arange(len(all_energy), dtype=np.float32)
                with np.errstate(invalid="ignore"):
                    coeffs = np.polyfit(x, all_energy, 1)
                e_slope = float(coeffs[0]) if np.isfinite(coeffs[0]) else 0.0
            else:
                e_slope = 0.0
            # Guard against NaN in energy stats
            e_mean = float(e_mean) if np.isfinite(e_mean) else 0.0
            e_std = float(e_std) if np.isfinite(e_std) else 0.0
            parts.append(np.array([e_mean, e_std, e_slope], dtype=np.float32))

        # 4. Temporal gradient: difference between recent and old halves (13 floats)
        if n >= 2:
            mid = n // 2
            old_half = mean_stack[:mid].mean(axis=0)
            new_half = mean_stack[mid:].mean(axis=0)
            temporal_grad = new_half - old_half
        else:
            temporal_grad = np.zeros(self.N_MFCC, dtype=np.float32)
        parts.append(temporal_grad)

        # 5. Recent raw stack (fixed size: max_recent_raw_chunks × 13 floats)
        # Always pad to fixed size so the feature vector length is constant
        # regardless of how many chunks arrived in the last flush window.
        # This ensures stable neuron-to-feature mapping for STDP.
        if cfg.include_recent_raw:
            padded = np.zeros(
                (cfg.max_recent_raw_chunks, self.N_MFCC), dtype=np.float32
            )
            if self._raw_stack is not None:
                raw = self._raw_stack[-cfg.max_recent_raw_chunks:]
                padded[-len(raw):] = raw  # right-align (most recent at end)
            parts.append(padded.flatten())

        features = np.concatenate(parts).astype(np.float32)

        if cfg.normalize_output and features.max() != features.min():
            fmin, fmax = features.min(), features.max()
            features = (features - fmin) / (fmax - fmin)

        return features

    def get_state(self) -> dict:
        """Serialize STM state for persistence."""
        return {
            "mean_buffer": [m.tolist() for m in self._mean_buffer],
            "delta_buffer": [d.tolist() for d in self._delta_buffer],
            "energy_buffer": [e.tolist() for e in self._energy_buffer],
            "raw_stack": self._raw_stack.tolist() if self._raw_stack is not None else None,
            "step_count": self._step_count,
        }

    def set_state(self, state: dict) -> None:
        """Restore STM state from persistence."""
        self._mean_buffer = [
            np.array(m, dtype=np.float32) for m in state.get("mean_buffer", [])
        ]
        self._delta_buffer = [
            np.array(d, dtype=np.float32) for d in state.get("delta_buffer", [])
        ]
        self._energy_buffer = [
            np.array(e, dtype=np.float32) for e in state.get("energy_buffer", [])
        ]
        raw = state.get("raw_stack")
        self._raw_stack = np.array(raw, dtype=np.float32) if raw is not None else None
        self._step_count = state.get("step_count", 0)

    def reset(self) -> None:
        """Clear the STM buffer."""
        self._mean_buffer.clear()
        self._delta_buffer.clear()
        self._energy_buffer.clear()
        self._raw_stack = None
        self._step_count = 0
