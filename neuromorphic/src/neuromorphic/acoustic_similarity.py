"""Acoustic similarity metrics for the speech curriculum.

Phase 2.7b (2026-06-07). The Phase 2.7a Klatt-style synthesizer
(``speech_synth.py``) turns articulator vectors into a real PCM
waveform; this module computes the per-frame features the speech
curriculum needs to grade what the brain actually produces.

Three primitives:

* :func:`mel_features` -- mel-filterbank energies per frame. Wraps
  :func:`speech_synth.mel_filterbank_energies` so callers don't
  recompute the filterbank when they already have one cached from
  the efference-copy publish path.
* :func:`mel_cosine_similarity` -- mean per-frame cosine between two
  feature sequences. Used by :class:`SpeechRepeatTask` and
  :class:`SpeechAssociationTask` to compare the brain's output
  against a target word or against same-concept prior productions.
* :func:`spectral_diversity` -- RMS of per-feature standard deviation
  across a window. Used by :class:`SpeechFirstWordsTask` and
  :class:`SpeechTokenTask` to detect whether the brain produces
  distinct sounds (high diversity) or one stuck output (~0).

Why this lives next to ``speech_synth`` rather than inside it
------------------------------------------------------------------
The synth's job is articulators -> PCM. Comparing PCMs is a different
concern (curriculum grading), and the comparison primitives are useful
to any external module that wants to score audio against a target
(target-word bank generator, tests, dashboards). Keeping the metric
module pure-functional + dependency-free (numpy only, no scipy beyond
what speech_synth already imports) keeps it cheap to test in
isolation.

Patent alignment
----------------
Used by SpeechAssociationTask to grade cross-modal binding (Claim 4)
against real acoustic targets instead of token indices -- which
strengthens Claim 4 by validating that binding generalises to the
actual motor output rather than to the post-hoc tokeniser.
"""

from __future__ import annotations

import logging

import numpy as np

from .speech_synth import mel_filterbank_energies


logger = logging.getLogger(__name__)


def mel_features(
    pcm: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 13,
) -> np.ndarray:
    """Per-frame log mel-filterbank energies. Thin wrapper for clarity.

    Returns shape ``(n_frames, n_mels)``, float32. Empty PCM yields
    ``(0, n_mels)``. Matches the contract used by
    :func:`make_temporal_audio_frame` so the same array can be reused
    for efference-copy publication and for curriculum grading
    without recomputation.
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    return mel_filterbank_energies(pcm, sample_rate=sample_rate, n_mels=n_mels)


def rms_energy(pcm: np.ndarray) -> float:
    """Root mean square energy of a PCM buffer.

    ``_publish_speech_audio`` already computes this for the efference
    silence gate; tasks read it via the speech_state dict so they
    don't repeat the work. Defined here for symmetry with the other
    primitives and so unit tests have a single import surface.
    """
    arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _normalise_rows(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows stay zero (cos becomes 0)."""
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    # Avoid divide-by-zero on silent frames; result on those rows is 0.
    safe = np.where(norms > eps, norms, 1.0)
    out = feat / safe
    out[norms[:, 0] <= eps] = 0.0
    return out


def mel_cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Mean per-frame cosine similarity between two mel-feature sequences.

    Args:
        a, b: float arrays of shape ``(n_frames, n_mels)``. Sequences
            may have different lengths; the shorter is aligned to the
            longer by linear-time interpolation (np.linspace index
            mapping) so the comparison is between "the same fraction
            of the way through the utterance".

    Returns:
        A scalar in ``[-1.0, 1.0]``. Identical sequences return ~1.0,
        silence vs anything returns 0.0, orthogonal spectra return 0.0
        or negative.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        return 0.0
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    if a.shape[1] != b.shape[1]:
        return 0.0

    # Align to a common time grid (length = max). For each output frame i,
    # pick the closest source frame. Cheaper than full resampling and
    # avoids introducing new dependencies.
    n = max(a.shape[0], b.shape[0])
    if a.shape[0] != n:
        idx = np.linspace(0, a.shape[0] - 1, n).astype(int)
        a = a[idx]
    if b.shape[0] != n:
        idx = np.linspace(0, b.shape[0] - 1, n).astype(int)
        b = b[idx]

    a_norm = _normalise_rows(a)
    b_norm = _normalise_rows(b)
    # Per-frame cosine, then mean. Skip frames where both sides are
    # silent (both rows zero -> dot = 0 already, which would drag the
    # mean toward 0 unfairly; instead drop them).
    dots = np.sum(a_norm * b_norm, axis=1)
    a_has = np.linalg.norm(a, axis=1) > 1e-8
    b_has = np.linalg.norm(b, axis=1) > 1e-8
    keep = a_has & b_has
    if not np.any(keep):
        return 0.0
    return float(np.mean(dots[keep]))


def spectral_diversity(features_window: list[np.ndarray]) -> float:
    """Across-window variance of mel-feature centroids.

    Args:
        features_window: list of per-utterance mel feature arrays
            (each ``(n_frames, n_mels)``). Typical use: the rolling
            window of the last K synthesized seconds.

    Returns:
        A scalar >= 0. Roughly 0 when every entry in the window
        produces the same mean mel pattern; grows when the brain
        explores distinct acoustic regions.

    The metric is the L2 norm of the per-feature standard deviation
    across the per-utterance centroids. We reduce each utterance to
    its time-averaged feature vector first, so the diversity number
    reflects "does the brain make different sounds at different times"
    rather than "is the spectrogram noisy within a single utterance".
    """
    if not features_window:
        return 0.0
    centroids: list[np.ndarray] = []
    for f in features_window:
        if f is None:
            continue
        arr = np.asarray(f, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        centroids.append(arr.mean(axis=0))
    if len(centroids) < 2:
        return 0.0
    stacked = np.stack(centroids, axis=0)
    std = stacked.std(axis=0)
    return float(np.linalg.norm(std))
