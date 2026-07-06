"""Klatt-style formant synthesizer -- virtual vocal tract for the audio brain.

Phase 2.7a (2026-06-03). The audio brain's speech motor cortex no longer
emits ASCII tokens. Instead its speech sub-range neurons drive ten
continuous articulator-style parameters that this module turns into a real
audio waveform via cascade formant synthesis. The same waveform that
trains the brain via efference copy is what a humanoid speaker would play.

Why this exists
---------------
A real brain controls speech via vocal tract muscles (jaw, tongue, lips,
larynx) that physically produce acoustic patterns. Tokens / ASCII bytes do
not appear anywhere in the biological pipeline. To stay structurally
faithful while running in software, we virtualize the vocal tract with a
formant synthesizer (the same approach used in Klatt 1980 cascade
synthesis and modern tools like Pink Trombone). The brain learns
continuous motor control over the virtual articulators; words emerge as
temporal trajectories through articulator space, not as token strings.

Architecture
------------
- Voicing source: bandlimited pulse train at F0 (vocal folds vibrating).
- Aspiration source: filtered white noise (breath, /h/).
- Frication source: white noise bandpassed near frication_freq (sibilants).
- Cascade of three formant resonators F1/F2/F3 (the vocal tract filter).
- Nasal coupling: small constant resonance mixed in for nasalised vowels.
- Output: sum of voicing-driven formant chain + aspiration + frication,
  scaled by overall amplitude.

Sample rate: 16 kHz by default (telephone-quality). All filters are simple
2-pole IIR resonators implemented with ``scipy.signal.lfilter`` so this
module has no dependencies beyond numpy and scipy (already in the project
requirements).

Reachability
~~~~~~~~~~~~
Target audio for the imitation curriculum is produced by this same
synthesizer with hand-tuned param trajectories (see
``scripts/generate_target_words.py`` in Phase 2.7c). That guarantees the
brain has motor trajectories that can reproduce its targets -- something a
brain learning from arbitrary human recordings could not assume.

Physical-hardware bridge
~~~~~~~~~~~~~~~~~~~~~~~~
The synth has no I/O. It is a pure transform from articulator parameters
to PCM. In simulation the PCM is published to NATS as efference copy
(``observation.audio``). On a humanoid, the same PCM goes to the speaker
driver; a microphone plugin handles the auditory return path. Nothing
inside this module changes between virtual and physical operation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal as scisig


logger = logging.getLogger(__name__)


# Default articulator parameter ranges. The decoder maps brain firing
# rates in [0, 1] linearly into these ranges before handing the params to
# the synthesizer.
@dataclass(frozen=True)
class ArticulatorRanges:
    f0_hz: tuple[float, float] = (80.0, 280.0)     # vocal-fold pitch
    f1_hz: tuple[float, float] = (270.0, 900.0)    # first formant (jaw + tongue height)
    f2_hz: tuple[float, float] = (800.0, 2400.0)   # second formant (tongue front/back)
    f3_hz: tuple[float, float] = (1900.0, 3400.0)  # third formant (lip rounding)
    voicing: tuple[float, float] = (0.0, 1.0)       # voiced source amplitude
    aspiration: tuple[float, float] = (0.0, 0.6)    # breath noise amplitude
    frication: tuple[float, float] = (0.0, 0.6)     # friction noise amplitude
    friction_freq_hz: tuple[float, float] = (1500.0, 7500.0)  # friction center
    amplitude: tuple[float, float] = (0.0, 1.0)     # overall gain
    nasality: tuple[float, float] = (0.0, 0.4)      # nasal coupling


# Parameter names in canonical order. The decoder produces a vector in
# this order; the synthesizer reads it the same way.
PARAM_NAMES: tuple[str, ...] = (
    "f0_hz",
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "voicing",
    "aspiration",
    "frication",
    "friction_freq_hz",
    "amplitude",
    "nasality",
)
N_PARAMS = len(PARAM_NAMES)
assert N_PARAMS == 10, "PARAM_NAMES must stay at 10 entries"


@dataclass
class SpeechSynthConfig:
    sample_rate: int = 16000
    # Formant bandwidths (Hz). Wider = breathier, narrower = more vowel-like.
    f1_bw: float = 60.0
    f2_bw: float = 90.0
    f3_bw: float = 150.0
    # Nasal formant (fixed, like real nasalisation): ~250 Hz with wide BW.
    nasal_freq: float = 250.0
    nasal_bw: float = 80.0
    # Glottal pulse shape: width as a fraction of the F0 period.
    glottal_open_quotient: float = 0.6
    # Synthesis gain: compensates for the cumulative attenuation of three
    # peak-normalised formant resonators in cascade. Klatt-style synths
    # all carry a fixed overall gain like this. Tuned so a mid-amplitude
    # voiced /a/ comes out around -20 dB FS, comfortably audible without
    # clipping the soft limit.
    synthesis_gain: float = 90.0
    # Output safety
    output_clip: float = 0.95


class SpeechSynth:
    """Stateful Klatt-style formant synthesizer.

    State carried between calls:
      * Glottal phase (so consecutive chunks don't restart the pulse train).
      * Filter delay lines (so resonators don't 'click' between chunks).
      * Last articulator parameter vector (for cross-chunk smoothing).

    Thread safety: not thread-safe. The audio brain calls this from a
    single asyncio task (``_speech_loop``).
    """

    def __init__(
        self,
        config: SpeechSynthConfig | None = None,
        ranges: ArticulatorRanges | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.cfg = config or SpeechSynthConfig()
        self.ranges = ranges or ArticulatorRanges()
        self._rng = rng or np.random.default_rng(2026)

        # Glottal phase accumulator, in radians [0, 2pi).
        self._glottal_phase: float = 0.0

        # Filter state for cascade formants + nasal. lfilter zi shape is
        # (max(len(a), len(b)) - 1,). Resonators are 2nd-order so zi is 2.
        self._zi_f1 = np.zeros(2, dtype=np.float64)
        self._zi_f2 = np.zeros(2, dtype=np.float64)
        self._zi_f3 = np.zeros(2, dtype=np.float64)
        self._zi_nasal = np.zeros(2, dtype=np.float64)
        # Frication bandpass (4th-order Butterworth: zi size = 4).
        self._zi_fric = np.zeros(4, dtype=np.float64)

        # Last params (used to interpolate smoothly across chunk boundaries).
        # None until the first synthesize() call; that call uses its own
        # target params for both start and end so there is no jump.
        self._last_params: np.ndarray | None = None

    # ----------------------------- Public API ------------------------------

    def synthesize(
        self,
        params: np.ndarray,
        duration_s: float,
    ) -> np.ndarray:
        """Synthesize ``duration_s`` seconds of audio from articulator params.

        Args:
            params: float array of shape (10,) in [0, 1]. Order matches
                ``PARAM_NAMES``. Values outside [0, 1] are clipped.
            duration_s: chunk length in seconds. Typically ~1.0 at brain
                step rate; smaller for finer time resolution.

        Returns:
            float32 mono PCM in [-1, 1], shape (round(duration_s * sr),).
        """
        params = np.asarray(params, dtype=np.float32).reshape(-1)
        if params.size != N_PARAMS:
            raise ValueError(
                f"speech params expected {N_PARAMS} values, got {params.size}"
            )
        params = np.clip(params, 0.0, 1.0)

        # Smooth across chunk boundary by linearly interpolating from the
        # last call's final params to this call's params over the chunk.
        if self._last_params is None:
            start = params
        else:
            start = self._last_params
        self._last_params = params.copy()

        sr = self.cfg.sample_rate
        n = int(round(duration_s * sr))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        # Per-sample param trajectories (linear interp).
        t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        traj = start[None, :] * (1.0 - t[:, None]) + params[None, :] * t[:, None]

        # Denormalise [0,1] to physical units.
        decoded = self._decode(traj)

        # Build sources.
        voiced = self._voicing_source(decoded["f0_hz"], decoded["voicing"])
        asp = self._noise(n) * decoded["aspiration"]
        fric = self._frication(decoded["frication"], decoded["friction_freq_hz"])

        # Cascade formant filter (operates on voiced + aspiration).
        excitation = voiced + asp
        formants = self._formant_cascade(excitation, decoded)

        # Nasal coupling adds a parallel low-frequency resonance.
        nasal = self._nasal_branch(excitation, decoded["nasality"])

        out = (formants + nasal + fric) * decoded["amplitude"] * self.cfg.synthesis_gain
        # Soft clip to avoid speaker damage / unit tests blowing up.
        clip = self.cfg.output_clip
        np.clip(out, -clip, clip, out=out)
        return out.astype(np.float32, copy=False)

    def reset(self) -> None:
        """Drop all carried-over state. Use between training episodes."""
        self._glottal_phase = 0.0
        self._zi_f1[:] = 0.0
        self._zi_f2[:] = 0.0
        self._zi_f3[:] = 0.0
        self._zi_nasal[:] = 0.0
        self._zi_fric[:] = 0.0
        self._last_params = None

    # --------------------------- Internal helpers --------------------------

    def _decode(self, traj: np.ndarray) -> dict[str, np.ndarray]:
        """Map normalised [0,1] trajectory to physical articulator values."""
        r = self.ranges
        out: dict[str, np.ndarray] = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = getattr(r, name)
            out[name] = (lo + traj[:, i] * (hi - lo)).astype(np.float64)
        return out

    def _voicing_source(
        self, f0_hz: np.ndarray, voicing_amp: np.ndarray,
    ) -> np.ndarray:
        """Bandlimited pulse train modulated by glottal pulse shape."""
        sr = self.cfg.sample_rate
        n = f0_hz.shape[0]
        # Integrate F0 to get phase; carries over from previous call.
        dphase = 2.0 * np.pi * f0_hz / sr
        phase = np.cumsum(dphase) + self._glottal_phase
        self._glottal_phase = float(phase[-1] % (2.0 * np.pi))

        # Glottal pulse shape: half-sine over the open quotient, zero rest.
        oq = self.cfg.glottal_open_quotient
        wrapped = (phase / (2.0 * np.pi)) % 1.0
        open_mask = wrapped < oq
        pulse = np.where(
            open_mask,
            np.sin(np.pi * wrapped / oq),
            0.0,
        ).astype(np.float64)

        return pulse * voicing_amp

    def _noise(self, n: int) -> np.ndarray:
        """White Gaussian noise, unit-ish amplitude."""
        return self._rng.standard_normal(n).astype(np.float64) * 0.5

    def _frication(
        self, frication_amp: np.ndarray, friction_freq_hz: np.ndarray,
    ) -> np.ndarray:
        """Friction noise bandpassed around a centre frequency.

        Center frequency varies across the chunk (smoothly), so we apply a
        single bandpass at the chunk-mean centre and leave fine-grained
        F-c modulation for a future refinement. Most fricatives (/s/, /sh/)
        keep a steady centre during a single utterance so this is fine for
        bootstrapping.
        """
        sr = self.cfg.sample_rate
        n = frication_amp.shape[0]
        if float(np.max(frication_amp)) < 1e-6:
            # Skip filtering when frication is silent (perf).
            return np.zeros(n, dtype=np.float64)

        fc = float(np.clip(np.mean(friction_freq_hz), 200.0, sr / 2 - 100.0))
        bw = max(400.0, 0.4 * fc)
        low = max(50.0, fc - bw / 2.0) / (sr / 2)
        high = min(0.99, (fc + bw / 2.0) / (sr / 2))
        try:
            b, a = scisig.butter(2, [low, high], btype="band")
        except ValueError:
            return np.zeros(n, dtype=np.float64)
        noise = self._noise(n)
        out, self._zi_fric = scisig.lfilter(b, a, noise, zi=self._zi_fric)
        return out * frication_amp

    def _formant_cascade(
        self, excitation: np.ndarray, decoded: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Three cascaded 2-pole resonators."""
        sr = self.cfg.sample_rate
        # Use the chunk-mean centre for each formant so filter coefficients
        # are constant within a chunk. Brain step is ~1 s; formant motion
        # within that window is small. Per-sample formant modulation can
        # be added in 2.7b if needed.
        f1 = float(np.mean(decoded["f1_hz"]))
        f2 = float(np.mean(decoded["f2_hz"]))
        f3 = float(np.mean(decoded["f3_hz"]))

        x, self._zi_f1 = self._resonator(excitation, f1, self.cfg.f1_bw, self._zi_f1)
        x, self._zi_f2 = self._resonator(x, f2, self.cfg.f2_bw, self._zi_f2)
        x, self._zi_f3 = self._resonator(x, f3, self.cfg.f3_bw, self._zi_f3)
        return x

    def _nasal_branch(
        self, excitation: np.ndarray, nasality: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        gain = float(np.mean(nasality))
        if gain < 1e-6:
            return np.zeros_like(excitation)
        x, self._zi_nasal = self._resonator(
            excitation, cfg.nasal_freq, cfg.nasal_bw, self._zi_nasal,
        )
        return x * gain

    def _resonator(
        self,
        x: np.ndarray,
        freq_hz: float,
        bw_hz: float,
        zi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single 2-pole IIR resonator, stateful via ``zi``.

        Standard Klatt resonator design with DC-blocked peak normalisation
        so the peak gain at ``freq_hz`` is approximately unity regardless
        of bandwidth::

            r  = exp(-pi * bw / sr)
            wc = 2*pi * freq / sr
            a0 = 1, a1 = -2*r*cos(wc), a2 = r**2
            b0 = 1 - r*cos(wc) - r           # peak-norm with DC subtraction

        Compared with the earlier ``(1 - r) * sin(wc)`` numerator, this
        keeps full energy at the formant frequency without the cascade
        attenuating the signal into inaudibility.
        """
        sr = self.cfg.sample_rate
        freq_hz = float(np.clip(freq_hz, 50.0, sr / 2 - 100.0))
        bw_hz = float(np.clip(bw_hz, 20.0, sr / 2 - 100.0))
        r = np.exp(-np.pi * bw_hz / sr)
        wc = 2.0 * np.pi * freq_hz / sr
        a = np.array([1.0, -2.0 * r * np.cos(wc), r * r], dtype=np.float64)
        # Peak-normalised numerator: produces ~unity gain at wc.
        b0 = (1.0 - r) * np.sqrt(1.0 - 2.0 * r * np.cos(2.0 * wc) + r * r)
        b = np.array([b0, 0.0, 0.0], dtype=np.float64)
        y, zo = scisig.lfilter(b, a, x.astype(np.float64), zi=zi)
        return y, zo


# -------------------- Convenience: spectral features -----------------------

def mel_filterbank_energies(
    pcm: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 13,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    fmin_hz: float = 80.0,
    fmax_hz: float | None = None,
) -> np.ndarray:
    """Tiny mel-filterbank feature extractor.

    Used to package efference-copy audio as a ``temporal_audio_frame`` for
    the brain's existing auditory pipeline (``encoding.py:_extract_temporal_audio``).
    Bio-rationale: a real cochlea does frequency decomposition before the
    auditory cortex sees a signal -- mel filtering approximates that.

    Returns:
        float32 array of shape (n_frames, n_mels), log-magnitude energies.
    """
    if pcm.size == 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    if fmax_hz is None:
        fmax_hz = sample_rate / 2.0
    frame_len = int(round(frame_ms * 1e-3 * sample_rate))
    hop_len = max(1, int(round(hop_ms * 1e-3 * sample_rate)))
    if frame_len <= 0 or pcm.size < frame_len:
        return np.zeros((0, n_mels), dtype=np.float32)

    # Hann window.
    window = np.hanning(frame_len).astype(np.float32)

    # Mel scale conversion.
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    mel_lo = hz_to_mel(fmin_hz)
    mel_hi = hz_to_mel(fmax_hz)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = np.array([mel_to_hz(m) for m in mel_pts])
    bin_pts = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
        if mid == lo:
            mid = lo + 1
        if hi == mid:
            hi = mid + 1
        if hi >= fb.shape[1]:
            hi = fb.shape[1] - 1
        if mid >= fb.shape[1]:
            mid = fb.shape[1] - 1
        if lo >= fb.shape[1] or mid <= lo or hi <= mid:
            continue
        fb[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        fb[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)

    # Frame, window, FFT, mel-project.
    n_frames = 1 + (pcm.size - frame_len) // hop_len
    out = np.zeros((n_frames, n_mels), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frame = pcm[start : start + frame_len].astype(np.float32) * window
        spec = np.fft.rfft(frame, n=n_fft)
        mag = np.abs(spec).astype(np.float32) ** 2
        out[i] = np.log1p(fb @ mag)
    return out


def make_temporal_audio_frame(
    pcm: np.ndarray, sample_rate: int = 16000,
) -> dict:
    """Package PCM as a ``temporal_audio_frame`` dict for the auditory pathway.

    Matches the contract used by ``AuditorySTM`` and the existing
    ``encoding._extract_temporal_audio`` path: mean MFCC + delta MFCC +
    energy. We use mel-filterbank energies in place of true MFCCs (no
    discrete cosine transform); the downstream code treats this as a
    feature vector and does not require DCT orthogonality.
    """
    mel = mel_filterbank_energies(pcm, sample_rate=sample_rate)
    if mel.shape[0] == 0:
        zero = [0.0] * 13
        # ``energy`` is intentionally a 1-element list, not a scalar:
        # ``encoding._extract_temporal_audio`` concatenates mean_mfcc /
        # delta_mfcc / energy via ``np.concatenate``, and a 0-D scalar
        # would raise "zero-dimensional arrays cannot be concatenated".
        # Match the existing convention used by the sensory gateway
        # (one energy value per chunk).
        return {
            "type": "temporal_audio_frame",
            "mean_mfcc": zero,
            "delta_mfcc": zero,
            "energy": [0.0],
        }
    mean = mel.mean(axis=0)
    delta = mel[-1] - mel[0]
    energy = float(np.mean(pcm * pcm))
    return {
        "type": "temporal_audio_frame",
        "mean_mfcc": mean.astype(np.float32).tolist(),
        "delta_mfcc": delta.astype(np.float32).tolist(),
        "energy": [energy],
    }
