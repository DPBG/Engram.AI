#!/usr/bin/env python3
"""
Solution 5 Quality Test — compares old (latest-only) vs new (temporal frame)
audio pipeline using REAL video files from the training set.

Tests:
1. Information preservation: How much MFCC data survives each pipeline?
2. Temporal coherence: Can the brain distinguish temporal transitions?
3. Cross-window distinctiveness: Are consecutive observations distinguishable?
4. Feature richness: How many bits of information reach the encoder?
5. End-to-end encoding: Compare spike patterns from old vs new pipeline.

Usage:
    python sandbox/coherence_test/test_solution5.py
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "sensory-gateway"))
sys.path.insert(0, str(PROJECT_ROOT / "sensory-gateway" / "sensors"))
sys.path.insert(0, str(PROJECT_ROOT / "sdk" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "neuromorphic" / "src"))

import numpy as np

# Audio constants (same as AudioFileSensor)
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.032
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
N_MFCC = 13
SILENCE_FLOOR = -23.0
SILENCE_THRESHOLD = 0.5
AUDIO_HZ = 10


def extract_audio(video_path: str) -> np.ndarray:
    """Extract audio from video using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le",
             "-ar", str(SAMPLE_RATE), "-ac", "1", tmp_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")
        with wave.open(tmp_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compute_mfcc(audio_chunk: np.ndarray) -> np.ndarray | None:
    """Compute MFCC (minimal fallback without python_speech_features)."""
    if len(audio_chunk) < CHUNK_SAMPLES:
        return None
    try:
        from python_speech_features import mfcc
        features = mfcc(audio_chunk, samplerate=SAMPLE_RATE,
                        winlen=CHUNK_DURATION, winstep=CHUNK_DURATION,
                        numcep=N_MFCC, nfilt=26, nfft=512)
        return features.mean(axis=0).astype(np.float32) if len(features) > 0 else None
    except ImportError:
        windowed = audio_chunk[:512] * np.hanning(min(len(audio_chunk), 512))
        spectrum = np.abs(np.fft.rfft(windowed, n=512)) ** 2
        log_spectrum = np.log(spectrum + 1e-10)
        step = max(1, len(log_spectrum) // N_MFCC)
        features = log_spectrum[:N_MFCC * step:step][:N_MFCC]
        return features.astype(np.float32)


def is_silent(features: np.ndarray) -> bool:
    return bool(np.all(np.abs(features - SILENCE_FLOOR) < SILENCE_THRESHOLD))


# Import neuromorphic modules directly (no SDK dependency needed)
from neuromorphic.auditory_stm import AuditorySTM, AuditorySTMConfig
from neuromorphic.config import NeuromorphicConfig
from neuromorphic.encoding import SpikeEncoder
from neuromorphic.regions import SensoryCortex


def compute_temporal_audio_frame(chunks):
    """Inline copy of aggregating_bus.compute_temporal_audio_frame (avoids SDK import)."""
    n = len(chunks)
    arr = np.array(chunks, dtype=np.float32)
    mean_mfcc = arr.mean(axis=0)
    if n >= 2:
        delta = np.zeros_like(arr)
        delta[0] = arr[1] - arr[0]
        delta[-1] = arr[-1] - arr[-2]
        if n > 2:
            delta[1:-1] = (arr[2:] - arr[:-2]) / 2.0
        delta_mfcc = delta.mean(axis=0)
    else:
        delta_mfcc = np.zeros(arr.shape[1], dtype=np.float32)
    energy = np.sqrt((arr ** 2).mean(axis=1))
    return {
        "type": "temporal_audio_frame",
        "n_chunks": n,
        "mean_mfcc": mean_mfcc.tolist(),
        "delta_mfcc": delta_mfcc.tolist(),
        "energy": energy.tolist(),
        "raw_stack": arr.tolist(),
    }


def extract_mfcc_chunks(audio: np.ndarray, max_seconds: float = 20.0):
    """Extract MFCC chunks from audio (same as AudioFileSensor)."""
    chunk_samples = int(SAMPLE_RATE / AUDIO_HZ)
    end_sample = int(min(max_seconds, len(audio) / SAMPLE_RATE) * SAMPLE_RATE)
    chunks = []
    pos = 0
    consecutive_silent = 0
    while pos + chunk_samples <= end_sample:
        chunk = audio[pos:pos + chunk_samples]
        feat = compute_mfcc(chunk)
        timestamp = pos / SAMPLE_RATE
        pos += chunk_samples
        if feat is None:
            continue
        if is_silent(feat):
            consecutive_silent += 1
            if consecutive_silent > 10:
                continue
        else:
            consecutive_silent = 0
        chunks.append((timestamp, feat))
    return chunks


def simulate_old_pipeline(mfcc_chunks, flush_hz=2.0):
    """OLD pipeline: keep only LAST MFCC chunk per flush window."""
    flush_interval = 1.0 / flush_hz
    if not mfcc_chunks:
        return []

    results = []
    buffer_latest = None
    buffer_count = 0
    flush_time = flush_interval

    for ts, feat in mfcc_chunks:
        while ts > flush_time:
            if buffer_latest is not None:
                results.append({
                    "flush_time": flush_time,
                    "features": buffer_latest.copy(),
                    "chunks_in_window": buffer_count,
                    "chunks_discarded": buffer_count - 1,
                })
            buffer_latest = None
            buffer_count = 0
            flush_time += flush_interval
        buffer_latest = feat
        buffer_count += 1

    # Final flush
    if buffer_latest is not None:
        results.append({
            "flush_time": flush_time,
            "features": buffer_latest.copy(),
            "chunks_in_window": buffer_count,
            "chunks_discarded": buffer_count - 1,
        })
    return results


def simulate_new_pipeline(mfcc_chunks, flush_hz=2.0):
    """NEW pipeline: accumulate ALL chunks, compute temporal audio frame."""
    flush_interval = 1.0 / flush_hz
    if not mfcc_chunks:
        return []

    results = []
    buffer = []
    flush_time = flush_interval

    for ts, feat in mfcc_chunks:
        while ts > flush_time:
            if buffer:
                frame = compute_temporal_audio_frame([f.tolist() for f in buffer])
                results.append({
                    "flush_time": flush_time,
                    "temporal_frame": frame,
                    "chunks_in_window": len(buffer),
                    "chunks_used": len(buffer),  # ALL chunks used
                })
            buffer = []
            flush_time += flush_interval
        buffer.append(feat)

    # Final flush
    if buffer:
        frame = compute_temporal_audio_frame([f.tolist() for f in buffer])
        results.append({
            "flush_time": flush_time,
            "temporal_frame": frame,
            "chunks_in_window": len(buffer),
            "chunks_used": len(buffer),
        })
    return results


def simulate_stm_pipeline(mfcc_chunks, flush_hz=2.0, brain_hz=0.74):
    """Full Solution 5 pipeline: aggregator temporal frame + AuditorySTM."""
    # First, run the aggregator
    new_flushes = simulate_new_pipeline(mfcc_chunks, flush_hz)
    if not new_flushes:
        return []

    stm = AuditorySTM(AuditorySTMConfig(window_steps=8, decay_rate=0.85))
    brain_step_interval = 1.0 / brain_hz

    results = []
    flush_idx = 0
    step_time = brain_step_interval

    while flush_idx < len(new_flushes):
        # Collect all flushes that happened before this brain step
        latest_frame = None
        while flush_idx < len(new_flushes) and new_flushes[flush_idx]["flush_time"] <= step_time:
            latest_frame = new_flushes[flush_idx]["temporal_frame"]
            flush_idx += 1

        # Feed to STM and get features
        if latest_frame is not None:
            stm.update(latest_frame)
            features = stm.get_features()
            if features is not None:
                results.append({
                    "step_time": step_time,
                    "features": features,
                    "stm_frames": stm.n_frames,
                    "feature_length": len(features),
                })

        step_time += brain_step_interval

        # Safety: don't run forever
        if step_time > new_flushes[-1]["flush_time"] + 2 * brain_step_interval:
            break

    return results


def main():
    test_dir = Path(__file__).parent
    videos = list(test_dir.glob("*.mp4"))
    if not videos:
        print("ERROR: No .mp4 files found in", test_dir)
        sys.exit(1)

    # Setup encoder for spike pattern comparison
    config = NeuromorphicConfig()
    config.populations.sensory_cortex = 1000
    encoder = SpikeEncoder(config, rng=np.random.default_rng(42))
    sensory = SensoryCortex(config)

    for video_path in videos:
        print(f"\n{'='*80}")
        print(f"SOLUTION 5 QUALITY TEST: {video_path.name}")
        print(f"{'='*80}")

        # Extract audio
        audio = extract_audio(str(video_path))
        audio_duration = len(audio) / SAMPLE_RATE
        print(f"\nAudio: {audio_duration:.1f}s")

        # Extract MFCC chunks
        mfcc_chunks = extract_mfcc_chunks(audio)
        print(f"MFCC chunks: {len(mfcc_chunks)}")

        # ====== OLD PIPELINE ======
        print(f"\n--- OLD Pipeline (latest-only) ---")
        old_flushes = simulate_old_pipeline(mfcc_chunks)
        old_total_chunks = sum(f["chunks_in_window"] for f in old_flushes)
        old_discarded = sum(f["chunks_discarded"] for f in old_flushes)
        print(f"  Flushes: {len(old_flushes)}")
        print(f"  Chunks: {old_total_chunks} total, {old_discarded} discarded "
              f"({old_discarded/max(1,old_total_chunks)*100:.0f}% loss)")
        print(f"  Features per flush: 13 floats (single MFCC snapshot)")

        # Cross-flush cosine similarity (how different are consecutive observations?)
        if len(old_flushes) >= 2:
            old_sims = []
            for i in range(len(old_flushes) - 1):
                a, b = old_flushes[i]["features"], old_flushes[i+1]["features"]
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                old_sims.append(float(cos))
            print(f"  Cross-flush cosine sim: mean={np.mean(old_sims):.4f}, "
                  f"std={np.std(old_sims):.4f}")
            print(f"  Distinct transitions (cos < 0.8): "
                  f"{sum(1 for s in old_sims if s < 0.8)}/{len(old_sims)}")

        # ====== NEW PIPELINE (aggregator only) ======
        print(f"\n--- NEW Pipeline (temporal audio frame) ---")
        new_flushes = simulate_new_pipeline(mfcc_chunks)
        new_total_chunks = sum(f["chunks_in_window"] for f in new_flushes)
        new_used = sum(f["chunks_used"] for f in new_flushes)
        print(f"  Flushes: {len(new_flushes)}")
        print(f"  Chunks: {new_total_chunks} total, ALL {new_used} used (0% loss)")

        if new_flushes:
            sample = new_flushes[0]["temporal_frame"]
            n_features = len(sample["mean_mfcc"]) + len(sample["delta_mfcc"]) + len(sample["energy"])
            print(f"  Features per flush: {n_features} floats "
                  f"(13 mean + 13 delta + {len(sample['energy'])} energy)")

        # Cross-flush similarity for temporal frames (use fixed-length mean+delta only)
        if len(new_flushes) >= 2:
            new_sims = []
            for i in range(len(new_flushes) - 1):
                a_frame = new_flushes[i]["temporal_frame"]
                b_frame = new_flushes[i+1]["temporal_frame"]
                # Use only mean+delta (26 floats) — energy varies in length per flush
                a_vec = np.concatenate([
                    a_frame["mean_mfcc"], a_frame["delta_mfcc"]
                ])
                b_vec = np.concatenate([
                    b_frame["mean_mfcc"], b_frame["delta_mfcc"]
                ])
                cos = np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec) + 1e-10)
                new_sims.append(float(cos))
            print(f"  Cross-flush cosine sim: mean={np.mean(new_sims):.4f}, "
                  f"std={np.std(new_sims):.4f}")
            print(f"  Distinct transitions (cos < 0.8): "
                  f"{sum(1 for s in new_sims if s < 0.8)}/{len(new_sims)}")

        # ====== FULL SOLUTION 5 (aggregator + STM) ======
        print(f"\n--- FULL Solution 5 (aggregator + AuditorySTM @ 0.74 Hz brain) ---")
        stm_results = simulate_stm_pipeline(mfcc_chunks)
        print(f"  Brain steps with audio: {len(stm_results)}")
        if stm_results:
            lengths = [r["feature_length"] for r in stm_results]
            print(f"  Feature vector length: {lengths[0]} floats "
                  f"(consistent: {len(set(lengths)) == 1})")
            stm_frames = [r["stm_frames"] for r in stm_results]
            print(f"  STM buffer depth: min={min(stm_frames)}, max={max(stm_frames)} frames")

        # Cross-step similarity for STM features
        if len(stm_results) >= 2:
            stm_sims = []
            for i in range(len(stm_results) - 1):
                a, b = stm_results[i]["features"], stm_results[i+1]["features"]
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                stm_sims.append(float(cos))
            print(f"  Cross-step cosine sim: mean={np.mean(stm_sims):.4f}, "
                  f"std={np.std(stm_sims):.4f}")
            print(f"  Distinct transitions (cos < 0.8): "
                  f"{sum(1 for s in stm_sims if s < 0.8)}/{len(stm_sims)}")

        # ====== SPIKE PATTERN COMPARISON ======
        print(f"\n--- Spike Pattern Comparison (encoder output) ---")

        # Encode a few old pipeline observations
        old_spikes = []
        for f in old_flushes[:10]:
            current = encoder.encode(sensory, f["features"].tolist(), "sensor.audiofile")
            aud_sr = sensory.get_subrange("auditory")
            aud_current = current[aud_sr.slice()]
            old_spikes.append(aud_current)

        # Encode a few STM pipeline observations
        stm_spikes = []
        for r in stm_results[:10]:
            current = encoder.encode(sensory, r["features"].tolist(), "sensor.audiofile")
            aud_sr = sensory.get_subrange("auditory")
            aud_current = current[aud_sr.slice()]
            stm_spikes.append(aud_current)

        if old_spikes and stm_spikes:
            # Compare active neuron counts
            old_active = [np.count_nonzero(s > 0) for s in old_spikes]
            stm_active = [np.count_nonzero(s > 0) for s in stm_spikes]
            print(f"  Active neurons (old): mean={np.mean(old_active):.0f}")
            print(f"  Active neurons (STM): mean={np.mean(stm_active):.0f}")

            # Compare pattern entropy (proxy for information content)
            def pattern_entropy(spikes_list):
                entropies = []
                for s in spikes_list:
                    if s.sum() == 0:
                        entropies.append(0.0)
                        continue
                    p = s / s.sum()
                    p = p[p > 0]
                    entropies.append(float(-np.sum(p * np.log2(p))))
                return entropies

            old_ent = pattern_entropy(old_spikes)
            stm_ent = pattern_entropy(stm_spikes)
            print(f"  Pattern entropy (old): mean={np.mean(old_ent):.2f} bits")
            print(f"  Pattern entropy (STM): mean={np.mean(stm_ent):.2f} bits")

            # Compare cross-observation discriminability
            def cross_discriminability(spikes_list):
                if len(spikes_list) < 2:
                    return []
                sims = []
                for i in range(len(spikes_list) - 1):
                    a, b = spikes_list[i], spikes_list[i+1]
                    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
                    if norm_a > 0 and norm_b > 0:
                        cos = np.dot(a, b) / (norm_a * norm_b)
                        sims.append(float(cos))
                return sims

            old_disc = cross_discriminability(old_spikes)
            stm_disc = cross_discriminability(stm_spikes)
            if old_disc and stm_disc:
                print(f"  Spike pattern discriminability (lower = more distinct):")
                print(f"    Old: mean_cos={np.mean(old_disc):.4f}")
                print(f"    STM: mean_cos={np.mean(stm_disc):.4f}")

        # ====== VERDICT ======
        print(f"\n{'='*80}")
        print(f"VERDICT for {video_path.name}")
        print(f"{'='*80}")
        if old_flushes:
            old_loss = old_discarded / max(1, old_total_chunks) * 100
            print(f"\n  OLD pipeline: {old_loss:.0f}% audio data loss")
            print(f"  NEW pipeline: 0% audio data loss")
            if len(old_flushes) >= 2 and len(stm_results) >= 2:
                old_distinct = sum(1 for s in old_sims if s < 0.8) / len(old_sims) * 100
                stm_distinct = sum(1 for s in stm_sims if s < 0.8) / len(stm_sims) * 100
                print(f"  OLD distinct transitions: {old_distinct:.0f}%")
                print(f"  STM distinct transitions: {stm_distinct:.0f}%")
                if stm_results:
                    print(f"  Feature richness: {stm_results[0]['feature_length']} vs 13 floats "
                          f"({stm_results[0]['feature_length']/13:.0f}x)")
                if stm_ent and old_ent:
                    improvement = (np.mean(stm_ent) - np.mean(old_ent)) / max(0.01, np.mean(old_ent)) * 100
                    print(f"  Spike entropy improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
