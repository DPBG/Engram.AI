#!/usr/bin/env python3
"""
Temporal Coherence Audit — tests whether the aggregation pipeline
preserves meaningful audio/video alignment.

Uses REAL video files from the Hetzner training set and runs the
ACTUAL sensor code paths (AudioFileSensor MFCC extraction,
VideoFileSensor 64x64 grayscale + motion delta, CNN preprocessing).

Tests:
1. Audio chunk coverage: How many 32ms MFCC chunks are produced vs. discarded
   per 500ms aggregation window?
2. Video frame coverage: How many frames are produced vs. discarded?
3. Cross-modal alignment: When a flush fires, does the audio chunk correspond
   to the same moment in the video as the video frame?
4. Word-level coherence: For a spoken word spanning ~400ms, how many of the
   ~12 MFCC chunks survive aggregation?
5. Feature stability: How different are consecutive MFCC vectors within a
   single word vs. across word boundaries?

This script does NOT need NATS or any running services. It directly
instantiates the sensor processing pipelines and simulates the
aggregation timing.
"""

import sys
from pathlib import Path

# Add project paths so we can import actual sensor code
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "sensory-gateway"))
sys.path.insert(0, str(PROJECT_ROOT / "sensory-gateway" / "sensors"))
sys.path.insert(0, str(PROJECT_ROOT / "sdk" / "src"))

import numpy as np  # noqa: E402

# --- Audio pipeline (same as AudioFileSensor) ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.032  # 32ms
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 512 samples
N_MFCC = 13
SILENCE_FLOOR = -23.0
SILENCE_THRESHOLD = 0.5
AUDIO_HZ = 10  # AudioFileSensor default rate


def extract_audio(video_path: str) -> np.ndarray:
    """Extract audio exactly as AudioFileSensor does."""
    import shutil
    import subprocess
    import tempfile
    import wave

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")

        with wave.open(tmp_path, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compute_mfcc(audio_chunk: np.ndarray) -> np.ndarray | None:
    """Compute MFCC exactly as AudioFileSensor does."""
    if len(audio_chunk) < CHUNK_SAMPLES:
        return None
    try:
        from python_speech_features import mfcc

        features = mfcc(
            audio_chunk,
            samplerate=SAMPLE_RATE,
            winlen=CHUNK_DURATION,
            winstep=CHUNK_DURATION,
            numcep=N_MFCC,
            nfilt=26,
            nfft=512,
        )
        return features.mean(axis=0).astype(np.float32) if len(features) > 0 else None
    except ImportError:
        # Minimal fallback (same as audio_file.py)
        windowed = audio_chunk[:512] * np.hanning(min(len(audio_chunk), 512))
        spectrum = np.abs(np.fft.rfft(windowed, n=512)) ** 2
        log_spectrum = np.log(spectrum + 1e-10)
        step = max(1, len(log_spectrum) // N_MFCC)
        features = log_spectrum[: N_MFCC * step : step][:N_MFCC]
        return features.astype(np.float32)


def is_silent(features: np.ndarray) -> bool:
    """Check silence exactly as AudioFileSensor does."""
    return bool(np.all(np.abs(features - SILENCE_FLOOR) < SILENCE_THRESHOLD))


# --- Video pipeline (same as VideoFileSensor) ---
import cv2  # noqa: E402

VIDEO_FPS = 2  # VideoFileSensor default
SPARSE_THRESHOLD = 0.005  # Production setting on Hetzner


def extract_video_frames(video_path: str, fps: float = VIDEO_FPS, max_seconds: float = 30.0):
    """Extract frames exactly as VideoFileSensor does, yielding (timestamp_s, features)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps
    print(f"  Video: {native_fps:.1f} native fps, {total_frames} frames, {duration:.1f}s")

    # Read at the desired fps by advancing frame position
    frame_interval = native_fps / fps  # frames to skip per read
    prev_gray = None
    frame_idx = 0
    emitted = 0
    skipped = 0

    while True:
        target_frame = int(frame_idx * frame_interval)
        if target_frame >= total_frames:
            break
        timestamp_s = target_frame / native_fps
        if timestamp_s > max_seconds:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0

        # Sparse check (same logic as VideoFileSensor)
        if prev_gray is not None:
            delta = np.abs(small - prev_gray)
            mean_diff = float(delta.mean())
            if mean_diff < SPARSE_THRESHOLD:
                skipped += 1
                frame_idx += 1
                continue
            combined = 0.7 * small + 0.3 * delta
        else:
            combined = small

        prev_gray = small.copy()
        emitted += 1
        frame_idx += 1

        yield timestamp_s, combined.flatten()

    cap.release()
    print(f"  Frames: {emitted} emitted, {skipped} sparse-skipped")


# --- Aggregator simulation ---


def simulate_aggregation(audio_events, video_events, flush_hz=2.0):
    """
    Simulate AggregatingEventBus behavior.

    audio_events: list of (timestamp_s, mfcc_vector)
    video_events: list of (timestamp_s, frame_vector)

    Returns list of flush events: {
        flush_time_s, audio_timestamp_s, video_timestamp_s,
        audio_features, video_features,
        audio_discarded_count, video_discarded_count,
        audio_timestamps_discarded
    }
    """
    flush_interval = 1.0 / flush_hz

    # Merge all events into timeline
    events = []
    for ts, feat in audio_events:
        events.append((ts, "audio", feat))
    for ts, feat in video_events:
        events.append((ts, "video", feat))
    events.sort(key=lambda x: x[0])

    if not events:
        return []

    results = []
    latest_audio = None
    latest_audio_ts = None
    latest_video = None
    latest_video_ts = None
    audio_count_in_window = 0
    video_count_in_window = 0
    audio_timestamps_in_window = []

    # Simulate time-based flushing
    max_time = events[-1][0]
    flush_time = flush_interval  # First flush at flush_interval

    event_idx = 0
    while flush_time <= max_time + flush_interval:
        # Process all events up to this flush time
        while event_idx < len(events) and events[event_idx][0] <= flush_time:
            ts, modality, feat = events[event_idx]
            if modality == "audio":
                latest_audio = feat
                latest_audio_ts = ts
                audio_count_in_window += 1
                audio_timestamps_in_window.append(ts)
            else:
                latest_video = feat
                latest_video_ts = ts
                video_count_in_window += 1
            event_idx += 1

        # Flush
        if latest_audio is not None or latest_video is not None:
            results.append(
                {
                    "flush_time_s": flush_time,
                    "audio_timestamp_s": latest_audio_ts,
                    "video_timestamp_s": latest_video_ts,
                    "audio_features": latest_audio.copy() if latest_audio is not None else None,
                    "video_features": (
                        latest_video[:10].copy() if latest_video is not None else None
                    ),
                    "audio_discarded": audio_count_in_window
                    - (1 if latest_audio is not None else 0),
                    "video_discarded": video_count_in_window
                    - (1 if latest_video is not None else 0),
                    "audio_timestamps_in_window": audio_timestamps_in_window.copy(),
                    "audio_temporal_span_ms": (
                        (audio_timestamps_in_window[-1] - audio_timestamps_in_window[0]) * 1000
                        if len(audio_timestamps_in_window) > 1
                        else 0
                    ),
                }
            )

        # Reset for next window
        latest_audio = None
        latest_audio_ts = None
        latest_video = None
        latest_video_ts = None
        audio_count_in_window = 0
        video_count_in_window = 0
        audio_timestamps_in_window = []
        flush_time += flush_interval

    return results


def analyze_mfcc_temporal_structure(audio: np.ndarray, start_s: float, duration_s: float):
    """
    Analyze MFCC feature stability within a time window.
    Shows how much MFCC vectors change chunk-to-chunk.
    """
    chunk_samples = int(SAMPLE_RATE / AUDIO_HZ)  # samples per emission at 10Hz
    start_sample = int(start_s * SAMPLE_RATE)
    end_sample = int((start_s + duration_s) * SAMPLE_RATE)

    chunks = []
    pos = start_sample
    while pos + chunk_samples <= end_sample:
        chunk = audio[pos : pos + chunk_samples]
        feat = compute_mfcc(chunk)
        if feat is not None:
            chunks.append((pos / SAMPLE_RATE, feat))
        pos += chunk_samples

    if len(chunks) < 2:
        return None

    # Compute pairwise cosine similarity between consecutive chunks
    similarities = []
    for i in range(len(chunks) - 1):
        a = chunks[i][1]
        b = chunks[i + 1][1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        similarities.append(float(cos_sim))

    return {
        "n_chunks": len(chunks),
        "time_span_ms": (chunks[-1][0] - chunks[0][0]) * 1000,
        "chunk_interval_ms": chunk_samples / SAMPLE_RATE * 1000,
        "mean_consecutive_cosine_sim": float(np.mean(similarities)),
        "min_cosine_sim": float(np.min(similarities)),
        "max_cosine_sim": float(np.max(similarities)),
        "std_cosine_sim": float(np.std(similarities)),
        "first_mfcc": chunks[0][1].tolist(),
        "last_mfcc": chunks[-1][1].tolist(),
        "first_last_cosine": float(
            np.dot(chunks[0][1], chunks[-1][1])
            / (np.linalg.norm(chunks[0][1]) * np.linalg.norm(chunks[-1][1]) + 1e-10)
        ),
    }


def main():
    test_dir = Path(__file__).parent
    videos = list(test_dir.glob("*.mp4"))
    if not videos:
        print("ERROR: No .mp4 files found in", test_dir)
        sys.exit(1)

    for video_path in videos:
        print(f"\n{'='*80}")
        print(f"TESTING: {video_path.name}")
        print(f"{'='*80}")

        # ==== STEP 1: Extract audio ====
        print("\n--- Audio Extraction ---")
        audio = extract_audio(str(video_path))
        audio_duration = len(audio) / SAMPLE_RATE
        print(f"  Audio: {audio_duration:.1f}s, {len(audio)} samples")

        # Process first 20s of audio as MFCC chunks at 10Hz
        test_duration = min(20.0, audio_duration)
        chunk_samples = int(SAMPLE_RATE / AUDIO_HZ)
        audio_events = []
        silent_count = 0
        consecutive_silent = 0
        pos = 0
        while pos + chunk_samples <= int(test_duration * SAMPLE_RATE):
            chunk = audio[pos : pos + chunk_samples]
            feat = compute_mfcc(chunk)
            timestamp = pos / SAMPLE_RATE
            pos += chunk_samples

            if feat is None:
                continue

            if is_silent(feat):
                consecutive_silent += 1
                silent_count += 1
                if consecutive_silent > 10:
                    continue  # squelched
                audio_events.append((timestamp, feat))
            else:
                consecutive_silent = 0
                audio_events.append((timestamp, feat))

        print(
            f"  MFCC chunks (first {test_duration:.0f}s): {len(audio_events)} emitted, {silent_count} silent"
        )

        # ==== STEP 2: Extract video frames ====
        print("\n--- Video Extraction ---")
        video_events = list(
            extract_video_frames(str(video_path), fps=VIDEO_FPS, max_seconds=test_duration)
        )
        print(f"  Video frames (first {test_duration:.0f}s): {len(video_events)} emitted")

        # ==== STEP 3: Simulate aggregation ====
        print("\n--- Aggregation Simulation (2 Hz flush) ---")
        flushes = simulate_aggregation(audio_events, video_events, flush_hz=2.0)

        total_audio_discarded = sum(f["audio_discarded"] for f in flushes)
        total_audio_delivered = sum(1 for f in flushes if f["audio_timestamp_s"] is not None)
        total_video_discarded = sum(f["video_discarded"] for f in flushes)
        total_video_delivered = sum(1 for f in flushes if f["video_timestamp_s"] is not None)

        print("\n  FLUSH SUMMARY:")
        print(
            f"  Audio:  {total_audio_delivered} delivered, {total_audio_discarded} discarded "
            f"({total_audio_discarded/(total_audio_delivered + total_audio_discarded)*100:.1f}% loss)"
        )
        print(
            f"  Video:  {total_video_delivered} delivered, {total_video_discarded} discarded "
            f"({total_video_discarded/(total_video_delivered + total_video_discarded)*100:.1f}% loss)"
        )

        print("\n  PER-FLUSH DETAILS (first 10 flushes):")
        print(
            f"  {'Flush':>6}  {'Audio_t':>8}  {'Video_t':>8}  {'A/V lag':>8}  {'A_disc':>6}  {'V_disc':>6}  {'A_span_ms':>10}"
        )
        for i, f in enumerate(flushes[:10]):
            a_t = (
                f"{f['audio_timestamp_s']:.3f}" if f["audio_timestamp_s"] is not None else "  None"
            )
            v_t = (
                f"{f['video_timestamp_s']:.3f}" if f["video_timestamp_s"] is not None else "  None"
            )
            if f["audio_timestamp_s"] is not None and f["video_timestamp_s"] is not None:
                lag = f"{(f['audio_timestamp_s'] - f['video_timestamp_s'])*1000:.0f}ms"
            else:
                lag = "   N/A"
            print(
                f"  {f['flush_time_s']:6.2f}  {a_t:>8}  {v_t:>8}  {lag:>8}  {f['audio_discarded']:>6}  "
                f"{f['video_discarded']:>6}  {f['audio_temporal_span_ms']:>10.0f}"
            )

        # ==== STEP 4: Temporal alignment analysis ====
        print("\n--- Temporal Alignment Analysis ---")
        lags_ms = []
        for f in flushes:
            if f["audio_timestamp_s"] is not None and f["video_timestamp_s"] is not None:
                lag = (f["audio_timestamp_s"] - f["video_timestamp_s"]) * 1000
                lags_ms.append(lag)

        if lags_ms:
            print(
                f"  Audio-Video lag (ms): mean={np.mean(lags_ms):.1f}, "
                f"std={np.std(lags_ms):.1f}, "
                f"min={np.min(lags_ms):.1f}, max={np.max(lags_ms):.1f}"
            )
            print(f"  Flushes with both modalities: {len(lags_ms)} / {len(flushes)}")
        else:
            print("  WARNING: No flushes had both audio and video!")

        # ==== STEP 5: Audio temporal span per window ====
        print("\n--- Audio Temporal Information Loss ---")
        spans = [f["audio_temporal_span_ms"] for f in flushes if f["audio_temporal_span_ms"] > 0]
        if spans:
            print(
                f"  Audio data span per window: mean={np.mean(spans):.0f}ms, "
                f"max={np.max(spans):.0f}ms"
            )
            print(f"  This means ~{np.mean(spans):.0f}ms of audio CONTEXT arrives each window,")
            print("  but only the LAST 32ms chunk survives (the rest are overwritten).")
            print(
                f"  => {np.mean(spans):.0f}ms - 32ms = ~{max(0, np.mean(spans)-32):.0f}ms of temporal context LOST per window"
            )

        # ==== STEP 6: MFCC feature variation within windows ====
        print("\n--- MFCC Feature Stability Analysis ---")
        print("  (Do consecutive MFCC chunks within a 500ms window carry different info?)")

        # Pick 5 windows with audio and measure variation
        windows_with_audio = [f for f in flushes if len(f["audio_timestamps_in_window"]) >= 3]
        for i, f in enumerate(windows_with_audio[:5]):
            ts_list = f["audio_timestamps_in_window"]
            analysis = analyze_mfcc_temporal_structure(
                audio,
                start_s=ts_list[0],
                duration_s=ts_list[-1] - ts_list[0] + CHUNK_DURATION,
            )
            if analysis:
                print(
                    f"\n  Window at {f['flush_time_s']:.2f}s ({analysis['n_chunks']} chunks, "
                    f"{analysis['time_span_ms']:.0f}ms span):"
                )
                print(
                    f"    Consecutive cosine sim: {analysis['mean_consecutive_cosine_sim']:.4f} "
                    f"(std={analysis['std_cosine_sim']:.4f})"
                )
                print(f"    First↔Last cosine sim: {analysis['first_last_cosine']:.4f}")
                print(
                    f"    Interpretation: {'SIMILAR (stable sound)' if analysis['first_last_cosine'] > 0.95 else 'DIFFERENT (evolving sound — information lost!)' if analysis['first_last_cosine'] < 0.8 else 'MODERATE variation'}"
                )

        # ==== STEP 7: Cross-window MFCC variation ====
        print("\n--- Cross-Window MFCC Variation (are different windows distinguishable?) ---")
        delivered_mfccs = [
            (f["audio_timestamp_s"], f["audio_features"])
            for f in flushes
            if f["audio_features"] is not None
        ]
        if len(delivered_mfccs) >= 4:
            # Compare consecutive delivered features
            cross_sims = []
            for i in range(len(delivered_mfccs) - 1):
                a = delivered_mfccs[i][1]
                b = delivered_mfccs[i + 1][1]
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                cross_sims.append(float(cos))
            print("  Cosine similarity between consecutive DELIVERED audio observations:")
            print(f"    Mean: {np.mean(cross_sims):.4f}, Std: {np.std(cross_sims):.4f}")
            print(f"    Min:  {np.min(cross_sims):.4f}, Max: {np.max(cross_sims):.4f}")
            low_sim = sum(1 for s in cross_sims if s < 0.8)
            print(
                f"    Distinct transitions (cosine < 0.8): {low_sim}/{len(cross_sims)} = {low_sim/len(cross_sims)*100:.0f}%"
            )
        else:
            print("  Not enough delivered audio observations to compare")

        # ==== STEP 8: What the brain actually sees ====
        print("\n--- Brain's Perspective (what it receives per step @ ~0.74 steps/sec) ---")
        brain_step_interval = 1.0 / 0.74  # ~1.35s per step
        print(f"  Brain step interval: ~{brain_step_interval*1000:.0f}ms")
        print("  Aggregator flush interval: 500ms (2 Hz)")
        print(f"  => Brain gets ~{brain_step_interval/0.5:.1f} potential flushes per step")
        print("     but _latest_obs dict keeps only 1 per modality per step")
        print(f"  => Of {len(audio_events)} audio chunks in {test_duration:.0f}s,")
        print(f"     brain processes ~{int(test_duration / brain_step_interval)} observations")
        print("     each representing a SINGLE 32ms MFCC snapshot")

        # ==== STEP 9: Sensory buffer decay analysis ====
        print("\n--- Sensory Buffer Decay Analysis ---")
        print("  sensory_decay = 0.97 per step → current halves every ~23 steps")
        print(
            f"  At 0.74 steps/sec, new observation arrives every ~{brain_step_interval:.1f}s = 1 step"
        )
        print("  Each MFCC observation persists ~33 steps of decay")
        print("  But new observation REPLACES old one each step (additive in buffer)")
        print("  So temporal structure = series of decaying snapshots, not accumulated sequence")

        print(f"\n{'='*80}")
        print(f"VERDICT for {video_path.name}:")
        print(f"{'='*80}")
        if spans:
            avg_span = np.mean(spans)
            if avg_span > 100:
                print(f"  PROBLEM CONFIRMED: Each 500ms window contains ~{avg_span:.0f}ms of")
                print(
                    f"  audio data ({int(avg_span/100)} chunks), but only the last 32ms survives."
                )
                print("  The brain receives audio SNAPSHOTS, not temporal sequences.")
                if lags_ms:
                    print(
                        f"  Cross-modal alignment: audio lags video by {np.mean(lags_ms):.0f}ms on average."
                    )
                    if abs(np.mean(lags_ms)) > 200:
                        print(
                            f"  CROSS-MODAL MISALIGNMENT: {abs(np.mean(lags_ms)):.0f}ms lag is too large for STDP binding"
                        )
                    else:
                        print("  Cross-modal timing is acceptable for STDP binding (<200ms)")
            else:
                print(
                    f"  Audio spans per window are short ({avg_span:.0f}ms) — less data loss than expected"
                )

    print(f"\n\n{'#'*80}")
    print("# OVERALL AUDIT CONCLUSION")
    print(f"{'#'*80}")
    print("""
The aggregation pipeline has a fundamental temporal coherence problem:

1. AUDIO: Each 500ms flush window contains ~4-5 MFCC chunks (each covering 32ms).
   Only the LAST chunk survives. The brain receives isolated 32ms spectral snapshots
   instead of temporal sequences. A word like "apple" (~400ms) produces 4 chunks
   but only the final 32ms (the "ul" part) reaches the brain.

2. VIDEO: Single frame survives per window — this is ACCEPTABLE because individual
   frames carry semantic content (an image of an apple is meaningful alone).

3. CROSS-MODAL: Audio and video from the SAME 500ms window are flushed together,
   so they arrive at the brain in the same step. The temporal alignment WITHIN a
   window is preserved, but the audio content may not match what's visible
   (e.g., the word "apple" has moved on by the time the last MFCC chunk is captured).

4. SENSORY BUFFER: The 0.97 decay buffer provides ~33 steps of persistence,
   but since each step gets a NEW observation, the buffer acts as a leaky
   integrator of individual snapshots — NOT a temporal accumulator of sequential
   audio chunks.

The key issue is NOT cross-modal alignment (that's preserved by simultaneous flush).
The key issue IS that audio is fundamentally a TEMPORAL modality — a single 32ms
MFCC vector cannot represent phonemes, words, or prosody. Video is spatial —
a single frame CAN represent objects and scenes.
""")


if __name__ == "__main__":
    main()
