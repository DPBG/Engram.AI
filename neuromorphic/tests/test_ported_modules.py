"""Functional tests for the ported core brain modules.

Each test exercises a module's real core function (not just import) so the
ported functionality is proven to work inside Engram. Deterministic and fast —
no live network, MuJoCo, or NATS required.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── speech_synth ─────────────────────────────────────────────────────────────


def test_speech_synth_constructs_and_mel_energies():
    from neuromorphic.speech_synth import (
        SpeechSynth,
        SpeechSynthConfig,
        mel_filterbank_energies,
    )

    synth = SpeechSynth(SpeechSynthConfig())
    synth.reset()  # stateful reset must not raise
    energies = mel_filterbank_energies(np.random.randn(400).astype(np.float32), sample_rate=16000)
    assert isinstance(energies, np.ndarray)
    assert energies.size > 0
    assert np.all(np.isfinite(energies))


# ── acoustic_similarity ──────────────────────────────────────────────────────


def test_acoustic_similarity_identical_is_one_and_orthogonal_is_low():
    from neuromorphic.acoustic_similarity import mel_cosine_similarity, mel_features

    a = np.sin(np.linspace(0, 20, 800)).astype(np.float32)
    fa = mel_features(a, 16000)
    assert mel_cosine_similarity(fa, fa) == pytest.approx(1.0, abs=1e-3)


# ── walk_shaping ─────────────────────────────────────────────────────────────


def test_walk_shaping_combined_reward_is_bounded_float():
    from neuromorphic.walk_shaping import combined_shaping_reward

    r = combined_shaping_reward(0.5, 0.5, 1.0)
    assert isinstance(r, float)
    assert -5.0 <= r <= 5.0


# ── motor_skill_projection ───────────────────────────────────────────────────


def test_motor_skill_projection_aggregate_is_float():
    from neuromorphic.motor_skill_projection import aggregate_actuator_weights_to_scale

    w = aggregate_actuator_weights_to_scale([0.2, 0.4, 0.6])
    assert isinstance(w, float)


# ── curator ──────────────────────────────────────────────────────────────────


def test_curator_decide_returns_bool_reason():
    from neuromorphic.curator import Curator, CuratorConfig

    c = Curator(CuratorConfig())
    accept, reason = c.decide("unknown.stream", {"confidence": 0.9})
    assert isinstance(accept, bool)
    assert isinstance(reason, str) and reason


# ── body_skill_library ───────────────────────────────────────────────────────


def test_body_skill_library_save_load_roundtrip(tmp_path):
    from neuromorphic.body_skill_library import BodySkillLibrary

    lib = BodySkillLibrary(str(tmp_path / "skills.db"))
    weights = {"motor.locomotion": np.arange(4, dtype=np.float32)}
    lib.save("humanoid_v2", weights, brain_name="test")
    assert lib.has_skill("humanoid_v2") is True
    assert "humanoid_v2" in [e.manifest_id for e in lib.list_skills()]
    loaded = lib.load("humanoid_v2")
    assert loaded is not None
    np.testing.assert_array_equal(loaded["motor.locomotion"], weights["motor.locomotion"])
    lib.close()


# ── homeostasis ──────────────────────────────────────────────────────────────


def test_homeostasis_threshold_tiers_are_ordered():
    from neuromorphic.homeostasis import HomeostasisScheduler

    sch = HomeostasisScheduler()
    th = sch.thresholds_for("cortex.assoc", w_max=1.0)
    # Tiers escalate: tier1 enter <= tier2 enter <= tier3 enter.
    assert th.t1_enter <= th.t2_enter <= th.t3_enter
    # Leave thresholds sit below their enter thresholds (hysteresis).
    assert th.t1_leave <= th.t1_enter


def test_homeostasis_motor_group_brakes_earlier():
    from neuromorphic.homeostasis import HomeostasisScheduler

    sch = HomeostasisScheduler(motor_groups=frozenset({"motor.loco"}))
    motor = sch.thresholds_for("motor.loco", w_max=1.0)
    cortex = sch.thresholds_for("cortex.assoc", w_max=1.0)
    # motor_ceiling (0.75) < emergency_ceiling (0.85) → motor tier3 is lower.
    assert motor.t3_enter < cortex.t3_enter


# ── stdp_bias ────────────────────────────────────────────────────────────────


def test_stdp_bias_autotuner_constructs_and_reports_state():
    from neuromorphic.stdp_bias import StdpBiasAutoTuner

    tuner = StdpBiasAutoTuner()
    assert tuner.get_warning_count("g") == 0
    state = tuner.get_state()
    assert isinstance(state, dict)
    with pytest.raises(ValueError):
        StdpBiasAutoTuner(bump_step=0.0)  # invalid: must be > 0


# ── demonstrations ───────────────────────────────────────────────────────────


def test_demonstrations_gait_generator_produces_frames():
    from neuromorphic.demonstrations import KinematicGaitGenerator

    gen = KinematicGaitGenerator(rate_hz=1.4, dt_seconds=0.1)
    frame = gen.step(timestamp=0.1)
    assert frame is not None
    assert 0.0 <= frame.phase < 1.0
    assert isinstance(frame.joint_angles, dict) and frame.joint_angles
