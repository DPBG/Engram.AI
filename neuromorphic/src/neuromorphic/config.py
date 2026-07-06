"""All tunable parameters for the neuromorphic cognitive core."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Dendritic compartment indices — named constants for readability
COMP_APICAL_DISTAL = 0  # feedforward input (sensory/feature projections)
COMP_BASAL = 1  # recurrent/lateral connections
COMP_APICAL_PROXIMAL = 2  # top-down feedback (predictive/concept)
COMP_PERISOMATIC = 3  # modulatory (meta-controller, brainstem)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


@dataclass
class LIFParams:
    """Leaky integrate-and-fire parameters for a neuron population."""

    threshold: float = -55.0  # mV — spike threshold
    reset: float = -70.0  # mV — reset after spike
    resting: float = -65.0  # mV — resting potential
    tau: float = 20.0  # ms — membrane time constant
    refractory_ms: float = 2.0  # ms — absolute refractory period
    noise_std: float = 0.5  # mV — background noise
    # Spike Frequency Adaptation (SFA) — prevents runaway excitation.
    # After each spike, threshold increases by sfa_increment mV, then decays
    # exponentially with time constant 1/(1-sfa_decay) steps.
    # At 5% firing: ~0.5mV adaptation (negligible).
    # At 50% firing: ~5mV adaptation (halves effective drive).
    sfa_increment: float = 0.3  # mV threshold increase per spike
    sfa_decay: float = 0.99  # per-step decay (tau ~100 steps = 100ms)


@dataclass
class STDPParams:
    """Spike-timing-dependent plasticity parameters."""

    a_plus: float = 0.012  # LTP amplitude
    a_minus: float = 0.010  # LTD amplitude (weaker than LTP → prevents idle decay)
    tau_plus: float = 20.0  # ms — LTP time window
    tau_minus: float = 20.0  # ms — LTD time window
    w_min: float = 0.01  # minimum weight (prevents synaptic death)
    w_max: float = 1.0  # maximum weight
    min_dt: float = 1.0  # ms — ignore spike pairs closer than this (noise filter)
    max_dt: float = 50.0  # ms — ignore spike pairs farther than this (outside STDP window)

    def boosted(self, factor: float) -> STDPParams:
        """Return a copy with A_plus/A_minus scaled by factor."""
        return STDPParams(
            a_plus=self.a_plus * factor,
            a_minus=self.a_minus * factor,
            tau_plus=self.tau_plus,
            tau_minus=self.tau_minus,
            w_min=self.w_min,
            w_max=self.w_max,
            min_dt=self.min_dt,
            max_dt=self.max_dt,
        )


@dataclass
class RSTDPParams:
    """Reward-modulated STDP parameters (for predictive layer)."""

    a_plus: float = 0.012
    a_minus: float = 0.010
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    w_min: float = 0.01  # prevents synaptic death
    w_max: float = 1.0
    modulation_baseline: float = 1.0  # neutral modulation
    modulation_match: float = 0.5  # prediction match → small reinforcement
    modulation_mismatch: float = 3.0  # surprise → strong learning
    surprise_bonus: float = 0.002  # small positive dw bias when modulation > mismatch threshold


@dataclass
class SleepPhaseConfig:
    """Periodic offline sleep consolidation (ported for OSCEN core-function parity).

    OFF by default. When enabled, every ``interval_s`` seconds of wall-clock the
    brain enters a ``duration_s``-second consolidation window. During the window:
    sensory observation gain is dropped (less noise), the 5HT baseline is raised
    (replay-friendly neuromod profile), and eligibility traces are replayed with a
    HARD-CAPPED weight boost.

    The cap is critical: unbounded replay can multiply weights past ``w_max`` and
    overflow. The boost is capped at ``boost_factor`` AND the result is clipped to
    the synapse group's ``w_max`` — do not relax either gate.

    Consumed by ``neuromorphic.sleep_phase``.
    """

    enabled: bool = False  # NEURO_SLEEP_PHASE
    interval_s: float = 14400.0  # 4 hours between sleep windows
    duration_s: float = 600.0  # 10 min consolidation window
    boost_factor: float = 1.2  # max trace -> weight multiplier (HARD CAP)
    sensory_gain_during_sleep: float = 0.1  # drop external input to 10%
    serotonin_boost: float = 1.5  # multiplier on 5HT baseline during sleep


@dataclass
class PopulationConfig:
    """Size configuration for each brain region."""

    brainstem: int = 15_000
    reflex_arc: int = 10_000
    sensory_cortex: int = 200_000
    motor_cortex: int = 100_000
    cerebellum: int = 100_000
    association_cortex: int = 200_000
    predictive_layer: int = 100_000
    working_memory: int = 25_000
    # New hierarchical regions (0 = disabled for backward compat)
    feature_layer: int = 0
    concept_layer: int = 0
    pattern_separator: int = 0  # dentate gyrus analog
    meta_controller: int = 0
    global_workspace: int = 0  # GWT workspace (CIP-23)

    @property
    def total(self) -> int:
        return (
            self.brainstem
            + self.reflex_arc
            + self.sensory_cortex
            + self.motor_cortex
            + self.cerebellum
            + self.association_cortex
            + self.predictive_layer
            + self.working_memory
            + self.feature_layer
            + self.concept_layer
            + self.pattern_separator
            + self.meta_controller
            + self.global_workspace
        )


@dataclass
class ConnectionConfig:
    """Sparsity and initial weight for each synapse group."""

    # Hardwired (non-plastic)
    sensory_reflex_sparsity: float = 0.0005
    sensory_reflex_weight: float = 0.8
    reflex_motor_sparsity: float = 0.001
    reflex_motor_weight: float = 0.9
    brainstem_sensory_sparsity: float = 0.003  # was 0.0005 (6x more connections)
    brainstem_sensory_weight: float = 0.5  # was 0.2 (2.5x stronger)
    # Brainstem arousal to cortical regions (reticular activating system).
    # Non-plastic tonic excitation — brings cortex within firing range so
    # feedforward STDP pathways can learn.
    brainstem_association_sparsity: float = 0.003  # match brainstem_motor
    brainstem_association_weight: float = 1.5  # match other brainstem arousal groups
    brainstem_cerebellum_sparsity: float = 0.003
    brainstem_cerebellum_weight: float = 1.5  # non-plastic, bypasses w_max clipping
    brainstem_feature_sparsity: float = (
        0.005  # higher sparsity — feature needs more arousal (weak sensory input)
    )
    brainstem_feature_weight: float = 1.5  # non-plastic, bypasses w_max clipping
    brainstem_working_sparsity: float = 0.003
    brainstem_working_weight: float = 1.5  # non-plastic, bypasses w_max clipping
    brainstem_predictive_sparsity: float = 0.003
    brainstem_predictive_weight: float = 1.5  # non-plastic, bypasses w_max clipping

    # Plastic (STDP) — weights strong enough to drive downstream firing
    sensory_association_sparsity: float = 0.005
    sensory_association_weight: float = 0.5
    association_lateral_sparsity: float = 0.003
    association_lateral_weight: float = 0.3
    sensory_motor_sparsity: float = 0.002
    sensory_motor_weight: float = 0.5
    brainstem_motor_sparsity: float = 0.003
    brainstem_motor_weight: float = 0.5
    sensory_cerebellum_sparsity: float = 0.002
    sensory_cerebellum_weight: float = 0.5
    motor_cerebellum_sparsity: float = 0.002
    motor_cerebellum_weight: float = 0.4
    cerebellum_motor_sparsity: float = 0.002
    cerebellum_motor_weight: float = 0.4
    association_predictive_sparsity: float = 0.005
    association_predictive_weight: float = 0.4
    predictive_recurrent_sparsity: float = 0.003
    predictive_recurrent_weight: float = 0.3
    predictive_association_sparsity: float = 0.002
    predictive_association_weight: float = 0.3
    association_working_sparsity: float = 0.003
    association_working_weight: float = 0.4
    working_recurrent_sparsity: float = 0.003  # WM self-recurrence (CIP-24, NMDA attractors)
    working_recurrent_weight: float = 0.3
    working_motor_sparsity: float = 0.003
    working_motor_weight: float = 0.4

    # New hierarchical connections (used when feature/concept/meta layers enabled)
    sensory_feature_sparsity: float = 0.005
    sensory_feature_weight: float = 0.5
    feature_association_sparsity: float = 0.005
    feature_association_weight: float = 0.5
    association_concept_sparsity: float = 0.005
    association_concept_weight: float = 0.4
    concept_lateral_sparsity: float = 0.003
    concept_lateral_weight: float = 0.3
    concept_predictive_sparsity: float = 0.005
    concept_predictive_weight: float = 0.4
    concept_working_sparsity: float = 0.003
    concept_working_weight: float = 0.4
    predictive_concept_sparsity: float = 0.002
    predictive_concept_weight: float = 0.3
    brainstem_concept_sparsity: float = (
        0.005  # non-plastic tonic arousal (matches brainstem_feature)
    )
    brainstem_concept_weight: float = (
        1.5  # non-plastic, bypasses w_max (matches downstream arousal groups)
    )
    # Pattern separator (dentate gyrus) connections
    association_dg_sparsity: float = 0.005
    association_dg_weight: float = 0.5
    dg_concept_sparsity: float = 0.005
    dg_concept_weight: float = 0.4
    brainstem_dg_sparsity: float = 0.003
    brainstem_dg_weight: float = 1.5  # non-plastic, bypasses w_max
    # Global workspace arousal — separate from brainstem_working so we can
    # reduce tonic drive independently.  Lower weight lets stimulus-driven
    # input (association/predictive/working) dominate, creating temporal
    # correlation that STDP can learn from.
    brainstem_workspace_sparsity: float = 0.003
    brainstem_workspace_weight: float = 0.5  # weaker than other brainstem (1.5)

    # Meta-controller connections
    meta_input_sparsity: float = 0.01  # broad sampling of cortex
    meta_input_weight: float = 0.3
    meta_output_sparsity: float = 0.005  # diffuse projection
    meta_output_weight: float = 0.2


@dataclass
class DriveConfig:
    """Homeostatic drive thresholds and decay rates."""

    energy_decay: float = 0.001  # per step
    energy_recovery: float = 0.005  # pull toward resting (~0.7), equilibrium ~0.5
    energy_critical: float = 0.2
    damage_recovery: float = 0.0005
    temperature_drift: float = 0.0002
    temperature_optimal: float = 0.5
    temperature_recovery: float = 0.005  # mean-revert toward optimal
    fatigue_rate: float = 0.0003
    fatigue_recovery: float = 0.002  # pull toward resting (~0.2), equilibrium ~0.35
    fatigue_recovery_threshold: float = 0.8
    tonic_current: float = 30.0  # mV — brainstem pacemaker baseline
    perturbation_interval: int = 50  # perturb every N update() calls
    perturbation_std: float = 0.05  # noise amplitude


@dataclass
class EncodingConfig:
    """Spike encoding parameters."""

    rate_gain: float = 100.0  # Hz — max firing rate
    noise_fraction: float = 0.1
    # Sensory sub-range boundaries (fractions of sensory_cortex size)
    visual_end: float = 0.4  # 0–40%
    auditory_end: float = 0.65  # 40–65%
    tactile_end: float = 0.85  # 65–85%
    # remaining 85–100% = proprioceptive
    modality_weights: dict = field(
        default_factory=lambda: {
            "visual": 3.0,  # highest weight (like human ~60% visual)
            "auditory": 2.0,  # text/voice — high weight
            "tactile": 1.0,
            "proprioceptive": 1.0,
            "body_visual": 1.5,  # self-view (MuJoCo body camera)
        }
    )
    inactivity_timeout: int = 200  # steps before a modality is considered inactive


@dataclass
class DecodingConfig:
    """Motor decoding parameters."""

    window_steps: int = 20  # sliding window for rate estimation
    firing_threshold: float = 0.3  # min rate to emit command
    cooldown_steps: int = 10  # min steps between commands per channel
    # Motor sub-range boundaries (fractions of motor_cortex size)
    locomotion_end: float = 0.3
    manipulation_end: float = 0.6
    head_end: float = 0.8
    speech_end: float = 0.8  # default = head_end → no speech sub-range (backward compat)
    # When speech_end > head_end, the range [head_end, speech_end] becomes the
    # "speech" sub-range — brain-native language/phoneme output learned via STDP.
    expression_end: float = (
        1.0  # speech_end–100% = expression (default); <1.0 enables cognitive sub-range
    )
    # When expression_end < 1.0, the range [expression_end, 1.0] becomes the
    # "cognitive" sub-range — non-physical actions (LLM queries, memory ops).

    def __post_init__(self):
        # Validate full motor sub-range ordering
        boundaries = [
            ("locomotion_end", self.locomotion_end),
            ("manipulation_end", self.manipulation_end),
            ("head_end", self.head_end),
            ("speech_end", self.speech_end),
            ("expression_end", self.expression_end),
        ]
        for name, val in boundaries:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
        for i in range(len(boundaries) - 1):
            n1, v1 = boundaries[i]
            n2, v2 = boundaries[i + 1]
            if v1 > v2:
                raise ValueError(
                    f"{n1} ({v1}) must be <= {n2} ({v2}). "
                    f"Required ordering: locomotion_end <= manipulation_end <= "
                    f"head_end <= speech_end <= expression_end <= 1.0"
                )


@dataclass
class InstinctConfig:
    """Orienting instinct parameters — innate attention biases."""

    novelty_boost: float = 3.0  # gain for new/long-silent modality
    novelty_window: int = 500  # steps of silence before considered novel
    change_boost: float = 2.0  # max gain for changed input
    change_threshold: float = 0.3  # normalized diff to trigger change detection
    change_history_len: int = 5  # recent inputs to compare against
    crossmodal_boost: float = 1.5  # gain when 2+ modalities fire together
    crossmodal_min_modalities: int = 2  # min modalities for binding boost
    habituation_rate: float = 0.02  # how fast repeated inputs habituate
    habituation_recovery: float = 0.005  # how fast habituation fades when silent
    habituation_cap: float = 0.8  # max habituation level (0.8 = 80% boost reduction)
    phase_transition_steps: int = 10_000  # smooth interpolation window on phase change
    # Phase-specific gain scales (adolescent amplifies novelty-seeking, mature dampens)
    adolescent_novelty_scale: float = 1.17  # novelty gain multiplier during adolescence
    adolescent_crossmodal_scale: float = 1.33  # crossmodal gain multiplier during adolescence
    mature_novelty_scale: float = 0.5  # novelty gain multiplier during maturity
    mature_crossmodal_scale: float = 0.8  # crossmodal gain multiplier during maturity


@dataclass
class CognitiveActionConfig:
    """Cognitive action channel — lets the brain query external LLMs.

    When prediction error stays high across sustained_window steps,
    the cognitive sub-range of motor cortex can fire to trigger an
    LLM query. The query is gated by cooldown and confidence threshold.
    """

    enabled: bool = False  # opt-in (requires expression_end < 1.0 in DecodingConfig)
    sustained_window: int = 50  # steps of high prediction error before trigger
    error_threshold: float = 0.6  # prediction error above this counts as "confused"
    confidence_threshold: float = 0.3  # min cognitive firing rate to emit query
    cooldown_steps: int = 200  # min steps between cognitive queries
    max_query_length: int = 256  # max tokens in generated query context
    response_injection_gain: float = 2.0  # gain multiplier for LLM response encoding
    # Action types the cognitive channel can request
    action_types: tuple[str, ...] = ("query_llm", "save_memory", "request_guidance")


@dataclass
class MotorFeedbackConfig:
    """Motor outcome feedback — closes the sensorimotor loop.

    When the brain fires a motor command, the consequence (success/failure)
    must feed back as proprioceptive input so STDP can learn which motor
    patterns work. This config controls how outcome signals are injected.

    Feedback sources:
      - IMU/gyro: balance state after locomotion commands
      - Camera: visual change confirming manipulation success
      - Joint encoders: target position reached
      - Force sensors: grip success/failure
      - Simulator: ground truth outcome
      - Human teacher: explicit reward/punishment via dashboard

    The brain also generates an internal DA burst after each motor command
    (echo tracker) so eligibility traces on motor pathways stay active
    during the feedback delay window.
    """

    enabled: bool = False  # opt-in via NEURO_MOTOR_FEEDBACK=1
    # Gain applied to outcome injection (like cognitive response_injection_gain)
    success_gain: float = 2.0  # boost for positive outcomes
    failure_gain: float = 0.5  # dampened injection for negative outcomes
    # Internal DA burst after motor fire — keeps eligibility traces alive
    echo_da_boost: float = 1.5  # DA multiplier during echo window
    echo_window_steps: int = 300  # steps to sustain DA boost after motor fire
    echo_window_ms: int = (
        0  # wall-clock ms for echo window (0=use step-based). For real hardware set ~2000-5000ms.
    )
    # Outcome-contingent DA — success/failure modulates DA so three-factor
    # learning can distinguish good from bad motor commands.
    outcome_da_success: float = 2.0  # DA multiplier on success outcome
    outcome_da_failure: float = 0.3  # DA multiplier on failure outcome
    outcome_da_steps: int = 100  # steps to sustain outcome DA modulation
    # R-STDP on motor pathways — prediction error gates motor learning
    # Separate from `enabled` so feedback can be ON with R-STDP OFF during
    # initial proprioceptive adaptation.  NEURO_MOTOR_RSTDP=1 to enable.
    motor_rstdp_enabled: bool = False
    # Motor babbling — random exploratory current injected into motor cortex
    # during early developmental phases. Like infant kicking, this creates
    # action-outcome pairs for self-supervised learning via three-factor STDP.
    babbling_enabled: bool = False  # opt-in via NEURO_MOTOR_BABBLING=1
    babbling_amplitude: float = (
        2.0  # amplitude of random motor current (low to not drown learned patterns)
    )
    babbling_rate: float = 0.1  # fraction of steps with babbling (0.1 = 10%)
    # Virtual body (MuJoCo) settings
    virtual_delay_ms: int = 75  # simulated feedback latency
    heartbeat_timeout_s: float = 30.0  # actuator considered disconnected after this
    mujoco_steps_per_command: int = 500  # physics steps per motor command (1s at dt=0.002)
    # Continuous physics loop — body steps even without motor commands
    mujoco_continuous: bool = False  # opt-in via NEURO_MUJOCO_CONTINUOUS=1
    mujoco_physics_hz: float = 50.0  # physics ticks per second (each tick = 1 mj_step)
    mujoco_proprio_hz: float = 5.0  # proprioceptive state emission rate (Hz)
    mujoco_viz_hz: float = 10.0  # visualization state publication rate (Hz)
    # Motor command rate limiting — protects physical servos from rapid direction changes
    motor_rate_limit_hz: float = 0.0  # 0=unlimited (MuJoCo). Set 10-20 for physical hardware.
    # Pain / nociceptive signal — brain learns joint limits through experience
    pain_enabled: bool = True  # on by default when motor feedback is enabled
    pain_limit_zone: float = 0.2  # outer 20% of joint range activates pain
    pain_da_penalty: float = 0.3  # DA multiplied by this when max pain = 1.0
    # Continuous height-proportional DA — every step modulates DA based on
    # how upright the body is, providing a gradient signal for motor learning.
    continuous_height_da: bool = True  # on by default when motor feedback enabled
    # Teaching pulse decay — inject_motor_teaching stores a decaying pulse
    # over N steps instead of a single-step current injection.
    teaching_decay_steps: int = 10  # exponential decay tau in brain steps
    # Standing pattern injection — PD-computed ideal torques injected as
    # weak motor cortex current during supported_stand phase.
    standing_pattern_gain: float = 0.3  # gain for PD standing pattern (hint, not override)
    # Motor homeostasis boost — stronger synaptic scaling on motor groups
    # during supported phase to break weight saturation.
    motor_homeostasis_boost: float = 2.0  # scaling strength multiplier during supported phase
    # Task curriculum — structured goals (stand, balance, reach, walk)
    tasks_enabled: bool = False  # opt-in via NEURO_TASKS=1
    # Population vector decoding — per-joint motor control instead of per-channel
    population_vector: bool = False  # opt-in via NEURO_MOTOR_POPULATION_VECTOR=1
    # Body actuator mapping: channel name -> list of actuator names.
    # Default is 29-DOF humanoid. Override for quadruped, arm, drone, etc.
    # The decoder and body read this at runtime, so changing it adapts the
    # entire motor pipeline without touching brain code.
    channel_actuators: dict[str, list[str]] = field(
        default_factory=lambda: {
            "locomotion": [
                "waist_yaw_m",
                "waist_roll_m",
                "waist_pitch_m",
                "r_hip_yaw_m",
                "r_hip_roll_m",
                "r_hip_pitch_m",
                "r_knee_m",
                "r_ankle_pitch_m",
                "r_ankle_roll_m",
                "l_hip_yaw_m",
                "l_hip_roll_m",
                "l_hip_pitch_m",
                "l_knee_m",
                "l_ankle_pitch_m",
                "l_ankle_roll_m",
            ],
            "manipulation": [
                "r_shoulder_pitch_m",
                "r_shoulder_roll_m",
                "r_shoulder_yaw_m",
                "r_elbow_m",
                "r_wrist_pitch_m",
                "r_wrist_yaw_m",
                "l_shoulder_pitch_m",
                "l_shoulder_roll_m",
                "l_shoulder_yaw_m",
                "l_elbow_m",
                "l_wrist_pitch_m",
                "l_wrist_yaw_m",
            ],
            "head": ["neck_pitch_m", "neck_yaw_m"],
        }
    )


@dataclass
class SafetyGateConfig:
    """Kernel safety gate — every motor command passes through the Moral Kernel.

    When enabled, motor proposals are submitted to the Kernel for
    ALLOW/DENY/TRANSFORM/DEFER decisions.  DENY and TRANSFORM decisions are
    fed back as proprioceptive signals so the brain learns safe motor patterns
    through STDP (negative outcomes weaken dangerous firing patterns).

    The safety gate is asynchronous and non-blocking: the simulation loop
    continues immediately, and decisions are processed when they arrive.
    This avoids adding latency to the brain step cycle.
    """

    enabled: bool = False  # opt-in via NEURO_SAFETY_GATE=1
    # Decision timeout — how long to wait for Kernel decision (seconds)
    decision_timeout: float = 2.0
    # Fail-open: if Kernel doesn't respond, allow the command (True) or deny (False)
    fail_open: bool = True
    # Inject negative feedback for DENY decisions
    deny_feedback: bool = True
    # Inject corrected feedback for TRANSFORM decisions
    transform_feedback: bool = True


@dataclass
class InhibitoryConfig:
    """Inhibitory neuron parameters — 80/20 E/I split per region."""

    inhibitory_fraction: float = 0.20  # 20% inhibitory (cortical ratio)
    inhibitory_tau: float = 10.0  # ms — fast-spiking interneurons
    inhibitory_refractory_ms: float = 1.0  # ms — shorter refractory
    inhibitory_threshold: float = -58.0  # mV — slightly lower threshold
    inhibitory_weight_scale: float = -1.0  # negative output (inhibition)


@dataclass
class AstrocyteConfig:
    """Astrocyte-gated plasticity (CIP-22).

    Simulated glial cells that monitor metabolic activity per brain region and
    gate synaptic plasticity based on energy cost. Provides a slow "fourth
    factor" for learning beyond three-factor Hebbian (STDP + eligibility +
    neuromodulation).

    When a region's firing rate is high and weight changes are large, the
    astrocyte's intracellular calcium rises. High calcium triggers
    gliotransmitter release which suppresses plasticity, protecting
    established memories from being overwritten (continual learning).

    Based on AGMP (Frontiers in Neuroscience, 2025): 31.4% on Split CIFAR-100
    vs EWC's 17.25%.
    """

    enabled: bool = False  # NEURO_ASTROCYTES=1
    tau_calcium: float = 5000.0  # ms, calcium integration time constant (very slow)
    tau_gliotransmitter: float = 2000.0  # ms, gliotransmitter release decay
    metabolic_threshold: float = 0.5  # calcium above this activates gating
    plasticity_gate_min: float = 0.1  # minimum plasticity when fully gated
    excitability_gate_min: float = 0.5  # minimum excitability when fully gated
    sigmoid_slope: float = 10.0  # steepness of gating sigmoid


@dataclass
class DualInhibitionConfig:
    """GABA-A/GABA-B dual inhibitory dynamics (CIP-21).

    Splits inhibitory neurons into two biologically-distinct subtypes:
      PV-fast (parvalbumin): GABA-A, tau ~6ms, targets perisomatic compartment.
        Provides spike timing precision, fast WTA, gamma oscillations.
      SST-slow (somatostatin): GABA-B, tau ~150ms, targets apical distal dendrites.
        Sustains inhibition for hundreds of ms, enables CPG locomotion rhythms,
        emergent theta oscillations, and prevents seizure-like runaway.

    When enabled, SST neuron spikes are removed from instant `signed_spikes`
    and instead accumulate a slow GABA-B conductance on post-synaptic targets.
    PV neurons retain instant inhibition (existing behavior).
    """

    enabled: bool = False  # NEURO_DUAL_INHIBITION=1
    pv_fraction: float = 0.6  # 60% PV (fast), 40% SST (slow) of inhibitory pool
    gaba_b_tau: float = 150.0  # ms, GABA-B decay time constant
    gaba_b_rise_tau: float = 30.0  # ms, slow rise time constant
    gaba_b_reversal: float = -90.0  # mV, deeper than GABA-A (-70mV)
    gaba_b_conductance_increment: float = 0.05  # conductance bump per SST spike
    sst_tau: float = 20.0  # ms, SST membrane time constant (slightly slower than PV)
    sst_threshold: float = -55.0  # mV, SST firing threshold
    sst_refractory_ms: float = 1.5  # ms, SST refractory period
    sst_target_compartment: int = 0  # COMP_APICAL_DISTAL (dendritic targeting)


@dataclass
class NMDAConfig:
    """NMDA receptor dynamics for attractor working memory (CIP-24).

    Adds slow excitatory synaptic conductance (tau ~100ms) with voltage-dependent
    magnesium block to working memory recurrent connections.  This creates
    bistable persistent activity states (attractors) that hold motor plans and
    concepts for 500+ ms without continuous input.

    Biology: NMDA receptors are glutamate-gated ion channels blocked by Mg2+ at
    resting potential.  Depolarization relieves the block, creating a positive
    feedback loop that sustains activity.  NR2B-specific NMDA antagonists abolish
    persistent PFC activity in primate working memory tasks.

    The Mg2+ block function: B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))
    NMDA current: I_nmda = g_nmda * B(V) * (E_nmda - V)

    Patent: CIP-24.  Extends Claim 1 (7th mechanism) and Claim 6 (motor plan
    persistence for robotic control).  No conflict with existing 6 claims.
    """

    enabled: bool = False  # NEURO_NMDA=1
    tau_nmda: float = 100.0  # ms, NMDA conductance decay time constant
    mg_concentration: float = 1.0  # mM, extracellular Mg2+ concentration
    mg_block_slope: float = 0.062  # mV^-1, voltage sensitivity of Mg2+ block
    nmda_fraction: float = 0.3  # fraction of synaptic input routed to NMDA (general)
    wm_recurrent_nmda: float = 0.7  # NMDA fraction for WM recurrent connections (higher)
    e_nmda: float = 0.0  # mV, NMDA reversal potential (excitatory, ~0 mV)
    conductance_scale: float = 0.05  # scale factor for NMDA conductance accumulation


@dataclass
class GlobalWorkspaceConfig:
    """Global Neuronal Workspace competition and broadcast (CIP-23).

    Implements Dehaene-Changeux Global Neuronal Workspace Theory.  A dedicated
    workspace region receives projections from higher-order cortical areas
    (association, concept, WM, predictive).  Neural coalitions compete via
    strong lateral inhibition.  When a coalition's firing rate crosses a
    nonlinear ignition threshold, its signal is broadcast to all regions,
    creating a unified "conscious access" moment that arbitrates multi-goal
    motor planning.

    For the physical body: when walking + avoiding obstacle + maintaining balance,
    GWT determines which goal gets priority motor access.  Only coherent,
    well-formed motor plans cross the ignition threshold.

    Patent: CIP-23.  Extends Claim 3 (cognitive architecture) and Claim 4
    (cross-modal binding).  No conflict with existing 6 claims.
    """

    enabled: bool = False  # NEURO_GLOBAL_WORKSPACE=1
    n_neurons: int = 5000  # workspace population size (small = bottleneck)
    ignition_threshold: float = 0.3  # firing rate threshold for ignition
    broadcast_gain: float = 3.0  # multiplicative gain on efferent current during ignition
    competition_inhibition: float = 2.0  # lateral inhibition weight (strong)
    refractory_steps: int = 50  # post-ignition silence (prevents rapid switching)
    sigmoid_slope: float = 10.0  # sharpness of ignition detection
    lateral_sparsity: float = 0.01  # workspace->workspace inhibition sparsity
    afferent_sparsity: float = 0.003  # input->workspace sparsity
    afferent_weight: float = 0.4  # input->workspace initial weight
    efferent_sparsity: float = 0.003  # workspace->target broadcast sparsity
    efferent_weight: float = 0.5  # workspace->target initial weight
    adolescent_plasticity_boost: float = (
        1.0  # multiplier on workspace eligibility during adolescent
    )


@dataclass
class DendriticCompartmentConfig:
    """Per-compartment biophysical parameters for dendritic processing.

    4 compartments model distinct dendritic domains:
      0 = apical distal   (feedforward sensory/feature input)
      1 = basal            (recurrent/lateral connections)
      2 = apical proximal  (top-down feedback from predictive/concept)
      3 = perisomatic      (modulatory — brainstem, meta-controller)

    Each compartment integrates locally, then couples to the soma via
    conductance. Apical and proximal compartments support dendritic
    spikes (NMDA/calcium plateau potentials) for supralinear boosting.
    """

    n_compartments: int = 4

    # Per-compartment membrane time constants (ms)
    # Distal dendrites are slower; perisomatic is fastest
    tau: tuple[float, ...] = (30.0, 20.0, 25.0, 10.0)

    # Per-compartment resting potentials (mV)
    resting: tuple[float, ...] = (-65.0, -65.0, -65.0, -65.0)

    # Conductance coupling from compartment to soma (dimensionless, 0-1)
    soma_coupling: tuple[float, ...] = (0.3, 0.5, 0.4, 0.8)

    # Dendritic spike thresholds (mV above resting). None = no dendritic spike.
    dend_spike_threshold: tuple[float | None, ...] = (8.0, None, 10.0, None)

    # Supralinear boost multiplier when dendritic spike fires
    dend_spike_boost: float = 2.5

    # Dendritic spike refractory period (ms)
    dend_spike_refractory_ms: float = (
        3.0  # reduced from 5ms — shorter refractory increases avg soma current
    )

    # Dendritic voltage floor (mV) — prevents runaway hyperpolarization from
    # inhibitory-dominated input. Biological K+ reversal potential ≈ -90mV.
    dend_voltage_floor: float = -90.0

    # Plateau potential: fraction of threshold offset maintained after dendritic spike.
    # Real dendritic calcium spikes create sustained depolarization (50-100ms).
    # After spike, voltage resets to rest + plateau_fraction * (thresh - rest) instead
    # of rest, and stays there during refractory (leaky integration is disabled).
    # Patent Claim 5: supralinear dendritic spikes with 2.5x amplification.
    dend_spike_plateau_fraction: float = 0.75

    # Whether compartments are enabled (allows opt-out for backward compat)
    enabled: bool = True

    def __post_init__(self):
        nc = self.n_compartments
        for name in ("tau", "resting", "soma_coupling", "dend_spike_threshold"):
            val = getattr(self, name)
            if len(val) != nc:
                raise ValueError(f"{name} length {len(val)} != n_compartments {nc}")


@dataclass
class CompartmentAssignmentConfig:
    """Maps each synapse group to a target compartment on its postsynaptic region.

    Default assignment follows biological dendritic organization:
      0 (apical distal):   feedforward sensory/feature projections
      1 (basal):           recurrent/lateral connections
      2 (apical proximal): top-down feedback (predictive, concept)
      3 (perisomatic):     modulatory (brainstem, meta-controller)
    """

    assignments: dict[str, int] = field(
        default_factory=lambda: {
            # Feedforward (compartment 0 — apical distal)
            "sensory_reflex": COMP_APICAL_DISTAL,
            "sensory_association": COMP_APICAL_DISTAL,
            "sensory_motor": COMP_APICAL_DISTAL,
            "sensory_cerebellum": COMP_APICAL_DISTAL,
            "sensory_feature": COMP_APICAL_DISTAL,
            "feature_association": COMP_APICAL_DISTAL,
            "association_concept": COMP_BASAL,
            "association_predictive": COMP_APICAL_DISTAL,
            "association_working": COMP_APICAL_DISTAL,
            "association_meta": COMP_APICAL_DISTAL,
            "reflex_motor": COMP_APICAL_DISTAL,
            # Recurrent/lateral (compartment 1 — basal)
            "association_lateral": COMP_BASAL,
            "concept_lateral": COMP_BASAL,
            "predictive_recurrent": COMP_BASAL,
            "motor_cerebellum": COMP_BASAL,
            # Top-down feedback (compartment 2 — apical proximal)
            "predictive_association": COMP_APICAL_PROXIMAL,
            "predictive_concept": COMP_APICAL_PROXIMAL,
            "concept_predictive": COMP_APICAL_PROXIMAL,
            "concept_working": COMP_APICAL_PROXIMAL,
            "cerebellum_motor": COMP_APICAL_PROXIMAL,
            "working_motor": COMP_APICAL_PROXIMAL,
            # Modulatory (compartment 3 — perisomatic)
            "brainstem_sensory": COMP_PERISOMATIC,
            "brainstem_motor": COMP_PERISOMATIC,
            "brainstem_association": COMP_PERISOMATIC,
            "brainstem_cerebellum": COMP_PERISOMATIC,
            "brainstem_feature": COMP_PERISOMATIC,
            "brainstem_working": COMP_PERISOMATIC,
            "brainstem_predictive": COMP_PERISOMATIC,
            "brainstem_concept": COMP_PERISOMATIC,
            "meta_association": COMP_PERISOMATIC,
            # Working memory recurrent (CIP-24)
            "working_recurrent": COMP_BASAL,
            # Pattern separator (dentate gyrus)
            "association_dg": COMP_APICAL_DISTAL,
            "dg_concept": COMP_BASAL,
            "brainstem_dg": COMP_PERISOMATIC,
            # Global Workspace (CIP-23)
            "association_workspace": COMP_APICAL_DISTAL,
            "predictive_workspace": COMP_APICAL_DISTAL,
            "working_workspace": COMP_APICAL_DISTAL,
            "concept_workspace": COMP_APICAL_DISTAL,
            "feature_workspace": COMP_APICAL_DISTAL,
            "meta_workspace": COMP_APICAL_DISTAL,
            "workspace_lateral": COMP_BASAL,
            "brainstem_workspace": COMP_PERISOMATIC,
            "workspace_association": COMP_APICAL_PROXIMAL,
            "workspace_predictive": COMP_APICAL_PROXIMAL,
            "workspace_working": COMP_APICAL_PROXIMAL,
            "workspace_motor": COMP_APICAL_PROXIMAL,
            "workspace_concept": COMP_APICAL_PROXIMAL,
            "workspace_feature": COMP_APICAL_PROXIMAL,
        }
    )


@dataclass
class PatternSeparatorConfig:
    """Pattern separator (dentate gyrus) — sparse expansion for episodic memory."""

    n_neurons: int = 0  # default off (backward compat); set via NEURO_DG_N
    k_winners: int = 1000  # ~2% sparsity at 50K neurons
    tau: float = 15.0  # ms — fast (granule cells)
    threshold: float = -52.0  # mV — hard to fire (selectivity)
    # STDP params: faster learning to quickly form orthogonal codes
    a_plus: float = 0.02
    a_minus: float = 0.016
    tau_plus: float = 15.0  # ms — tighter window
    tau_minus: float = 15.0


@dataclass
class OscillatoryConfig:
    """Gamma/theta oscillatory dynamics for attention gating and memory consolidation.

    Gamma (~40 Hz): gates sensory current for attention binding (Patent §22.1).
    Theta (~6 Hz): gates STDP plasticity — encoding during peak, consolidation during trough.
    Phase coherence across modalities triggers cross-modal binding boost.
    """

    enabled: bool = False  # NEURO_OSCILLATIONS=1
    gamma_freq: float = 40.0  # Hz — attention/binding oscillation
    theta_freq: float = 6.0  # Hz — memory encoding/consolidation cycle
    gamma_amplitude: float = 0.15  # modulation depth (±15% sensory gain), must be < 1.0
    theta_amplitude: float = 0.25  # modulation depth (±25% plasticity), must be < 1.0
    phase_coherence_threshold: float = 0.7  # cross-modal binding detection

    def __post_init__(self):
        if self.gamma_amplitude >= 1.0:
            raise ValueError(
                f"gamma_amplitude must be < 1.0 (got {self.gamma_amplitude}), "
                "otherwise gain can reach zero or go negative"
            )
        if self.theta_amplitude >= 1.0:
            raise ValueError(
                f"theta_amplitude must be < 1.0 (got {self.theta_amplitude}), "
                "otherwise plasticity factor can reach zero or go negative"
            )


@dataclass
class ConceptLayerConfig:
    """Concept bottleneck layer — k-WTA sparse coding."""

    n_neurons: int = 10_000  # concept layer size
    k_winners: int = 200  # k-WTA: top-k fire each step (2% sparsity)
    tau: float = 40.0  # ms — slower integration for temporal pooling
    threshold: float = -52.0  # mV — harder to fire (selectivity)
    # Concept layer STDP: slower learning for stable abstractions
    a_plus: float = 0.005
    a_minus: float = 0.004
    tau_plus: float = 30.0  # ms — wider time window
    tau_minus: float = 30.0


@dataclass
class FeatureLayerConfig:
    """Feature integration layer — V4/IT-like intermediate processing."""

    n_neurons: int = 80_000  # between sensory and association
    # Feature layer STDP: faster learning for low-level features
    a_plus: float = 0.015
    a_minus: float = 0.012
    tau_plus: float = 15.0  # ms — tighter time window
    tau_minus: float = 15.0


@dataclass
class MetaControllerConfig:
    """Meta-controller — neuromodulatory hub for plasticity gating."""

    n_neurons: int = 6_000  # total meta-controller neurons
    # Sub-population fractions (must sum to 1.0)
    monitor_frac: float = 0.333  # 2K — reads firing rates, weight changes
    da_frac: float = 0.167  # 1K — dopamine: per-region plasticity multiplier
    ach_frac: float = 0.083  # 500 — acetylcholine: feedforward vs recurrent
    ne_frac: float = 0.083  # 500 — norepinephrine: global gain/arousal
    serotonin_frac: float = 0.083  # 500 — serotonin: consolidation brake
    integrator_frac: float = 0.250  # 1.5K — internal processing
    # Output ranges — multipliers against developmental baselines.
    # Firing rate [0,1] maps linearly to [lo, hi].
    # All ranges have non-zero floors to prevent zeroing out neuromodulators
    # when meta controller sub-populations are quiescent (common early in training).
    da_range: tuple[float, float] = (0.2, 2.5)
    ach_range: tuple[float, float] = (0.3, 2.0)
    ne_range: tuple[float, float] = (0.5, 3.0)
    serotonin_range: tuple[float, float] = (0.1, 1.5)
    # Learning (slower than cortical)
    tau: float = 30.0  # ms
    a_plus: float = 0.003  # 4x smaller than cortical
    a_minus: float = 0.0025


@dataclass
class EligibilityTraceConfig:
    """Eligibility traces for three-factor learning."""

    tau_eligibility: float = 1000.0  # ms — eligibility window (500-2000ms)
    trace_decay: float = 0.999  # per-step exponential decay
    significance_ratio: float = 0.1  # fraction of a_plus used as significance threshold
    untracked_decay_interval: int = 100  # steps between full-array decay sweeps


@dataclass
class PruningConfig:
    """Synaptic pruning during adolescent phase — "sculpt or lose"."""

    prune_interval: int = 10_000  # steps between pruning rounds
    weight_threshold: float = 0.05  # prune synapses below this weight
    max_prune_fraction: float = 0.05  # max 5% removal per round
    min_survival_rounds: int = 3  # rounds surviving → eligible for identity tag


@dataclass
class MyelinationConfig:
    """Myelination — lock in stable, high-weight synapses."""

    stability_window: int = 5000  # steps of stable weight → eligible
    stability_tolerance: float = 0.02  # max weight change to be "stable"
    stability_check_interval: int = 100  # steps between stability tracking updates
    weight_threshold: float = 0.6  # min weight to be myelination-eligible
    plasticity_reduction: float = 0.1  # myelinated learning rate = 10% of normal
    identity_plasticity_reduction: float = 0.01  # identity-tagged = 1% of normal
    identity_stability_multiplier: int = 2  # stability > N × stability_window → identity (alt path)
    target_fraction: float = 0.25  # exit adolescence when 25% myelinated


@dataclass
class NeighborhoodConsolidationConfig:
    """Temporal neighborhood consolidation — synaptic tagging & capture."""

    da_burst_threshold: float = 3.0  # DA level above this triggers consolidation
    rescue_radius_ms: float = 500.0  # rescue eligibility traces within this window
    rescue_strength: float = 0.5  # rescued traces boosted by this factor
    rescue_floor: float = 0.001  # min trace magnitude after rescue (prevents decayed-to-zero loss)


@dataclass
class AdolescentSTDPConfig:
    """Widened STDP windows during adolescent phase (Brzosko et al. 2019).

    Adolescence favors potentiation over depression — DA D1/D5 receptor
    activity widens the LTP window more than LTD (Brzosko et al. 2019).
    """

    tau_plus: float = 30.0  # ms — widened from 20ms (DA D1/D5 effect)
    tau_minus: float = 25.0  # ms — widened from 20ms
    a_plus_scale: float = 1.5  # LTP amplitude boost (0.012 → 0.018)
    a_minus_scale: float = 1.0  # LTD amplitude: no boost (preserve asymmetry favoring potentiation)


@dataclass
class AdolescentEntryConfig:
    """Dynamic, experience-dependent entry criteria for adolescent phase.

    Criteria are tracked independently — each has its own satisfaction window.
    Adolescence is entered when all criteria have been satisfied at least once
    within the rolling `criteria_window` steps (staggered milestones).
    """

    min_concept_patterns: int = 5  # distinct concept patterns needed
    concept_similarity_threshold: float = 0.3  # cosine sim < this = distinct
    sensory_stability_threshold: float = 0.02  # firing rate variance below this
    feature_stdp_decline: float = 0.4  # feature STDP deltas < 40% of peak
    min_steps: int = 1_000_000  # minimum simulation steps before entry
    consecutive_checks: int = 5  # all criteria must hold for N checks
    check_interval: int = 50_000  # steps between entry checks
    min_duration: int = 200_000  # minimum adolescent steps before exit allowed
    max_duration: int = 50_000_000  # max adolescent steps before forced exit
    criteria_window: int = 200_000  # steps within which all criteria must have been met
    sensory_buffer_size: int = 100  # rolling window size for sensory variance
    concept_sample_interval: int = 1000  # steps between concept pattern sampling
    max_stored_patterns: int = 50  # max concept patterns stored in tracker
    peak_decay: float = 0.99999  # per-step decay for feature STDP peak memory


@dataclass
class CriticalPeriodConfig:
    """Developmental schedule for neuromodulatory baselines."""

    # Phase boundaries (in simulation steps)
    infant_end: int = 600_000  # ~10 min at 1kHz
    toddler_end: int = 3_600_000  # ~60 min
    juvenile_end: int = 21_600_000  # ~6 hours

    def __post_init__(self):
        if not (0 < self.infant_end < self.toddler_end < self.juvenile_end):
            raise ValueError(
                f"Critical period boundaries must be strictly increasing: "
                f"infant_end={self.infant_end}, toddler_end={self.toddler_end}, "
                f"juvenile_end={self.juvenile_end}"
            )

    # Infant phase — wide open plasticity
    infant_da: float = 2.0
    infant_ach: float = 2.5
    infant_ne: float = 2.0
    infant_serotonin: float = 0.1

    # Toddler phase — guided exploration
    toddler_da: float = 1.5
    toddler_ach: float = 1.5
    toddler_ne: float = 1.5
    toddler_serotonin: float = 0.4

    # Juvenile phase — refinement
    juvenile_da: float = 1.2
    juvenile_ach: float = 1.2
    juvenile_ne: float = 1.2
    juvenile_serotonin: float = 0.6

    # Adolescent phase — supercharged sponge (peak plasticity)
    adolescent_da: float = 2.5
    adolescent_ach: float = 2.0
    adolescent_ne: float = 2.2
    adolescent_serotonin: float = 0.15

    # Mature phase — stable
    mature_da: float = 1.0
    mature_ach: float = 1.0
    mature_ne: float = 1.0
    mature_serotonin: float = 0.5


@dataclass
class BCMConfig:
    """BCM metaplasticity — per-neuron adaptive modification threshold."""

    theta_tau: float = 10000.0  # steps — EMA window for activity history
    theta_init: float = 0.01  # initial modification threshold
    epsilon: float = 1e-6  # numerical stability


@dataclass
class TrainingAccelerationConfig:
    """Training acceleration parameters — software-only speedups."""

    # Sparse/event-based encoding: only emit frames with significant change
    sparse_encoding_threshold: float = 0.05  # min mean-abs-diff to emit frame
    # Convergence-aware curriculum: auto-advance when learning stabilizes
    convergence_window: int = 50  # steps to track STDP delta moving average
    convergence_threshold: float = 0.0005  # mean |dw| below this → converged
    convergence_patience: int = 20  # consecutive converged checks before advance


@dataclass
class NeuromorphicConfig:
    """Top-level configuration for the neuromorphic cognitive core."""

    # Sub-configs
    populations: PopulationConfig = field(default_factory=PopulationConfig)
    connections: ConnectionConfig = field(default_factory=ConnectionConfig)
    lif: LIFParams = field(default_factory=LIFParams)
    stdp: STDPParams = field(default_factory=STDPParams)
    rstdp: RSTDPParams = field(default_factory=RSTDPParams)
    drives: DriveConfig = field(default_factory=DriveConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    instincts: InstinctConfig = field(default_factory=InstinctConfig)

    # Hierarchical upgrade configs
    inhibitory: InhibitoryConfig = field(default_factory=InhibitoryConfig)
    dual_inhibition: DualInhibitionConfig = field(default_factory=DualInhibitionConfig)
    astrocyte: AstrocyteConfig = field(default_factory=AstrocyteConfig)
    nmda: NMDAConfig = field(default_factory=NMDAConfig)
    global_workspace: GlobalWorkspaceConfig = field(default_factory=GlobalWorkspaceConfig)
    pattern_separator: PatternSeparatorConfig = field(default_factory=PatternSeparatorConfig)
    oscillatory: OscillatoryConfig = field(default_factory=OscillatoryConfig)
    concept_layer: ConceptLayerConfig = field(default_factory=ConceptLayerConfig)
    feature_layer: FeatureLayerConfig = field(default_factory=FeatureLayerConfig)
    meta_controller: MetaControllerConfig = field(default_factory=MetaControllerConfig)
    eligibility: EligibilityTraceConfig = field(default_factory=EligibilityTraceConfig)
    critical_period: CriticalPeriodConfig = field(default_factory=CriticalPeriodConfig)
    bcm: BCMConfig = field(default_factory=BCMConfig)
    training_accel: TrainingAccelerationConfig = field(default_factory=TrainingAccelerationConfig)

    # Dendritic compartments
    dendrites: DendriticCompartmentConfig = field(default_factory=DendriticCompartmentConfig)
    compartment_assignments: CompartmentAssignmentConfig = field(
        default_factory=CompartmentAssignmentConfig
    )

    # Cognitive action channel
    cognitive_action: CognitiveActionConfig = field(default_factory=CognitiveActionConfig)

    # Motor feedback loop
    motor_feedback: MotorFeedbackConfig = field(default_factory=MotorFeedbackConfig)

    # Safety gate — Kernel-gated motor commands
    safety_gate: SafetyGateConfig = field(default_factory=SafetyGateConfig)

    # Auditory short-term memory (echoic memory buffer)
    auditory_stm_window: int = 8  # temporal audio frames to retain
    auditory_stm_decay: float = 0.85  # per-step decay for older frames

    # Adolescent brain phase configs
    pruning: PruningConfig = field(default_factory=PruningConfig)
    myelination: MyelinationConfig = field(default_factory=MyelinationConfig)
    neighborhood: NeighborhoodConsolidationConfig = field(
        default_factory=NeighborhoodConsolidationConfig
    )
    adolescent_stdp: AdolescentSTDPConfig = field(default_factory=AdolescentSTDPConfig)
    adolescent_entry: AdolescentEntryConfig = field(default_factory=AdolescentEntryConfig)

    # Offline sleep consolidation (OFF by default; see neuromorphic.sleep_phase).
    sleep_phase: SleepPhaseConfig = field(default_factory=SleepPhaseConfig)

    # Per-region LIF overrides
    brainstem_lif: LIFParams = field(
        default_factory=lambda: LIFParams(tau=40.0, threshold=-52.0, reset=-65.0, noise_std=1.0)
    )
    reflex_lif: LIFParams = field(
        default_factory=lambda: LIFParams(tau=10.0, threshold=-60.0, noise_std=0.2)
    )
    working_memory_lif: LIFParams = field(
        default_factory=lambda: LIFParams(tau=30.0, threshold=-55.0, noise_std=0.4)
    )

    # Simulation
    dt: float = 1.0  # ms per step
    steps_per_tick: int = 1  # simulation steps per NATS tick

    # Persistence
    save_interval_s: float = 60.0  # save state every N seconds
    sqlite_path: str = "/data/sqlite/neuromorphic.db"

    # Metrics
    metrics_interval_s: float = 10.0

    # NATS
    nats_url: str = "nats://localhost:4222"

    # Homeostatic drive update interval
    drive_update_interval_s: float = 1.0

    # Sensory buffer decay — exponential hold so single observations integrate
    sensory_decay: float = 0.97  # retain 97% per step → ~33 effective steps
    # Synaptic gain — bridges encoding magnitude (~100) to synaptic output (~2.5)
    synaptic_gain: float = 5.0

    # Weight homeostasis
    homeostasis_interval: int = 1000  # steps between homeostasis rounds
    homeostasis_target_frac: float = 0.5  # target mean weight as fraction of w_max
    homeostasis_rate: float = 0.01  # multiplicative scaling rate toward target
    # During adolescent phase, skip the inverse-plasticity reduction so homeostasis
    # can counterbalance the widened STDP windows (1.5x a_plus).  Without this,
    # high plasticity → low homeostasis → runaway weight drift.
    homeostasis_adolescent_bypass: bool = True
    # Emergency ceiling: if any group's mean weight exceeds this fraction of w_max,
    # trigger an immediate homeostasis round at 3x rate for that group.
    homeostasis_emergency_ceiling: float = 0.85

    # STDP update interval — only run STDP every N steps (3-5x speedup)
    # Safe because last_spike_time is recorded every step and STDP tau=20ms >> interval
    stdp_update_interval: int = 10

    # Minimum firing rate to run STDP — skip groups where both pre and post
    # populations fire below this rate (produces negligible weight change).
    # Patent-compliant: mechanism is still active, just short-circuits the
    # zero-result computation. Set to 0.0 to disable skipping.
    stdp_min_firing_rate: float = 0.005  # 0.5% of population

    # Eligibility trace application interval — run neuromod+decay every N steps.
    # Uses compensated decay (decay^N) so the result is mathematically equivalent.
    # Modulator signal is accumulated over the interval.  Set to 1 for every-step
    # application (original behavior).  Higher values trade tiny precision loss
    # for significant speedup when eligibility is the bottleneck.
    elig_apply_interval: int = 3

    def __post_init__(self):
        if self.homeostasis_interval < 1:
            raise ValueError(f"homeostasis_interval must be >= 1, got {self.homeostasis_interval}")
        if self.homeostasis_rate <= 0:
            raise ValueError(f"homeostasis_rate must be > 0, got {self.homeostasis_rate}")
        if not 0.0 <= self.homeostasis_target_frac <= 1.0:
            raise ValueError(
                f"homeostasis_target_frac must be in [0, 1], got {self.homeostasis_target_frac}"
            )
        if self.homeostasis_emergency_ceiling < self.homeostasis_target_frac:
            raise ValueError(
                f"homeostasis_emergency_ceiling ({self.homeostasis_emergency_ceiling}) "
                f"must be >= homeostasis_target_frac ({self.homeostasis_target_frac})"
            )

    @classmethod
    def from_env(cls) -> NeuromorphicConfig:
        """Build config from environment variables with defaults.

        Population sizes: NEURO_<REGION>_N (int)
        Sparsity scale:   NEURO_SPARSITY_SCALE (float, default 1.0)
            Multiplies all connection sparsities. Use < 1.0 when scaling up
            populations to keep RAM manageable (e.g., 0.5 = half as many
            synapses per neuron pair). Biologically valid — real brains have
            sparser long-range connections at larger scales.
        Dendrites:        NEURO_DENDRITES (0/1, default 1)
        """
        pop = PopulationConfig(
            brainstem=_env_int("NEURO_BRAINSTEM_N", 15_000),
            reflex_arc=_env_int("NEURO_REFLEX_N", 10_000),
            sensory_cortex=_env_int("NEURO_SENSORY_N", 200_000),
            motor_cortex=_env_int("NEURO_MOTOR_N", 100_000),
            cerebellum=_env_int("NEURO_CEREBELLUM_N", 100_000),
            association_cortex=_env_int("NEURO_ASSOCIATION_N", 200_000),
            predictive_layer=_env_int("NEURO_PREDICTIVE_N", 100_000),
            working_memory=_env_int("NEURO_WORKING_MEM_N", 25_000),
            feature_layer=_env_int("NEURO_FEATURE_N", 0),
            concept_layer=_env_int("NEURO_CONCEPT_N", 0),
            pattern_separator=_env_int("NEURO_DG_N", 0),
            meta_controller=_env_int("NEURO_META_N", 0),
            global_workspace=_env_int("NEURO_WORKSPACE_N", 0),
        )

        # Global sparsity scaling — multiply all connection sparsities
        sparsity_scale = _env_float("NEURO_SPARSITY_SCALE", 1.0)
        conn = ConnectionConfig(
            brainstem_workspace_weight=_env_float(
                "NEURO_BRAINSTEM_WORKSPACE_WEIGHT",
                0.5,
            ),
        )
        if sparsity_scale != 1.0:
            for field_name in ConnectionConfig.__dataclass_fields__:
                if field_name.endswith("_sparsity"):
                    default_val = getattr(conn, field_name)
                    setattr(conn, field_name, default_val * sparsity_scale)

        # Dendrites on/off
        dendrites_enabled = _env_int("NEURO_DENDRITES", 1) != 0
        dend_cfg = DendriticCompartmentConfig(enabled=dendrites_enabled)

        # Cognitive action channel
        cog_enabled = _env_int("NEURO_COGNITIVE_ENABLED", 0) != 0
        cog_cfg = CognitiveActionConfig(enabled=cog_enabled)

        # Motor feedback loop
        mfb_enabled = _env_int("NEURO_MOTOR_FEEDBACK", 0) != 0
        mfb_rstdp = _env_int("NEURO_MOTOR_RSTDP", 0) != 0
        mfb_cfg = MotorFeedbackConfig(
            enabled=mfb_enabled,
            motor_rstdp_enabled=mfb_rstdp,
            virtual_delay_ms=_env_int("NEURO_VIRTUAL_DELAY_MS", 75),
            heartbeat_timeout_s=_env_float("NEURO_HEARTBEAT_TIMEOUT", 30.0),
            mujoco_steps_per_command=_env_int("NEURO_MUJOCO_STEPS", 500),
            mujoco_continuous=_env_int("NEURO_MUJOCO_CONTINUOUS", 0) != 0,
            mujoco_physics_hz=_env_float("NEURO_MUJOCO_PHYSICS_HZ", 50.0),
            mujoco_proprio_hz=_env_float("NEURO_MUJOCO_PROPRIO_HZ", 5.0),
            mujoco_viz_hz=_env_float("NEURO_MUJOCO_VIZ_HZ", 10.0),
            pain_enabled=_env_int("NEURO_PAIN_ENABLED", 1) != 0,
            pain_limit_zone=_env_float("NEURO_PAIN_ZONE", 0.2),
            pain_da_penalty=_env_float("NEURO_PAIN_DA_PENALTY", 0.3),
            echo_window_ms=_env_int("NEURO_MOTOR_ECHO_MS", 0),
            motor_rate_limit_hz=_env_float("NEURO_MOTOR_RATE_HZ", 0.0),
            tasks_enabled=_env_int("NEURO_TASKS", 0) != 0,
            population_vector=_env_int("NEURO_MOTOR_POPULATION_VECTOR", 0) != 0,
            babbling_enabled=_env_int("NEURO_MOTOR_BABBLING", 0) != 0,
            babbling_amplitude=_env_float("NEURO_BABBLING_AMPLITUDE", 2.0),
            babbling_rate=_env_float("NEURO_BABBLING_RATE", 0.1),
            outcome_da_success=_env_float("NEURO_OUTCOME_DA_SUCCESS", 2.0),
            outcome_da_failure=_env_float("NEURO_OUTCOME_DA_FAILURE", 0.3),
            outcome_da_steps=_env_int("NEURO_OUTCOME_DA_STEPS", 100),
            echo_window_steps=_env_int("NEURO_ECHO_WINDOW_STEPS", 300),
            continuous_height_da=_env_int("NEURO_CONTINUOUS_HEIGHT_DA", 1) != 0,
            teaching_decay_steps=_env_int("NEURO_TEACHING_DECAY_STEPS", 10),
            standing_pattern_gain=_env_float("NEURO_STANDING_PATTERN_GAIN", 0.3),
            motor_homeostasis_boost=_env_float("NEURO_MOTOR_HOMEOSTASIS_BOOST", 2.0),
        )

        # Astrocyte-gated plasticity — CIP-22
        astro_enabled = _env_int("NEURO_ASTROCYTES", 0) != 0
        astro_cfg = AstrocyteConfig(
            enabled=astro_enabled,
            tau_calcium=_env_float("NEURO_ASTRO_TAU_CALCIUM", 5000.0),
            metabolic_threshold=_env_float("NEURO_ASTRO_THRESHOLD", 0.3),
        )

        # Dual inhibition (GABA-A/GABA-B) — CIP-21
        dual_inh_enabled = _env_int("NEURO_DUAL_INHIBITION", 0) != 0
        dual_inh_cfg = DualInhibitionConfig(
            enabled=dual_inh_enabled,
            pv_fraction=_env_float("NEURO_PV_FRACTION", 0.6),
            gaba_b_tau=_env_float("NEURO_GABA_B_TAU", 150.0),
        )

        # NMDA receptor dynamics — CIP-24
        nmda_enabled = _env_int("NEURO_NMDA", 0) != 0
        nmda_cfg = NMDAConfig(
            enabled=nmda_enabled,
            tau_nmda=_env_float("NEURO_NMDA_TAU", 100.0),
            mg_concentration=_env_float("NEURO_NMDA_MG", 1.0),
            nmda_fraction=_env_float("NEURO_NMDA_FRACTION", 0.3),
            wm_recurrent_nmda=_env_float("NEURO_NMDA_WM_FRACTION", 0.7),
        )

        # Global Workspace competition & broadcast — CIP-23
        gw_enabled = _env_int("NEURO_GLOBAL_WORKSPACE", 0) != 0
        gw_cfg = GlobalWorkspaceConfig(
            enabled=gw_enabled,
            n_neurons=_env_int("NEURO_WORKSPACE_NEURONS", 5000),
            ignition_threshold=_env_float("NEURO_WORKSPACE_IGNITION", 0.3),
            broadcast_gain=_env_float("NEURO_WORKSPACE_GAIN", 3.0),
            competition_inhibition=_env_float("NEURO_WORKSPACE_INHIBITION", 2.0),
            refractory_steps=_env_int("NEURO_WORKSPACE_REFRACTORY", 50),
            lateral_sparsity=_env_float("NEURO_GW_LATERAL_SPARSITY", 0.01),
            afferent_sparsity=_env_float("NEURO_GW_AFFERENT_SPARSITY", 0.003),
            adolescent_plasticity_boost=_env_float("NEURO_GW_PLASTICITY_BOOST", 1.0),
        )
        # When workspace is enabled, set population size from config
        if gw_enabled and pop.global_workspace == 0:
            pop = PopulationConfig(
                **{k: v for k, v in pop.__dict__.items() if k != "global_workspace"},
                global_workspace=gw_cfg.n_neurons,
            )

        # Oscillatory dynamics — gamma/theta rhythms for attention gating & memory
        osc_enabled = _env_int("NEURO_OSCILLATIONS", 0) != 0
        osc_cfg = OscillatoryConfig(
            enabled=osc_enabled,
            gamma_freq=_env_float("NEURO_GAMMA_FREQ", 40.0),
            theta_freq=_env_float("NEURO_THETA_FREQ", 6.0),
            gamma_amplitude=_env_float("NEURO_GAMMA_AMP", 0.15),
            theta_amplitude=_env_float("NEURO_THETA_AMP", 0.25),
        )

        # Safety gate — Kernel-gated motor commands
        sg_enabled = _env_int("NEURO_SAFETY_GATE", 0) != 0
        sg_cfg = SafetyGateConfig(
            enabled=sg_enabled,
            decision_timeout=_env_float("NEURO_SAFETY_TIMEOUT", 2.0),
            fail_open=_env_int("NEURO_SAFETY_FAIL_OPEN", 1) != 0,
        )

        # Speech sub-range: when NEURO_SPEECH_END > head_end (0.8), neurons in
        # [head_end, speech_end] learn language output via STDP during training.
        speech_end = _env_float("NEURO_SPEECH_END", 0.8)  # default = head_end → disabled

        # If cognitive channel enabled, set expression_end to 0.85 by default
        # (frees 15% of motor cortex for cognitive sub-range)
        dec = DecodingConfig(
            speech_end=speech_end,
            expression_end=_env_float("NEURO_EXPRESSION_END", 0.85 if cog_enabled else 1.0),
        )

        # Critical period phase boundaries (env-configurable for PoC compression)
        cp_cfg = CriticalPeriodConfig(
            infant_end=_env_int("NEURO_INFANT_END", 600_000),
            toddler_end=_env_int("NEURO_TODDLER_END", 3_600_000),
            juvenile_end=_env_int("NEURO_JUVENILE_END", 21_600_000),
        )

        # Myelination tuning (env-configurable for training speed)
        myel_cfg = MyelinationConfig(
            stability_window=_env_int("NEURO_MYEL_STABILITY_WINDOW", 5000),
            stability_tolerance=_env_float("NEURO_MYEL_STABILITY_TOLERANCE", 0.02),
            weight_threshold=_env_float("NEURO_MYEL_WEIGHT_THRESHOLD", 0.6),
            target_fraction=_env_float("NEURO_MYEL_TARGET_FRACTION", 0.25),
        )

        # Adolescent entry min steps (scales with phase compression)
        adol_entry = AdolescentEntryConfig(
            min_steps=_env_int("NEURO_ADOLESCENT_MIN_STEPS", 1_000_000),
            check_interval=_env_int("NEURO_ADOLESCENT_CHECK_INTERVAL", 50_000),
            min_duration=_env_int("NEURO_ADOLESCENT_MIN_DURATION", 200_000),
            max_duration=_env_int("NEURO_ADOLESCENT_MAX_DURATION", 50_000_000),
        )

        cfg = cls(
            populations=pop,
            connections=conn,
            dendrites=dend_cfg,
            dual_inhibition=dual_inh_cfg,
            astrocyte=astro_cfg,
            nmda=nmda_cfg,
            global_workspace=gw_cfg,
            decoding=dec,
            cognitive_action=cog_cfg,
            motor_feedback=mfb_cfg,
            oscillatory=osc_cfg,
            safety_gate=sg_cfg,
            critical_period=cp_cfg,
            myelination=myel_cfg,
            adolescent_entry=adol_entry,
            dt=_env_float("NEURO_DT", 1.0),
            save_interval_s=_env_float("NEURO_SAVE_INTERVAL", 60.0),
            sqlite_path=os.environ.get("SQLITE_PATH", "/data/sqlite/neuromorphic.db"),
            nats_url=os.environ.get("NATS_URL", "nats://localhost:4222"),
            stdp=STDPParams(
                min_dt=_env_float("NEURO_STDP_MIN_DT", 1.0),
                max_dt=_env_float("NEURO_STDP_MAX_DT", 50.0),
            ),
            stdp_update_interval=_env_int("NEURO_STDP_INTERVAL", 10),
            elig_apply_interval=_env_int("NEURO_ELIG_INTERVAL", 3),
            homeostasis_interval=_env_int("NEURO_HOMEOSTASIS_INTERVAL", 1000),
            homeostasis_target_frac=_env_float("NEURO_HOMEOSTASIS_TARGET", 0.5),
            homeostasis_rate=_env_float("NEURO_HOMEOSTASIS_RATE", 0.01),
            homeostasis_adolescent_bypass=_env_int("NEURO_HOMEOSTASIS_ADOL_BYPASS", 1) != 0,
            homeostasis_emergency_ceiling=_env_float("NEURO_HOMEOSTASIS_CEILING", 0.85),
        )
        return cfg

    def estimate_memory_bytes(self) -> int:
        """Estimate total memory usage in bytes."""
        pop = self.populations
        conn = self.connections
        # Neuron state: 4 arrays × N × 4 bytes (float32)
        neuron_bytes = pop.total * 4 * 4

        # Sparse matrices: per-nnz storage includes CSR arrays, COO cache,
        # eligibility traces (plastic groups), and BCM theta (n_post).
        def sparse_bytes(n_pre: int, n_post: int, sparsity: float, plastic: bool = True) -> int:
            nnz = int(n_pre * n_post * sparsity)
            base = nnz * 12  # CSR: data(4) + indices(4) + indptr overhead(~4)
            coo = nnz * 8  # cached COO: row(4) + col(4)
            elig = nnz * 4 if plastic else 0  # eligibility trace array
            bcm = n_post * 4 if plastic else 0  # BCM theta per post-neuron
            return base + coo + elig + bcm

        matrix_bytes = sum(
            [
                sparse_bytes(pop.sensory_cortex, pop.reflex_arc, conn.sensory_reflex_sparsity),
                sparse_bytes(pop.reflex_arc, pop.motor_cortex, conn.reflex_motor_sparsity),
                sparse_bytes(
                    pop.brainstem,
                    pop.sensory_cortex,
                    conn.brainstem_sensory_sparsity,
                    plastic=False,
                ),
                sparse_bytes(
                    pop.brainstem,
                    pop.association_cortex,
                    conn.brainstem_association_sparsity,
                    plastic=False,
                ),
                sparse_bytes(
                    pop.brainstem, pop.cerebellum, conn.brainstem_cerebellum_sparsity, plastic=False
                ),
                sparse_bytes(
                    pop.sensory_cortex, pop.association_cortex, conn.sensory_association_sparsity
                ),
                sparse_bytes(
                    pop.association_cortex,
                    pop.association_cortex,
                    conn.association_lateral_sparsity,
                ),
                sparse_bytes(pop.sensory_cortex, pop.motor_cortex, conn.sensory_motor_sparsity),
                sparse_bytes(pop.brainstem, pop.motor_cortex, conn.brainstem_motor_sparsity),
                sparse_bytes(pop.sensory_cortex, pop.cerebellum, conn.sensory_cerebellum_sparsity),
                sparse_bytes(pop.motor_cortex, pop.cerebellum, conn.motor_cerebellum_sparsity),
                sparse_bytes(pop.cerebellum, pop.motor_cortex, conn.cerebellum_motor_sparsity),
                sparse_bytes(
                    pop.association_cortex,
                    pop.predictive_layer,
                    conn.association_predictive_sparsity,
                ),
                sparse_bytes(
                    pop.predictive_layer, pop.predictive_layer, conn.predictive_recurrent_sparsity
                ),
                sparse_bytes(
                    pop.predictive_layer,
                    pop.association_cortex,
                    conn.predictive_association_sparsity,
                ),
                sparse_bytes(
                    pop.association_cortex, pop.working_memory, conn.association_working_sparsity
                ),
                sparse_bytes(
                    pop.working_memory, pop.working_memory, conn.working_recurrent_sparsity
                ),
                sparse_bytes(pop.working_memory, pop.motor_cortex, conn.working_motor_sparsity),
                sparse_bytes(
                    pop.brainstem,
                    pop.working_memory,
                    conn.brainstem_working_sparsity,
                    plastic=False,
                ),
                sparse_bytes(
                    pop.brainstem,
                    pop.predictive_layer,
                    conn.brainstem_predictive_sparsity,
                    plastic=False,
                ),
            ]
        )

        # Hierarchical connections (when enabled)
        if pop.feature_layer > 0:
            matrix_bytes += sparse_bytes(
                pop.sensory_cortex, pop.feature_layer, conn.sensory_feature_sparsity
            )
            matrix_bytes += sparse_bytes(
                pop.feature_layer, pop.association_cortex, conn.feature_association_sparsity
            )
            matrix_bytes += sparse_bytes(
                pop.brainstem, pop.feature_layer, conn.brainstem_feature_sparsity, plastic=False
            )
        if pop.concept_layer > 0:
            matrix_bytes += sparse_bytes(
                pop.association_cortex, pop.concept_layer, conn.association_concept_sparsity
            )
            matrix_bytes += sparse_bytes(
                pop.concept_layer, pop.concept_layer, conn.concept_lateral_sparsity
            )
            matrix_bytes += sparse_bytes(
                pop.concept_layer, pop.predictive_layer, conn.concept_predictive_sparsity
            )
            matrix_bytes += sparse_bytes(
                pop.concept_layer, pop.working_memory, conn.concept_working_sparsity
            )
            matrix_bytes += sparse_bytes(
                pop.predictive_layer, pop.concept_layer, conn.predictive_concept_sparsity
            )
            matrix_bytes += sparse_bytes(
                pop.brainstem, pop.concept_layer, conn.brainstem_concept_sparsity, plastic=False
            )
        if pop.pattern_separator > 0:
            matrix_bytes += sparse_bytes(
                pop.association_cortex, pop.pattern_separator, conn.association_dg_sparsity
            )
            if pop.concept_layer > 0:
                matrix_bytes += sparse_bytes(
                    pop.pattern_separator, pop.concept_layer, conn.dg_concept_sparsity
                )
            matrix_bytes += sparse_bytes(
                pop.brainstem, pop.pattern_separator, conn.brainstem_dg_sparsity, plastic=False
            )
        if pop.meta_controller > 0:
            matrix_bytes += sparse_bytes(
                pop.association_cortex, pop.meta_controller, conn.meta_input_sparsity
            )
            matrix_bytes += sparse_bytes(
                pop.meta_controller,
                pop.association_cortex,
                conn.meta_output_sparsity,
                plastic=False,
            )
        if pop.global_workspace > 0:
            gw = self.global_workspace
            ws = pop.global_workspace
            # Afferent: association, predictive, WM -> workspace (always)
            for src in [pop.association_cortex, pop.predictive_layer, pop.working_memory]:
                matrix_bytes += sparse_bytes(src, ws, gw.afferent_sparsity)
            # Efferent: workspace -> association, predictive, WM, motor (always)
            for tgt in [
                pop.association_cortex,
                pop.predictive_layer,
                pop.working_memory,
                pop.motor_cortex,
            ]:
                matrix_bytes += sparse_bytes(ws, tgt, gw.efferent_sparsity)
            # Conditional afferent/efferent
            if pop.concept_layer > 0:
                matrix_bytes += sparse_bytes(pop.concept_layer, ws, gw.afferent_sparsity)
                matrix_bytes += sparse_bytes(ws, pop.concept_layer, gw.efferent_sparsity)
            if pop.feature_layer > 0:
                matrix_bytes += sparse_bytes(pop.feature_layer, ws, gw.afferent_sparsity)
                matrix_bytes += sparse_bytes(ws, pop.feature_layer, gw.efferent_sparsity)
            if pop.meta_controller > 0:
                matrix_bytes += sparse_bytes(pop.meta_controller, ws, gw.afferent_sparsity)
            # Lateral + brainstem
            matrix_bytes += sparse_bytes(ws, ws, gw.lateral_sparsity, plastic=False)
            matrix_bytes += sparse_bytes(
                pop.brainstem, ws, conn.brainstem_working_sparsity, plastic=False
            )

        # Dendritic compartment state: ~52 bytes per cortical neuron with 4 compartments
        dend_bytes = 0
        if self.dendrites.enabled:
            nc = self.dendrites.n_compartments
            # Cortical neurons get dendrites (not brainstem, reflex, meta)
            cortical_n = (
                pop.sensory_cortex
                + pop.motor_cortex
                + pop.cerebellum
                + pop.association_cortex
                + pop.predictive_layer
                + pop.working_memory
                + pop.feature_layer
                + pop.concept_layer
                + pop.pattern_separator
                + pop.global_workspace
            )
            # v_dendrite + dend_refractory + compartment_input: nc * N * 4 bytes each
            # compartment_active_at_spike: nc * N * 1 byte
            dend_bytes = cortical_n * nc * (4 + 4 + 4 + 1)

        # 1.15x overhead for Python objects, SciPy internals, index arrays
        total = int((neuron_bytes + matrix_bytes + dend_bytes) * 1.15)
        return total

    def log_memory_estimate(self) -> None:
        """Log memory estimate and warn if high."""
        est = self.estimate_memory_bytes()
        gb = est / (1024**3)
        logger.info(
            f"Neuromorphic memory estimate: {gb:.2f} GB " f"({self.populations.total:,} neurons)"
        )
        if gb > 6.0:
            logger.warning(
                f"Memory estimate ({gb:.1f} GB) is high — "
                "consider reducing population sizes or sparsity"
            )
