"""Wires regions + connections together, runs the simulation loop."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from neuromorphic.config import (
    NeuromorphicConfig,
    STDPParams,
)
from neuromorphic.cross_modal_probe import CrossModalMetrics, CrossModalProbe
from neuromorphic.decoding import CognitiveDecoder, PredictionDecoder, SpeechDecoder, SpikeDecoder
from neuromorphic.drives import HomeostaticDriveSystem
from neuromorphic.encoding import DynamicSensoryAllocator, SpikeEncoder, _resolve_modality
from neuromorphic.instincts import OrientingInstincts
from neuromorphic.neuromodulation import NeuromodulationSystem
from neuromorphic.oscillations import OscillatorBank
from neuromorphic.reflexes import ReflexManager
from neuromorphic.regions import (
    AssociationCortex,
    BrainRegion,
    Brainstem,
    Cerebellum,
    ConceptLayer,
    FeatureLayer,
    GlobalWorkspace,
    MetaControllerRegion,
    MotorCortex,
    PatternSeparator,
    PredictiveLayer,
    ReflexArc,
    SensoryCortex,
    WorkingMemory,
    create_all_regions,
)
from neuromorphic.synapses import SynapseGroup

logger = logging.getLogger(__name__)


class NeuromorphicNetwork:
    """
    The complete neuromorphic cognitive core.

    Creates up to 12 brain regions and 19+ sparse synapse groups (base 19,
    plus optional feature/concept/pattern_separator/meta groups), and runs the simulation loop:
    inject inputs → propagate spikes → STDP/R-STDP → advance.
    """

    def __init__(self, config: NeuromorphicConfig | None = None, seed: int | None = None):
        self.config = config or NeuromorphicConfig()
        self._rng = np.random.default_rng(seed)
        self._step_count: int = 0
        # Teaching pulse buffer: decaying current injected over multiple steps
        # instead of a single-step injection.  Keys are motor channel names.
        self._teaching_buffer: dict[str, np.ndarray] = {}
        _tau = max(1, self.config.motor_feedback.teaching_decay_steps)
        self._teaching_decay: float = np.exp(-1.0 / _tau)  # per-step decay factor
        # Motor homeostasis boost: service sets this > 1.0 during supported
        # phase to break motor weight saturation.  Reset to 1.0 each step.
        self._motor_scaling_boost: float = 1.0

        # Log memory estimate
        self.config.log_memory_estimate()

        # Create regions (each gets an independent child RNG for reproducibility)
        pop = self.config.populations
        n_regions = (
            8
            + (pop.feature_layer > 0)
            + (pop.concept_layer > 0)
            + (pop.pattern_separator > 0)
            + (pop.meta_controller > 0)
        )
        logger.info(f"Creating {pop.total:,} neurons across {n_regions} regions...")
        self.regions: dict[str, BrainRegion] = create_all_regions(
            self.config, self._rng.spawn(1)[0]
        )

        # Typed accessors — base regions
        self.brainstem: Brainstem = self.regions["brainstem"]
        self.reflex_arc: ReflexArc = self.regions["reflex_arc"]
        self.sensory: SensoryCortex = self.regions["sensory_cortex"]
        self.motor: MotorCortex = self.regions["motor_cortex"]
        self.cerebellum: Cerebellum = self.regions["cerebellum"]
        self.association: AssociationCortex = self.regions["association_cortex"]
        self.predictive: PredictiveLayer = self.regions["predictive_layer"]
        self.working_mem: WorkingMemory = self.regions["working_memory"]

        # Typed accessors — hierarchical regions (optional)
        self.feature: FeatureLayer | None = self.regions.get("feature_layer")
        self.concept: ConceptLayer | None = self.regions.get("concept_layer")
        self.pattern_sep: PatternSeparator | None = self.regions.get("pattern_separator")
        self.meta_ctrl: MetaControllerRegion | None = self.regions.get("meta_controller")
        self.workspace: GlobalWorkspace | None = self.regions.get("global_workspace")
        # All-inhibitory sign vector for workspace lateral competition (CIP-23).
        # GWT requires pure lateral inhibition for coalition selection --
        # using signed_spikes (80/20 E/I) would be net excitatory.
        self._ws_lateral_sign: np.ndarray | None = (
            -np.ones(self.config.populations.global_workspace, dtype=np.float32)
            if self.workspace is not None
            else None
        )

        # Create synapse groups
        logger.info("Creating synapse groups (sparse matrices)...")
        self.synapses: dict[str, SynapseGroup] = self._create_synapses()

        # Subsystems
        self.drives = HomeostaticDriveSystem(self.config, self._rng.spawn(1)[0])
        self.reflexes = ReflexManager()
        self.encoder = SpikeEncoder(self.config, self._rng.spawn(1)[0])
        self.motor_decoder = SpikeDecoder(self.config)
        self.prediction_decoder = PredictionDecoder(self.config)
        self.cognitive_decoder = CognitiveDecoder(self.config)
        self.speech_decoder = SpeechDecoder(self.config)

        # Sensory current buffer — holds input across multiple steps with exponential decay
        # so that a single observation has time to integrate through LIF neurons
        self._sensory_buffer = np.zeros(self.config.populations.sensory_cortex, dtype=np.float32)
        self._sensory_decay = np.float32(self.config.sensory_decay)

        # Synaptic current gain — scales sparse-matrix output to match encoding magnitude
        self._synaptic_gain = np.float32(self.config.synaptic_gain)

        # Dynamic sensory allocation — redistributes sensory cortex based on active modalities
        self.allocator = DynamicSensoryAllocator(self.config)

        # Orienting instincts — innate attention biases (novelty, change, cross-modal)
        self.instincts = OrientingInstincts(self.config)

        # Oscillatory dynamics — gamma/theta rhythm generators (Patent §22.1, §26.2)
        self.oscillators: OscillatorBank | None = None
        if self.config.oscillatory.enabled:
            self.oscillators = OscillatorBank(self.config.oscillatory)
            logger.info(
                f"Oscillatory dynamics enabled: gamma {self.config.oscillatory.gamma_freq:.0f} Hz, "
                f"theta {self.config.oscillatory.theta_freq:.0f} Hz"
            )

        # Astrocyte-gated plasticity (CIP-22) — one astrocyte per region
        self.astrocytes: AstrocyteNetwork | None = None
        if self.config.astrocyte.enabled:
            from neuromorphic.astrocytes import AstrocyteNetwork

            region_names = list(self.regions.keys())
            self.astrocytes = AstrocyteNetwork(self.config.astrocyte, region_names)
            logger.info(f"Astrocyte network enabled: {len(region_names)} astrocytes")

        # Per-region weight change magnitudes for astrocyte metabolic signal
        self._weight_change_by_region: dict[str, float] = {}

        # Cross-modal recall probe — read-only measurement (Patent Claim 4)
        self.cross_modal_probe = CrossModalProbe()
        self._crossmodal_probe_interval = (
            30  # run probe every Nth get_metrics() call (~5min at 10s interval)
        )
        self._crossmodal_probe_counter = 0
        self._crossmodal_last_result: dict[str, Any] = CrossModalMetrics().to_dict()

        # Neuromodulation — 4-channel system with critical period developmental schedule
        self.neuromodulation = NeuromodulationSystem(self.config)

        # Windowed spike count for firing rate reporting
        # (single-step spikes are too brief to catch with 10s metrics polling)
        # Window = 10 steps — at ~2 steps/sec with 187K neurons, covers ~5s
        self._rate_window = 10
        self._spike_counts: dict[str, int] = {name: 0 for name in self.regions}
        self._window_step = 0
        self._firing_rate_report: dict[str, float] = {name: 0.0 for name in self.regions}
        # Throttle the (verbose) mean-weight drift log: emit once every N rate
        # windows. On fast hardware the rate window elapses in tens of ms, which
        # otherwise floods the console. Set NEURO_WEIGHT_LOG_INTERVAL=1 for every
        # window, or 0 to disable the mean-weights log entirely.
        self._weight_log_interval = int(os.environ.get("NEURO_WEIGHT_LOG_INTERVAL", "50"))
        self._weight_log_counter = 0

        # Homeostasis instrumentation counters (Phase 2c)
        self._homeostasis_regular_count = 0
        self._homeostasis_emergency_count = 0

        # G6: Pre-filter plastic synapses (avoid repeated dict iteration)
        self._plastic_synapses: list[tuple[str, SynapseGroup]] = [
            (name, syn) for name, syn in self.synapses.items() if syn.plastic
        ]

        # Patent Claim 1c: BCM metaplasticity — enable on all plastic synapse groups.
        # BCM theta tracks per-postsynaptic-neuron firing rate history and scales
        # STDP (both a_plus and a_minus) so active neurons are harder to potentiate.
        for _name, syn in self._plastic_synapses:
            syn.enable_bcm(self.config.bcm)

        # Strategy 1: Thread pool for parallel STDP — NumPy/SciPy releases GIL
        # during array ops, so threads genuinely parallelize across CPU cores.
        n_stdp_threads = min(
            int(os.environ.get("NEURO_STDP_THREADS", "8")),
            len(self._plastic_synapses),
        )
        self._stdp_executor: ThreadPoolExecutor | None = None
        if n_stdp_threads > 1:
            self._stdp_executor = ThreadPoolExecutor(
                max_workers=n_stdp_threads,
                thread_name_prefix="stdp",
            )
            logger.info(f"Parallel STDP enabled: {n_stdp_threads} threads")

        # Scale-adaptive parallel synaptic routing (docs/adr/0002 §1).
        # Below the crossover (~55K–220K neurons), ThreadPoolExecutor dispatch
        # in _route_parallel costs more than the SpMV it parallelizes; default
        # dev scale (_NEURO_SMALL, ~55K) is ~37% faster with serial routing.
        _route_min = int(os.environ.get("NEURO_PARALLEL_ROUTE_MIN_NEURONS", "100000"))
        _route_mode = os.environ.get("NEURO_PARALLEL_ROUTE", "auto").lower()
        _total_neurons = self.config.populations.total
        if _route_mode == "always":
            self._parallel_routing_enabled = True
        elif _route_mode == "never":
            self._parallel_routing_enabled = False
        else:
            self._parallel_routing_enabled = _total_neurons >= _route_min
        if self._parallel_routing_enabled:
            logger.info(
                f"Parallel synaptic routing enabled "
                f"({_total_neurons:,} neurons, threshold {_route_min:,})"
            )
        else:
            logger.info(
                f"Parallel synaptic routing disabled — serial _route_parallel "
                f"({_total_neurons:,} neurons < {_route_min:,}; see ADR 0002)"
            )

        # Strategy 3: Adaptive per-group STDP interval — stable groups skip more steps
        self._base_stdp_interval = self.config.stdp_update_interval
        # Per-group interval multiplier: starts at 1, doubles when stable, resets on activity
        self._group_stdp_mult: dict[str, int] = {name: 1 for name, _ in self._plastic_synapses}
        self._adaptive_stdp_max_mult = int(os.environ.get("NEURO_ADAPTIVE_STDP_MAX", "4"))
        # Threshold: if last_stdp_delta < this fraction of convergence_threshold, group is "stable"
        self._adaptive_stdp_threshold = self.config.training_accel.convergence_threshold * 0.5

        # Adolescent phase state
        self._adolescent_initialized: bool = False
        self._original_stdp_params: dict[str, STDPParams] = {}  # backup for STDP widening
        self._feature_stdp_peak: float = 0.0  # track peak for entry criteria

        # G4: Sensory rate buffer for adolescent entry (deque for O(1) append + eviction)
        self._sensory_rate_buffer: deque[float] = deque(
            maxlen=self.config.adolescent_entry.sensory_buffer_size
        )

        # G3: Cached myelination fraction (only recompute after pruning/myelination rounds)
        self._cached_myel_fraction: float = 0.0

        # Eligibility interval modulator accumulator (see step() Phase 17)
        self._elig_modulator_accum: float = 0.0

        # Step timing instrumentation (populated each step, EMA-smoothed)
        # Lock protects _step_timing and _step_timing_ema from concurrent
        # reads (get_step_timing on event loop) while step() writes in executor.
        import threading

        self._timing_lock = threading.Lock()
        self._step_timing: dict[str, float] = {}
        self._step_timing_ema: dict[str, float] = {}
        self._timing_ema_alpha: float = 0.1  # smoothing factor

        # Convergence tracking for curriculum auto-advance
        accel = self.config.training_accel
        self._convergence_window = accel.convergence_window
        self._convergence_threshold = accel.convergence_threshold
        self._convergence_patience = accel.convergence_patience
        self._convergence_history: list[float] = []
        self._convergence_stable_count: int = 0

        total_nnz = sum(s.nnz for s in self.synapses.values())
        logger.info(
            f"Network ready: {self.config.populations.total:,} neurons, "
            f"{total_nnz:,} synapses across {len(self.synapses)} groups"
        )

    def _create_synapses(self) -> dict[str, SynapseGroup]:
        """Create synapse groups per the connection map."""
        c = self.config.connections
        pop = self.config.populations
        stdp = self.config.stdp
        rstdp = self.config.rstdp

        # Count total synapse groups for RNG spawning
        n_syn = 20  # base (15 original + 4 brainstem arousal + 1 working_recurrent CIP-24)
        if pop.feature_layer > 0:
            n_syn += 3  # sensory→feature, feature→association, brainstem→feature
        if pop.concept_layer > 0:
            n_syn += 6  # assoc→concept, concept lateral, concept→pred, concept→wm, pred→concept, brainstem→concept
        if pop.pattern_separator > 0:
            n_syn += 2  # assoc→dg, brainstem→dg (always)
            if pop.concept_layer > 0:
                n_syn += 1  # dg→concept (only when concept layer exists)
        if pop.meta_controller > 0:
            n_syn += 2  # meta input, meta output
        if pop.global_workspace > 0:
            # CIP-23: afferent (assoc, pred, wm) + efferent (assoc, pred, wm, motor)
            # + lateral + brainstem = 3 + 4 + 1 + 1 = 9 base
            n_syn += 9
            if pop.concept_layer > 0:
                n_syn += 2  # concept→workspace, workspace→concept
            if pop.feature_layer > 0:
                n_syn += 2  # feature→workspace, workspace→feature
            if pop.meta_controller > 0:
                n_syn += 1  # meta→workspace (no broadcast back to meta)

        syn_rngs = self._rng.spawn(n_syn)
        _ri = iter(syn_rngs)

        synapses: dict[str, SynapseGroup] = {}

        # === Hardwired (NON-plastic) ===

        synapses["sensory_reflex"] = SynapseGroup(
            n_pre=pop.sensory_cortex,
            n_post=pop.reflex_arc,
            sparsity=c.sensory_reflex_sparsity,
            init_weight=c.sensory_reflex_weight,
            plastic=False,
            rng=next(_ri),
            name="sensory→reflex",
        )
        synapses["reflex_motor"] = SynapseGroup(
            n_pre=pop.reflex_arc,
            n_post=pop.motor_cortex,
            sparsity=c.reflex_motor_sparsity,
            init_weight=c.reflex_motor_weight,
            plastic=False,
            rng=next(_ri),
            name="reflex→motor",
        )
        synapses["brainstem_sensory"] = SynapseGroup(
            n_pre=pop.brainstem,
            n_post=pop.sensory_cortex,
            sparsity=c.brainstem_sensory_sparsity,
            init_weight=c.brainstem_sensory_weight,
            plastic=False,
            rng=next(_ri),
            name="brainstem→sensory",
        )
        # Reticular activating system: brainstem arousal to all cortical regions.
        # Non-plastic — provides tonic excitation so feedforward STDP can bootstrap.
        synapses["brainstem_association"] = SynapseGroup(
            n_pre=pop.brainstem,
            n_post=pop.association_cortex,
            sparsity=c.brainstem_association_sparsity,
            init_weight=c.brainstem_association_weight,
            plastic=False,
            rng=next(_ri),
            name="brainstem→association",
        )
        synapses["brainstem_cerebellum"] = SynapseGroup(
            n_pre=pop.brainstem,
            n_post=pop.cerebellum,
            sparsity=c.brainstem_cerebellum_sparsity,
            init_weight=c.brainstem_cerebellum_weight,
            plastic=False,
            rng=next(_ri),
            name="brainstem→cerebellum",
        )
        synapses["brainstem_working"] = SynapseGroup(
            n_pre=pop.brainstem,
            n_post=pop.working_memory,
            sparsity=c.brainstem_working_sparsity,
            init_weight=c.brainstem_working_weight,
            plastic=False,
            rng=next(_ri),
            name="brainstem→working_memory",
        )
        synapses["brainstem_predictive"] = SynapseGroup(
            n_pre=pop.brainstem,
            n_post=pop.predictive_layer,
            sparsity=c.brainstem_predictive_sparsity,
            init_weight=c.brainstem_predictive_weight,
            plastic=False,
            rng=next(_ri),
            name="brainstem→predictive",
        )

        # === Plastic (STDP) — base connections ===
        # Patent Claim 1b: all plastic groups use eligibility traces for three-factor learning
        elig = self.config.eligibility

        synapses["sensory_association"] = SynapseGroup(
            n_pre=pop.sensory_cortex,
            n_post=pop.association_cortex,
            sparsity=c.sensory_association_sparsity,
            init_weight=c.sensory_association_weight,
            plastic=True,
            stdp_params=stdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="sensory→association",
        )
        synapses["association_lateral"] = SynapseGroup(
            n_pre=pop.association_cortex,
            n_post=pop.association_cortex,
            sparsity=c.association_lateral_sparsity,
            init_weight=c.association_lateral_weight,
            plastic=True,
            stdp_params=stdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="association→association",
        )
        # Motor feedback: optionally add R-STDP so prediction error gates motor learning.
        # Both feedback.enabled AND motor_rstdp_enabled must be true — otherwise
        # enabling motor_rstdp_enabled (which defaults True) would change motor
        # learning behavior even when the feedback loop itself is disabled.
        _mfb = self.config.motor_feedback
        motor_rstdp = rstdp if (_mfb.enabled and _mfb.motor_rstdp_enabled) else None
        synapses["sensory_motor"] = SynapseGroup(
            n_pre=pop.sensory_cortex,
            n_post=pop.motor_cortex,
            sparsity=c.sensory_motor_sparsity,
            init_weight=c.sensory_motor_weight,
            plastic=True,
            stdp_params=stdp,
            rstdp_params=motor_rstdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="sensory→motor",
        )
        synapses["brainstem_motor"] = SynapseGroup(
            n_pre=pop.brainstem,
            n_post=pop.motor_cortex,
            sparsity=c.brainstem_motor_sparsity,
            init_weight=c.brainstem_motor_weight,
            plastic=True,
            stdp_params=stdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="brainstem→motor",
        )
        synapses["sensory_cerebellum"] = SynapseGroup(
            n_pre=pop.sensory_cortex,
            n_post=pop.cerebellum,
            sparsity=c.sensory_cerebellum_sparsity,
            init_weight=c.sensory_cerebellum_weight,
            plastic=True,
            stdp_params=stdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="sensory→cerebellum",
        )
        synapses["motor_cerebellum"] = SynapseGroup(
            n_pre=pop.motor_cortex,
            n_post=pop.cerebellum,
            sparsity=c.motor_cerebellum_sparsity,
            init_weight=c.motor_cerebellum_weight,
            plastic=True,
            stdp_params=stdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="motor→cerebellum",
        )
        synapses["cerebellum_motor"] = SynapseGroup(
            n_pre=pop.cerebellum,
            n_post=pop.motor_cortex,
            sparsity=c.cerebellum_motor_sparsity,
            init_weight=c.cerebellum_motor_weight,
            plastic=True,
            stdp_params=stdp,
            rstdp_params=motor_rstdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="cerebellum→motor",
        )
        synapses["association_predictive"] = SynapseGroup(
            n_pre=pop.association_cortex,
            n_post=pop.predictive_layer,
            sparsity=c.association_predictive_sparsity,
            init_weight=c.association_predictive_weight,
            plastic=True,
            stdp_params=stdp,
            rstdp_params=rstdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="association→predictive",
        )
        synapses["predictive_recurrent"] = SynapseGroup(
            n_pre=pop.predictive_layer,
            n_post=pop.predictive_layer,
            sparsity=c.predictive_recurrent_sparsity,
            init_weight=c.predictive_recurrent_weight,
            plastic=True,
            stdp_params=stdp,
            rstdp_params=rstdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="predictive→predictive",
        )
        synapses["predictive_association"] = SynapseGroup(
            n_pre=pop.predictive_layer,
            n_post=pop.association_cortex,
            sparsity=c.predictive_association_sparsity,
            init_weight=c.predictive_association_weight,
            plastic=True,
            stdp_params=stdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="predictive→association",
        )
        synapses["association_working"] = SynapseGroup(
            n_pre=pop.association_cortex,
            n_post=pop.working_memory,
            sparsity=c.association_working_sparsity,
            init_weight=c.association_working_weight,
            plastic=True,
            stdp_params=stdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="association→working_memory",
        )
        # WM self-recurrence — creates attractor states when NMDA enabled (CIP-24)
        synapses["working_recurrent"] = SynapseGroup(
            n_pre=pop.working_memory,
            n_post=pop.working_memory,
            sparsity=c.working_recurrent_sparsity,
            init_weight=c.working_recurrent_weight,
            plastic=True,
            stdp_params=stdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="working_memory→working_memory",
        )
        synapses["working_motor"] = SynapseGroup(
            n_pre=pop.working_memory,
            n_post=pop.motor_cortex,
            sparsity=c.working_motor_sparsity,
            init_weight=c.working_motor_weight,
            plastic=True,
            stdp_params=stdp,
            rstdp_params=motor_rstdp,
            eligibility_config=elig,
            rng=next(_ri),
            name="working_memory→motor",
        )

        # === Hierarchical connections (conditional) ===

        if pop.feature_layer > 0:
            fl = self.config.feature_layer
            feat_stdp = STDPParams(
                a_plus=fl.a_plus,
                a_minus=fl.a_minus,
                tau_plus=fl.tau_plus,
                tau_minus=fl.tau_minus,
                w_min=stdp.w_min,
                w_max=stdp.w_max,
            )
            synapses["sensory_feature"] = SynapseGroup(
                n_pre=pop.sensory_cortex,
                n_post=pop.feature_layer,
                sparsity=c.sensory_feature_sparsity,
                init_weight=c.sensory_feature_weight,
                plastic=True,
                stdp_params=feat_stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="sensory→feature",
            )
            synapses["brainstem_feature"] = SynapseGroup(
                n_pre=pop.brainstem,
                n_post=pop.feature_layer,
                sparsity=c.brainstem_feature_sparsity,
                init_weight=c.brainstem_feature_weight,
                plastic=False,
                rng=next(_ri),
                name="brainstem→feature",
            )
            synapses["feature_association"] = SynapseGroup(
                n_pre=pop.feature_layer,
                n_post=pop.association_cortex,
                sparsity=c.feature_association_sparsity,
                init_weight=c.feature_association_weight,
                plastic=True,
                stdp_params=feat_stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="feature→association",
            )

        if pop.concept_layer > 0:
            cl = self.config.concept_layer
            concept_stdp = STDPParams(
                a_plus=cl.a_plus,
                a_minus=cl.a_minus,
                tau_plus=cl.tau_plus,
                tau_minus=cl.tau_minus,
                w_min=stdp.w_min,
                w_max=stdp.w_max,
            )
            synapses["association_concept"] = SynapseGroup(
                n_pre=pop.association_cortex,
                n_post=pop.concept_layer,
                sparsity=c.association_concept_sparsity,
                init_weight=c.association_concept_weight,
                plastic=True,
                stdp_params=concept_stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="association→concept",
            )
            synapses["concept_lateral"] = SynapseGroup(
                n_pre=pop.concept_layer,
                n_post=pop.concept_layer,
                sparsity=c.concept_lateral_sparsity,
                init_weight=c.concept_lateral_weight,
                plastic=True,
                stdp_params=concept_stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="concept→concept",
            )
            synapses["concept_predictive"] = SynapseGroup(
                n_pre=pop.concept_layer,
                n_post=pop.predictive_layer,
                sparsity=c.concept_predictive_sparsity,
                init_weight=c.concept_predictive_weight,
                plastic=True,
                stdp_params=concept_stdp,
                rstdp_params=rstdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="concept→predictive",
            )
            synapses["concept_working"] = SynapseGroup(
                n_pre=pop.concept_layer,
                n_post=pop.working_memory,
                sparsity=c.concept_working_sparsity,
                init_weight=c.concept_working_weight,
                plastic=True,
                stdp_params=concept_stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="concept→working_memory",
            )
            synapses["predictive_concept"] = SynapseGroup(
                n_pre=pop.predictive_layer,
                n_post=pop.concept_layer,
                sparsity=c.predictive_concept_sparsity,
                init_weight=c.predictive_concept_weight,
                plastic=True,
                stdp_params=concept_stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="predictive→concept",
            )
            # Brainstem arousal → concept layer (non-plastic tonic excitation)
            synapses["brainstem_concept"] = SynapseGroup(
                n_pre=pop.brainstem,
                n_post=pop.concept_layer,
                sparsity=c.brainstem_concept_sparsity,
                init_weight=c.brainstem_concept_weight,
                plastic=False,
                rng=next(_ri),
                name="brainstem→concept",
            )

        if pop.pattern_separator > 0:
            dg = self.config.pattern_separator
            dg_stdp = STDPParams(
                a_plus=dg.a_plus,
                a_minus=dg.a_minus,
                tau_plus=dg.tau_plus,
                tau_minus=dg.tau_minus,
                w_min=stdp.w_min,
                w_max=stdp.w_max,
            )
            # Association → pattern separator (plastic, learns what to separate)
            synapses["association_dg"] = SynapseGroup(
                n_pre=pop.association_cortex,
                n_post=pop.pattern_separator,
                sparsity=c.association_dg_sparsity,
                init_weight=c.association_dg_weight,
                plastic=True,
                stdp_params=dg_stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="association→pattern_separator",
            )
            # Pattern separator → concept layer (plastic, learns orthogonal mapping)
            # Only created when concept layer also enabled — otherwise DG has no downstream target
            if pop.concept_layer > 0:
                synapses["dg_concept"] = SynapseGroup(
                    n_pre=pop.pattern_separator,
                    n_post=pop.concept_layer,
                    sparsity=c.dg_concept_sparsity,
                    init_weight=c.dg_concept_weight,
                    plastic=True,
                    stdp_params=dg_stdp,
                    eligibility_config=elig,
                    rng=next(_ri),
                    name="pattern_separator→concept",
                )
            # Brainstem arousal → pattern separator (non-plastic tonic excitation)
            synapses["brainstem_dg"] = SynapseGroup(
                n_pre=pop.brainstem,
                n_post=pop.pattern_separator,
                sparsity=c.brainstem_dg_sparsity,
                init_weight=c.brainstem_dg_weight,
                plastic=False,
                rng=next(_ri),
                name="brainstem→pattern_separator",
            )

        if pop.meta_controller > 0:
            # Meta-controller reads from association cortex (broad sampling)
            synapses["association_meta"] = SynapseGroup(
                n_pre=pop.association_cortex,
                n_post=pop.meta_controller,
                sparsity=c.meta_input_sparsity,
                init_weight=c.meta_input_weight,
                plastic=True,
                stdp_params=STDPParams(
                    a_plus=self.config.meta_controller.a_plus,
                    a_minus=self.config.meta_controller.a_minus,
                ),
                eligibility_config=elig,
                rng=next(_ri),
                name="association→meta",
            )
            # Meta-controller projects diffusely (read only — effect is through neuromodulation, not synaptic current)
            synapses["meta_association"] = SynapseGroup(
                n_pre=pop.meta_controller,
                n_post=pop.association_cortex,
                sparsity=c.meta_output_sparsity,
                init_weight=c.meta_output_weight,
                plastic=False,
                rng=next(_ri),
                name="meta→association",
            )
            # Meta-controller relies on internal E/I dynamics (InhibitoryConfig)
            # + strong SFA (1.5) for competition.  A signed_spikes lateral group
            # with 80/20 E/I is net excitatory and causes runaway potentiation.

        # === Global Workspace synapse groups (CIP-23, conditional) ===
        if pop.global_workspace > 0:
            gw = self.config.global_workspace
            ws = pop.global_workspace

            # Afferent: higher-order regions → workspace (plastic, STDP)
            synapses["association_workspace"] = SynapseGroup(
                n_pre=pop.association_cortex,
                n_post=ws,
                sparsity=gw.afferent_sparsity,
                init_weight=gw.afferent_weight,
                plastic=True,
                stdp_params=stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="association→workspace",
            )
            synapses["predictive_workspace"] = SynapseGroup(
                n_pre=pop.predictive_layer,
                n_post=ws,
                sparsity=gw.afferent_sparsity,
                init_weight=gw.afferent_weight,
                plastic=True,
                stdp_params=stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="predictive→workspace",
            )
            synapses["working_workspace"] = SynapseGroup(
                n_pre=pop.working_memory,
                n_post=ws,
                sparsity=gw.afferent_sparsity,
                init_weight=gw.afferent_weight,
                plastic=True,
                stdp_params=stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="working_memory→workspace",
            )

            # Efferent: workspace → regions (broadcast, plastic)
            synapses["workspace_association"] = SynapseGroup(
                n_pre=ws,
                n_post=pop.association_cortex,
                sparsity=gw.efferent_sparsity,
                init_weight=gw.efferent_weight,
                plastic=True,
                stdp_params=stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="workspace→association",
            )
            synapses["workspace_predictive"] = SynapseGroup(
                n_pre=ws,
                n_post=pop.predictive_layer,
                sparsity=gw.efferent_sparsity,
                init_weight=gw.efferent_weight,
                plastic=True,
                stdp_params=stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="workspace→predictive",
            )
            synapses["workspace_working"] = SynapseGroup(
                n_pre=ws,
                n_post=pop.working_memory,
                sparsity=gw.efferent_sparsity,
                init_weight=gw.efferent_weight,
                plastic=True,
                stdp_params=stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="workspace→working_memory",
            )
            synapses["workspace_motor"] = SynapseGroup(
                n_pre=ws,
                n_post=pop.motor_cortex,
                sparsity=gw.efferent_sparsity,
                init_weight=gw.efferent_weight,
                plastic=True,
                stdp_params=stdp,
                eligibility_config=elig,
                rng=next(_ri),
                name="workspace→motor",
            )

            # Lateral inhibition (strong, non-plastic — forces competition)
            synapses["workspace_lateral"] = SynapseGroup(
                n_pre=ws,
                n_post=ws,
                sparsity=gw.lateral_sparsity,
                init_weight=gw.competition_inhibition,
                plastic=False,
                rng=next(_ri),
                name="workspace→workspace(lateral)",
            )

            # Brainstem arousal → workspace (non-plastic tonic excitation)
            # Lower weight than other brainstem groups so stimulus-driven input
            # dominates and STDP can learn from temporal correlations.
            synapses["brainstem_workspace"] = SynapseGroup(
                n_pre=pop.brainstem,
                n_post=ws,
                sparsity=c.brainstem_workspace_sparsity,
                init_weight=c.brainstem_workspace_weight,
                plastic=False,
                rng=next(_ri),
                name="brainstem→workspace",
            )

            # Conditional afferent/efferent for optional regions
            if pop.concept_layer > 0:
                synapses["concept_workspace"] = SynapseGroup(
                    n_pre=pop.concept_layer,
                    n_post=ws,
                    sparsity=gw.afferent_sparsity,
                    init_weight=gw.afferent_weight,
                    plastic=True,
                    stdp_params=stdp,
                    eligibility_config=elig,
                    rng=next(_ri),
                    name="concept→workspace",
                )
                synapses["workspace_concept"] = SynapseGroup(
                    n_pre=ws,
                    n_post=pop.concept_layer,
                    sparsity=gw.efferent_sparsity,
                    init_weight=gw.efferent_weight,
                    plastic=True,
                    stdp_params=stdp,
                    eligibility_config=elig,
                    rng=next(_ri),
                    name="workspace→concept",
                )
            if pop.feature_layer > 0:
                synapses["feature_workspace"] = SynapseGroup(
                    n_pre=pop.feature_layer,
                    n_post=ws,
                    sparsity=gw.afferent_sparsity,
                    init_weight=gw.afferent_weight,
                    plastic=True,
                    stdp_params=stdp,
                    eligibility_config=elig,
                    rng=next(_ri),
                    name="feature→workspace",
                )
                synapses["workspace_feature"] = SynapseGroup(
                    n_pre=ws,
                    n_post=pop.feature_layer,
                    sparsity=gw.efferent_sparsity,
                    init_weight=gw.efferent_weight,
                    plastic=True,
                    stdp_params=stdp,
                    eligibility_config=elig,
                    rng=next(_ri),
                    name="workspace→feature",
                )
            if pop.meta_controller > 0:
                synapses["meta_workspace"] = SynapseGroup(
                    n_pre=pop.meta_controller,
                    n_post=ws,
                    sparsity=gw.afferent_sparsity,
                    init_weight=gw.afferent_weight,
                    plastic=True,
                    stdp_params=stdp,
                    eligibility_config=elig,
                    rng=next(_ri),
                    name="meta→workspace",
                )

        # Assign target compartments from config
        assignments = self.config.compartment_assignments.assignments
        for name, syn in synapses.items():
            if name in assignments:
                syn.target_compartment = assignments[name]

        return synapses

    def _route_current(
        self,
        synapse_name: str,
        pre_spikes: np.ndarray,
        pre_sign: np.ndarray | None,
        target_region: BrainRegion,
    ) -> np.ndarray:
        """Compute synapse current and route to the correct dendritic compartment.

        If the synapse has a target_compartment and the region has dendrites,
        injects into the dendritic compartment. Otherwise, injects as direct
        somatic current (backward compatible / subcortical regions).

        Returns the computed current for any additional use.
        """
        syn = self.synapses[synapse_name]
        current = syn.compute_current(pre_spikes, pre_sign) * self._synaptic_gain
        comp = syn.target_compartment
        if comp is not None and target_region.population.v_dendrite is not None:
            target_region.inject_compartment_current(comp, current)
        else:
            target_region.inject_current(current)
        return current

    def _compute_current(
        self,
        synapse_name: str,
        pre_spikes: np.ndarray,
        pre_sign: np.ndarray | None,
    ) -> np.ndarray:
        """Compute synapse current WITHOUT injecting (thread-safe pure function).

        Used by _route_parallel for parallel SpMV computation.
        """
        syn = self.synapses[synapse_name]
        return syn.compute_current(pre_spikes, pre_sign) * self._synaptic_gain

    def _inject_current(
        self,
        synapse_name: str,
        current: np.ndarray,
        target_region: BrainRegion,
    ) -> None:
        """Inject pre-computed current into the target region (NOT thread-safe)."""
        syn = self.synapses[synapse_name]
        comp = syn.target_compartment
        if comp is not None and target_region.population.v_dendrite is not None:
            target_region.inject_compartment_current(comp, current)
        else:
            target_region.inject_current(current)

    def _route_gaba_b(
        self,
        synapse_name: str,
        source_region: BrainRegion,
        target_region: BrainRegion,
    ) -> None:
        """Route SST (slow inhibitory) spikes through lateral synapses into GABA-B.

        When dual inhibition (CIP-21) is enabled, SST neuron spikes are excluded
        from the instant `signed_spikes` used by normal routing. Instead, their
        spikes are routed through the same lateral weight matrix, and the resulting
        current is accumulated into the target population's slow GABA-B conductance.

        No-op when dual inhibition is disabled or when no SST neurons fired.
        """
        if not self.config.dual_inhibition.enabled:
            return
        sst_sp = source_region.population.sst_spikes
        if not sst_sp.any():
            return
        syn = self.synapses[synapse_name]
        # Route SST spikes as excitatory through the weight matrix (absolute current)
        # The sign is handled by accumulate_gaba_b which takes abs()
        sst_current = syn.compute_current(sst_sp.astype(np.float32), None) * self._synaptic_gain
        target_region.population.accumulate_gaba_b(sst_current)

    def _route_nmda(
        self,
        synapse_name: str,
        source_region: BrainRegion,
        target_region: BrainRegion,
    ) -> None:
        """Route excitatory spikes through synapse into slow NMDA conductance.

        When NMDA (CIP-24) is enabled, a fraction of the recurrent excitatory
        current is accumulated into the target population's slow NMDA conductance
        (tau ~100ms). The voltage-dependent Mg2+ block in neurons.step() then
        creates bistable attractor states.

        No-op when NMDA is disabled or when no spikes fired.
        """
        if not self.config.nmda.enabled:
            return
        if target_region.population._nmda_conductance is None:
            return
        pop = source_region.population
        # Only excitatory (glutamatergic) neurons produce NMDA current.
        # Inhibitory (GABAergic) neurons don't have NMDA receptors.
        exc_spikes = pop.spikes & ~pop.is_inhibitory if pop.is_inhibitory.any() else pop.spikes
        if not exc_spikes.any():
            return
        syn = self.synapses[synapse_name]
        current = syn.compute_current(exc_spikes.astype(np.float32), None) * self._synaptic_gain
        # Apply NMDA fraction -- only a portion of excitatory current is NMDA-mediated
        nmda_frac = (
            self.config.nmda.wm_recurrent_nmda
            if "recurrent" in synapse_name
            else self.config.nmda.nmda_fraction
        )
        current *= np.float32(nmda_frac)
        target_region.population.accumulate_nmda(current)

    def _route_parallel(
        self,
        routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]],
    ) -> None:
        """Compute multiple synapse currents in parallel, then inject sequentially.

        Each tuple is (synapse_name, pre_spikes, pre_sign, target_region).
        SciPy CSR SpMV releases the GIL, so threads give genuine parallelism.
        Injection uses += on shared arrays, so it must be sequential.

        Falls back to serial execution when no thread pool is available or
        when the network is below NEURO_PARALLEL_ROUTE_MIN_NEURONS (ADR 0002).
        """
        if len(routes) <= 1 or self._stdp_executor is None or not self._parallel_routing_enabled:
            # Serial fallback
            for syn_name, pre_sp, pre_sign, target in routes:
                self._route_current(syn_name, pre_sp, pre_sign, target)
            return

        # Phase 1: Compute all currents in parallel (pure, thread-safe)
        futures = [
            self._stdp_executor.submit(self._compute_current, syn_name, pre_sp, pre_sign)
            for syn_name, pre_sp, pre_sign, _target in routes
        ]
        currents = [f.result() for f in futures]

        # Phase 2: Inject sequentially (writes to shared compartment arrays)
        for (syn_name, _pre_sp, _pre_sign, target), current in zip(routes, currents):
            self._inject_current(syn_name, current, target)

    def step(
        self,
        sensory_current: np.ndarray | None = None,
        da_multiplier: float = 1.0,
        sensory_gap: int = 0,
    ) -> dict[str, Any]:
        """
        Run one simulation step.

        Args:
            sensory_current: Optional external current for sensory cortex.
            da_multiplier: External DA scaling applied AFTER neuromodulation.update()
                but BEFORE plasticity_multiplier is read. Used for outcome-contingent
                DA (success/failure), motor echo DA boost, and pain DA penalty.
                1.0 = no change, >1.0 = DA burst (potentiation), <1.0 = DA dip.
            sensory_gap: Steps since last sensory observation. When >200,
                homeostatic scaling is frozen to prevent weight collapse during
                sensory starvation. Default 0 (no starvation).

        Returns:
            Dict with motor_commands, reflex_responses, prediction_error, step_count.
        """
        dt = self.config.dt
        s = self.synapses
        _zero = np.float32(0.0)
        _t0 = time.perf_counter()
        # Pre-populate all keys so partial dicts are never exposed on exception
        _timings: dict[str, float] = {
            "1_brainstem_sensory": 0.0,
            "2_reflex_feature": 0.0,
            "3_association": 0.0,
            "4_concept_predictive": 0.0,
            "5_meta_wm_cereb_motor": 0.0,
            "6_decode": 0.0,
            "7_neuromod_phase": 0.0,
            "8_stdp": 0.0,
            "9_eligibility": 0.0,
            "10_consolidation_homeostasis": 0.0,
            "total": 0.0,
        }

        # 1. Inject drive current into brainstem (no dendrites — subcortical)
        drive_current = self.drives.compute_brainstem_current(self.brainstem)
        self.brainstem.step(drive_current, dt)

        # 2. Arousal + sensory buffer → sensory cortex
        # brainstem_sensory routes to perisomatic compartment (via _route_current)
        self._route_current("brainstem_sensory", self.brainstem.spikes, None, self.sensory)
        if sensory_current is not None:
            self._sensory_buffer = sensory_current.copy()
        # Oscillatory gamma gating: modulate sensory current with gamma rhythm (Patent §22.1)
        if self.oscillators is not None:
            self.oscillators.step(dt)
            gamma_g = np.float32(self.oscillators.gamma_gain() * self.oscillators.binding_boost())
            self._sensory_buffer *= gamma_g
        # Safety cap: instincts (up to ~9x) × oscillatory (up to ~1.7x) can compound.
        # Cap at 5× encoding rate_gain to prevent numerical instability while
        # preserving strong novelty/binding signals.
        _max_sensory = np.float32(5.0 * self.config.encoding.rate_gain)
        np.clip(self._sensory_buffer, 0.0, _max_sensory, out=self._sensory_buffer)
        # Sensory buffer is direct somatic injection (external input, not synaptic)
        self.sensory.inject_current(self._sensory_buffer)
        self._sensory_buffer *= self._sensory_decay
        # Step sensory (dendrites integrate compartment inputs, then soma fires)
        self.sensory.step(np.zeros(self.sensory.n, dtype=np.float32), dt)
        _t1 = time.perf_counter()
        _timings["1_brainstem_sensory"] = _t1 - _t0

        # Helper: get signed spikes (E/I) for a region.
        # ONLY used for lateral/recurrent connections WITHIN the same region.
        # Inter-region projections use unsigned spikes (None) — in biology,
        # only excitatory pyramidal cells project between cortical areas.
        # Inhibitory interneurons act locally within their home region.
        def signed(region_name: str) -> np.ndarray | None:
            r = self.regions.get(region_name)
            if r is None:
                return None
            return r.population.signed_spikes

        # 3-5. Reflex + Feature layer — inputs are independent, compute in parallel
        _reflex_feature_routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]] = [
            ("sensory_reflex", self.sensory.spikes, None, self.reflex_arc),
        ]
        if self.feature is not None:
            _reflex_feature_routes.append(
                ("sensory_feature", self.sensory.spikes, None, self.feature)
            )
            if "brainstem_feature" in s:
                _reflex_feature_routes.append(
                    ("brainstem_feature", self.brainstem.spikes, None, self.feature)
                )
        self._route_parallel(_reflex_feature_routes)

        self.reflex_arc.step(np.zeros(self.reflex_arc.n, dtype=np.float32), dt)
        reflex_responses = self.reflexes.check(self.reflex_arc)
        if self.feature is not None:
            self.feature.step(np.zeros(self.feature.n, dtype=np.float32), dt)
        _t2 = time.perf_counter()
        _timings["2_reflex_feature"] = _t2 - _t1

        # 6. Association cortex — multi-modal binding (routes to compartments)
        # Parallel SpMV: all inputs to association are independent.
        # SciPy CSR releases GIL → genuine thread parallelism.
        _assoc_routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]] = [
            ("brainstem_association", self.brainstem.spikes, None, self.association),
            ("sensory_association", self.sensory.spikes, None, self.association),
            (
                "association_lateral",
                self.association.spikes,
                signed("association_cortex"),
                self.association,
            ),
            ("predictive_association", self.predictive.spikes, None, self.association),
        ]
        if self.feature is not None:
            _assoc_routes.append(
                ("feature_association", self.feature.spikes, None, self.association)
            )
        if self.meta_ctrl is not None and "meta_association" in s:
            _assoc_routes.append(
                ("meta_association", self.meta_ctrl.spikes, None, self.association)
            )
        self._route_parallel(_assoc_routes)
        # GABA-B: route SST slow inhibition for association lateral (CIP-21)
        self._route_gaba_b("association_lateral", self.association, self.association)
        self.association.step(np.zeros(self.association.n, dtype=np.float32), dt)
        _t3 = time.perf_counter()
        _timings["3_association"] = _t3 - _t2

        # 6b. Pattern separator (dentate gyrus) — sparse expansion between association and concept
        if self.pattern_sep is not None:
            _dg_routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]] = [
                ("association_dg", self.association.spikes, None, self.pattern_sep),
            ]
            if "brainstem_dg" in s:
                _dg_routes.append(("brainstem_dg", self.brainstem.spikes, None, self.pattern_sep))
            self._route_parallel(_dg_routes)
            self.pattern_sep.step(np.zeros(self.pattern_sep.n, dtype=np.float32), dt)

        # 7. Concept layer (if enabled) — parallel inputs
        if self.concept is not None:
            _concept_routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]] = [
                ("association_concept", self.association.spikes, None, self.concept),
                ("concept_lateral", self.concept.spikes, signed("concept_layer"), self.concept),
            ]
            if "predictive_concept" in s:
                _concept_routes.append(
                    ("predictive_concept", self.predictive.spikes, None, self.concept)
                )
            # Pattern separator output feeds into concept layer (if both enabled)
            if self.pattern_sep is not None and "dg_concept" in s:
                _concept_routes.append(("dg_concept", self.pattern_sep.spikes, None, self.concept))
            if "brainstem_concept" in s:
                _concept_routes.append(
                    ("brainstem_concept", self.brainstem.spikes, None, self.concept)
                )
            self._route_parallel(_concept_routes)
            # GABA-B: route SST slow inhibition for concept lateral (CIP-21)
            self._route_gaba_b("concept_lateral", self.concept, self.concept)
            self.concept.step(np.zeros(self.concept.n, dtype=np.float32), dt)

        # 8. Predictive layer — parallel inputs
        _pred_routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]] = [
            ("association_predictive", self.association.spikes, None, self.predictive),
            (
                "predictive_recurrent",
                self.predictive.spikes,
                signed("predictive_layer"),
                self.predictive,
            ),
        ]
        if self.concept is not None:
            _pred_routes.append(("concept_predictive", self.concept.spikes, None, self.predictive))
        if "brainstem_predictive" in s:
            _pred_routes.append(
                ("brainstem_predictive", self.brainstem.spikes, None, self.predictive)
            )
        self._route_parallel(_pred_routes)
        # GABA-B: route SST slow inhibition for predictive recurrent (CIP-21)
        self._route_gaba_b("predictive_recurrent", self.predictive, self.predictive)
        self.predictive.step(np.zeros(self.predictive.n, dtype=np.float32), dt)
        _t4 = time.perf_counter()
        _timings["4_concept_predictive"] = _t4 - _t3

        # 9. Prediction error → modulation
        prediction_error = self.prediction_decoder.compute_prediction_error(self.predictive)
        modulation = self.prediction_decoder.compute_modulation(prediction_error)

        # 10-12. Meta + Working Memory + Cerebellum — all three are INDEPENDENT.
        # Meta needs: association.  WM needs: association, concept.
        # Cerebellum needs: brainstem, sensory, motor (prev step spikes).
        # None depend on each other → compute ALL inputs in parallel, then step
        # all three regions in parallel (each writes only to its own arrays).

        # Gather all input routes for the three independent regions
        _parallel_routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]] = []
        if self.meta_ctrl is not None:
            _parallel_routes.append(
                ("association_meta", self.association.spikes, None, self.meta_ctrl)
            )
        _parallel_routes.extend(
            [
                ("association_working", self.association.spikes, None, self.working_mem),
                ("brainstem_working", self.brainstem.spikes, None, self.working_mem),
                (
                    "working_recurrent",
                    self.working_mem.spikes,
                    signed("working_memory"),
                    self.working_mem,
                ),
                ("brainstem_cerebellum", self.brainstem.spikes, None, self.cerebellum),
                ("sensory_cerebellum", self.sensory.spikes, None, self.cerebellum),
                ("motor_cerebellum", self.motor.spikes, None, self.cerebellum),
            ]
        )
        if self.concept is not None:
            _parallel_routes.append(
                ("concept_working", self.concept.spikes, None, self.working_mem)
            )
        self._route_parallel(_parallel_routes)

        # NMDA: accumulate slow excitatory conductance from WM recurrence (CIP-24)
        self._route_nmda("working_recurrent", self.working_mem, self.working_mem)
        # GABA-B: route SST slow inhibition for WM recurrent (CIP-21)
        self._route_gaba_b("working_recurrent", self.working_mem, self.working_mem)

        # Step all three regions — independent, can run in parallel via executor
        if self._stdp_executor is not None:
            _region_futures = []
            if self.meta_ctrl is not None:
                _region_futures.append(
                    self._stdp_executor.submit(
                        self.meta_ctrl.step, np.zeros(self.meta_ctrl.n, dtype=np.float32), dt
                    )
                )
            _region_futures.append(
                self._stdp_executor.submit(
                    self.working_mem.step, np.zeros(self.working_mem.n, dtype=np.float32), dt
                )
            )
            _region_futures.append(
                self._stdp_executor.submit(
                    self.cerebellum.step, np.zeros(self.cerebellum.n, dtype=np.float32), dt
                )
            )
            for f in _region_futures:
                f.result()
        else:
            if self.meta_ctrl is not None:
                self.meta_ctrl.step(np.zeros(self.meta_ctrl.n, dtype=np.float32), dt)
            self.working_mem.step(np.zeros(self.working_mem.n, dtype=np.float32), dt)
            self.cerebellum.step(np.zeros(self.cerebellum.n, dtype=np.float32), dt)

        # 12b. Global Workspace — receives from all higher-order regions,
        # competes via lateral inhibition, broadcasts on ignition (CIP-23)
        if self.workspace is not None:
            _ws_routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]] = [
                ("association_workspace", self.association.spikes, None, self.workspace),
                ("predictive_workspace", self.predictive.spikes, None, self.workspace),
                ("working_workspace", self.working_mem.spikes, None, self.workspace),
                ("workspace_lateral", self.workspace.spikes, self._ws_lateral_sign, self.workspace),
                ("brainstem_workspace", self.brainstem.spikes, None, self.workspace),
            ]
            if self.concept is not None and "concept_workspace" in s:
                _ws_routes.append(("concept_workspace", self.concept.spikes, None, self.workspace))
            if self.feature is not None and "feature_workspace" in s:
                _ws_routes.append(("feature_workspace", self.feature.spikes, None, self.workspace))
            if self.meta_ctrl is not None and "meta_workspace" in s:
                _ws_routes.append(("meta_workspace", self.meta_ctrl.spikes, None, self.workspace))
            self._route_parallel(_ws_routes)
            self._route_gaba_b("workspace_lateral", self.workspace, self.workspace)
            self.workspace.step(np.zeros(self.workspace.n, dtype=np.float32), dt)

            # Broadcast: route workspace output to target regions.
            # During ignition, the workspace fires strongly and the efferent
            # synapses naturally carry the amplified signal. The broadcast_gain
            # multiplicatively scales the current for even stronger effect.
            if self.workspace.spikes.any():
                _broadcast_gain = self.workspace.get_broadcast_gain()
                _ws_broadcast: list[tuple[str, BrainRegion]] = [
                    ("workspace_association", self.association),
                    ("workspace_predictive", self.predictive),
                    ("workspace_working", self.working_mem),
                    ("workspace_motor", self.motor),
                ]
                if self.concept is not None and "workspace_concept" in s:
                    _ws_broadcast.append(("workspace_concept", self.concept))
                if self.feature is not None and "workspace_feature" in s:
                    _ws_broadcast.append(("workspace_feature", self.feature))
                for syn_name, target in _ws_broadcast:
                    syn = s[syn_name]
                    current = (
                        syn.compute_current(self.workspace.spikes.astype(np.float32), None)
                        * self._synaptic_gain
                        * np.float32(_broadcast_gain)
                    )
                    self._inject_current(syn_name, current, target)

        # 13. Motor cortex — depends on cerebellum + WM (must wait for above)
        _motor_routes: list[tuple[str, np.ndarray, np.ndarray | None, BrainRegion]] = [
            ("reflex_motor", self.reflex_arc.spikes, None, self.motor),
            ("sensory_motor", self.sensory.spikes, None, self.motor),
            ("brainstem_motor", self.brainstem.spikes, None, self.motor),
            ("cerebellum_motor", self.cerebellum.spikes, None, self.motor),
            ("working_motor", self.working_mem.spikes, None, self.motor),
        ]
        self._route_parallel(_motor_routes)
        # Apply decaying teaching pulse buffer -- persists across multiple steps
        # so the teaching signal survives the brain's slow stepping rate.
        expired = []
        for ch, buf in self._teaching_buffer.items():
            self.motor.inject_current_subrange(ch, buf)
            buf *= self._teaching_decay  # exponential decay
            if np.abs(buf).max() < 0.01:
                expired.append(ch)
        for ch in expired:
            del self._teaching_buffer[ch]
        # Motor babbling: random exploratory current into motor cortex.
        # Like infant kicking -- creates action-outcome pairs for learning.
        # Active during pre-mature phases when babbling is enabled.
        mfb = self.config.motor_feedback
        if mfb.babbling_enabled and self.neuromodulation.phase != "mature":
            if self._rng.random() < mfb.babbling_rate:
                babble = self._rng.standard_normal(self.motor.n).astype(np.float32)
                babble *= np.float32(mfb.babbling_amplitude)
                self.motor.inject_current(babble)
        self.motor.step(np.zeros(self.motor.n, dtype=np.float32), dt)
        _t5 = time.perf_counter()
        _timings["5_meta_wm_cereb_motor"] = _t5 - _t4

        # 14. Decode motor output
        motor_commands = self.motor_decoder.step(self.motor)

        # 14b. Decode cognitive actions (if enabled)
        cognitive_commands = self.cognitive_decoder.step(self.motor, prediction_error)

        # 14c. Decode speech output (if enabled)
        speech_commands = self.speech_decoder.step(self.motor)
        _t6 = time.perf_counter()
        _timings["6_decode"] = _t6 - _t5

        # 15. Update neuromodulation (critical period baselines + meta-controller)
        self._step_count += 1
        meta_outputs = None
        if self.meta_ctrl is not None:
            meta_outputs = self.meta_ctrl.read_neuromodulators()

        # Feed external signals for adolescent entry/exit criteria
        self._update_adolescent_signals()
        self.neuromodulation.update(self._step_count, meta_outputs)

        # Apply external DA multiplier AFTER update() so it isn't overwritten.
        # This is the correct injection point for outcome-contingent DA, motor
        # echo boost, and pain penalty -- all need to affect plasticity_multiplier.
        if da_multiplier != 1.0:
            self.neuromodulation.da *= da_multiplier

        # 15b. Handle adolescent phase transitions and update instinct phase
        self._handle_adolescent_phase()
        self.instincts.set_phase(self.neuromodulation.phase)
        _t7 = time.perf_counter()
        _timings["7_neuromod_phase"] = _t7 - _t6

        # 16. STDP / R-STDP updates for plastic connections (skip most steps for speed)
        if self._step_count % self.config.stdp_update_interval == 0:
            self._update_plasticity(modulation)
        _t8 = time.perf_counter()
        _timings["8_stdp"] = _t8 - _t7

        # 16b. Track convergence (H6: weighted by synapse count — larger groups matter more)
        # Also accumulate per-region weight change magnitudes for astrocyte metabolic signal.
        # Only include groups that actually ran STDP this step (adaptive interval
        # may cause some groups to skip — their stale last_stdp_delta must not
        # contribute to convergence tracking).
        if self._step_count % self.config.stdp_update_interval == 0:
            total_nnz = 0
            weighted_delta = 0.0
            region_delta_sum: dict[str, float] = {}
            region_delta_count: dict[str, int] = {}
            base = self._base_stdp_interval
            for _name, syn in self._plastic_synapses:
                # Check if this group actually ran this step
                mult = self._group_stdp_mult.get(_name, 1)
                if self._step_count % (base * mult) != 0:
                    continue  # stale delta — skip
                nnz = syn.nnz
                if nnz > 0:
                    weighted_delta += syn.last_stdp_delta * nnz
                    total_nnz += nnz
                    # Accumulate by post-synaptic region for astrocyte
                    _, post_region = self._get_syn_regions(_name)
                    if post_region is not None:
                        rname = post_region.name
                        region_delta_sum[rname] = (
                            region_delta_sum.get(rname, 0.0) + syn.last_stdp_delta * nnz
                        )
                        region_delta_count[rname] = region_delta_count.get(rname, 0) + nnz
            # Update per-region weight change magnitudes
            self._weight_change_by_region = {
                rname: region_delta_sum[rname] / region_delta_count[rname]
                for rname in region_delta_sum
                if region_delta_count[rname] > 0
            }
            if total_nnz > 0:
                mean_delta = weighted_delta / total_nnz
                self._convergence_history.append(mean_delta)
                if len(self._convergence_history) > self._convergence_window:
                    self._convergence_history = self._convergence_history[
                        -self._convergence_window :
                    ]
                if mean_delta < self._convergence_threshold:
                    self._convergence_stable_count += 1
                else:
                    self._convergence_stable_count = 0

        # 17. Eligibility trace decay + neuromodulatory gating (fused, parallelized)
        # Uses apply_neuromodulation_and_decay() which operates only on _elig_active set.
        # A1: Pass per-synapse plasticity mask directly (no .mean() approximation)
        plasticity_mult = self.neuromodulation.plasticity_multiplier
        # Theta oscillation modulates plasticity: peak = enhanced encoding, trough = consolidation
        if self.oscillators is not None:
            plasticity_mult *= self.oscillators.theta_plasticity_factor()
        is_adol = self.neuromodulation.is_adolescent

        # Eligibility interval: accumulate modulator signal and apply every N steps
        # with compensated decay (decay^N).  Saves N-1 full-array scans per interval.
        elig_interval = self.config.elig_apply_interval
        self._elig_modulator_accum += plasticity_mult

        if self._step_count % elig_interval == 0:
            # Average accumulated modulator over the interval
            avg_modulator = self._elig_modulator_accum / elig_interval
            self._elig_modulator_accum = 0.0

            # Build task list for eligible groups
            elig_groups = []
            for name, syn in self._plastic_synapses:
                if syn.eligibility is not None:
                    mask = None
                    if is_adol and syn.myelinated is not None:
                        mask = syn.get_adolescent_plasticity_mask(self.config.myelination)
                    # CIP-22: Astrocyte fourth-factor gating on eligibility application.
                    # Scale the neuromodulator signal by the astrocyte plasticity gate
                    # for the post-synaptic region.  Gate=1.0 when astrocytes disabled.
                    group_modulator = avg_modulator
                    if self.astrocytes is not None:
                        _, post_region = self._get_syn_regions(name)
                        astro_gate = self.astrocytes.get_plasticity_gate(
                            post_region.name if post_region is not None else ""
                        )
                        group_modulator = avg_modulator * astro_gate
                    # CIP-23: Workspace plasticity boost during adolescent phase.
                    # Low DA (0.14) + strong lateral inhibition = near-zero weight
                    # updates for workspace groups.  Boost compensates without
                    # bypassing neuromod gating (Claim 1d compliant).
                    gw_boost = self.config.global_workspace.adolescent_plasticity_boost
                    if is_adol and gw_boost != 1.0 and "workspace" in name:
                        group_modulator *= gw_boost
                    elig_groups.append((syn, mask, group_modulator))

            if elig_groups:
                if self._stdp_executor is not None and len(elig_groups) > 1:
                    # Parallel Phase 17: each group writes to its own weights.data + eligibility
                    futures = [
                        self._stdp_executor.submit(
                            syn.apply_neuromodulation_and_decay,
                            gmod,
                            mask,
                            elig_interval,
                        )
                        for syn, mask, gmod in elig_groups
                    ]
                    for f in futures:
                        f.result()
                else:
                    for syn, mask, gmod in elig_groups:
                        syn.apply_neuromodulation_and_decay(gmod, mask, elig_interval)

        _t9 = time.perf_counter()
        _timings["9_eligibility"] = _t9 - _t8

        # 17b. Neighborhood consolidation (DA burst → rescue nearby traces)
        if is_adol:
            nc = self.config.neighborhood
            t = self._step_count * self.config.dt
            for name, syn in self._plastic_synapses:
                if syn.eligibility is not None:
                    pre_region, post_region = self._get_syn_regions(name)
                    if pre_region and post_region:
                        syn.apply_neighborhood_consolidation(
                            self.neuromodulation.da,
                            nc,
                            t,
                            pre_region.last_spike_time,
                            post_region.last_spike_time,
                        )

        # 17c. BCM metaplasticity — update per-neuron modification thresholds.
        # Patent Claim 1c: theta tracks postsynaptic firing rate² (EMA, tau=10000 steps).
        # Uses current-step spikes as instantaneous rate estimate; the slow EMA smooths.
        if (
            self._step_count % 10 == 0
        ):  # every 10 steps — theta_tau=10000 makes per-step updates wasteful
            for name, syn in self._plastic_synapses:
                if syn.bcm_theta is not None:
                    _, post_region = self._get_syn_regions(name)
                    if post_region is not None:
                        syn.update_bcm_threshold(post_region.spikes.astype(np.float32))

        # 18. Periodic weight homeostasis (H1/H2: see normalize_weights docstring)
        # Guard: Skip homeostasis entirely during sensory starvation.  Without
        # input, homeostasis pulls all weights toward target_frac (0.5), erasing
        # learned structure.  200-step threshold matches one normal homeostasis
        # interval -- if the brain hasn't received input for that long, freeze.
        _STARVATION_HOMEOSTASIS_THRESHOLD = 200  # noqa: N806
        _sensory_starving = sensory_gap > _STARVATION_HOMEOSTASIS_THRESHOLD

        # Adaptive interval: 10x more frequent when any region fires above 30%.
        # H2: During adolescent, bypass the inverse-plasticity reduction so
        # homeostasis can counterbalance the widened STDP windows (1.5x a_plus).
        _max_rate = max(self._firing_rate_report.values()) if self._firing_rate_report else 0.0
        _home_interval = self.config.homeostasis_interval
        _home_rate = self.config.homeostasis_rate
        if _max_rate > 0.30:
            _home_interval = max(_home_interval // 10, 50)
            _home_rate = min(_home_rate * 3.0, 0.05)
        _adol_bypass = is_adol and self.config.homeostasis_adolescent_bypass
        # Motor groups that should receive homeostasis boost during supported phase.
        # Excludes brainstem_motor (arousal, not learned) and reflex_motor (hardwired).
        _MOTOR_BOOST_GROUPS = {  # noqa: N806
            "sensory_motor",
            "cerebellum_motor",
            "working_motor",
            "workspace_motor",
        }
        _emergency_ceiling = self.config.homeostasis_emergency_ceiling
        if _sensory_starving:
            if self._step_count % 1000 == 0:
                logger.warning(
                    "Homeostasis FROZEN: sensory starvation (gap=%d steps > %d threshold)",
                    sensory_gap,
                    _STARVATION_HOMEOSTASIS_THRESHOLD,
                )
        elif self._step_count % _home_interval == 0:
            self._homeostasis_regular_count += 1
            for _name, syn in self._plastic_synapses:
                _rate = _home_rate
                if self._motor_scaling_boost > 1.0 and _name in _MOTOR_BOOST_GROUPS:
                    _rate *= self._motor_scaling_boost
                # Pass plasticity mask during adolescent so myelinated/identity
                # synapses resist homeostatic correction (they're locked).
                _hmask = None
                if is_adol and syn.myelinated is not None:
                    _hmask = syn.get_adolescent_plasticity_mask(self.config.myelination)
                syn.normalize_weights(
                    target_frac=self.config.homeostasis_target_frac,
                    base_rate=_rate,
                    plasticity_multiplier=plasticity_mult,
                    adolescent_bypass=_adol_bypass,
                    plasticity_mask=_hmask,
                )
        # Emergency ceiling: every 10 steps, check if any group's mean weight
        # exceeds the ceiling and apply 3x-rate homeostasis.  Gated to every
        # 10 steps (not every step) to avoid scanning ~4.76 GB of weight data
        # per step at 1M scale.  Skips groups already corrected by the regular
        # homeostasis round above (double-correction guard).
        _did_regular = self._step_count % _home_interval == 0
        if (
            not _sensory_starving
            and _emergency_ceiling < 1.0
            and self._step_count % 10 == 0
            and not _did_regular
        ):
            for _name, syn in self._plastic_synapses:
                if syn.weights.nnz > 0:
                    _mean_w = float(syn.weights.data.mean())
                    _w_max = syn.stdp_params.w_max
                    if _mean_w > _emergency_ceiling * _w_max:
                        self._homeostasis_emergency_count += 1
                        _emergency_rate = min(_home_rate * 3.0, 0.05)
                        _emask = None
                        if is_adol and syn.myelinated is not None:
                            _emask = syn.get_adolescent_plasticity_mask(self.config.myelination)
                        syn.normalize_weights(
                            target_frac=self.config.homeostasis_target_frac,
                            base_rate=_emergency_rate,
                            plasticity_multiplier=plasticity_mult,
                            adolescent_bypass=_adol_bypass,
                            plasticity_mask=_emask,
                        )
                        # Throttle log: once per 100 steps per group
                        if self._step_count % 100 == 0:
                            logger.info(
                                "Emergency homeostasis: %s mean=%.4f (ceiling=%.2f), "
                                "applied 3x rate",
                                _name,
                                _mean_w,
                                _emergency_ceiling * _w_max,
                            )
        # Reset boost OUTSIDE the interval check to prevent stale boost
        # persisting across task transitions between homeostasis rounds.
        self._motor_scaling_boost = 1.0
        _t10 = time.perf_counter()
        _timings["10_consolidation_homeostasis"] = _t10 - _t9

        # Accumulate spike counts for windowed firing rate
        for name, region in self.regions.items():
            if region.spikes.any():
                self._spike_counts[name] += int(region.spikes.sum())
        self._window_step += 1
        if self._window_step >= self._rate_window:
            for name in self.regions:
                n = self.regions[name].n
                self._firing_rate_report[name] = self._spike_counts[name] / (self._rate_window * n)
                self._spike_counts[name] = 0
            self._window_step = 0

            # Log mean weights for all plastic groups (weight drift monitoring).
            # Throttled to once every _weight_log_interval rate windows to avoid
            # flooding the console on fast hardware.
            self._weight_log_counter += 1
            if (
                self._weight_log_interval > 0
                and self._weight_log_counter >= self._weight_log_interval
            ):
                self._weight_log_counter = 0
                _weight_parts = []
                for _wn, _ws in self._plastic_synapses:
                    if _ws.weights.nnz > 0:
                        _weight_parts.append(f"{_wn}={float(_ws.weights.data.mean()):.4f}")
                if _weight_parts:
                    logger.info("Mean weights: %s", " | ".join(_weight_parts))

            # Homeostasis instrumentation (Phase 2c): log execution counts then reset
            if self._homeostasis_regular_count > 0 or self._homeostasis_emergency_count > 0:
                logger.info(
                    "Homeostasis: regular=%d emergency=%d (interval=%d rate=%.3f bypass=%s)",
                    self._homeostasis_regular_count,
                    self._homeostasis_emergency_count,
                    _home_interval,
                    _home_rate,
                    _adol_bypass,
                )
                self._homeostasis_regular_count = 0
                self._homeostasis_emergency_count = 0

            # Phase 4c: Meta controller burst detection -- log when firing > 5%
            if self.meta_ctrl is not None:
                _meta_rate = self._firing_rate_report.get("meta_controller", 0.0)
                if _meta_rate > 0.05:
                    _nm_out = self.meta_ctrl.read_neuromodulators()
                    logger.warning(
                        "Meta controller burst: rate=%.1f%% DA=%.2f ACh=%.2f NE=%.2f 5HT=%.2f",
                        _meta_rate * 100,
                        _nm_out.get("da", 0),
                        _nm_out.get("ach", 0),
                        _nm_out.get("ne", 0),
                        _nm_out.get("serotonin", 0),
                    )

            # CIP-22: Update astrocyte calcium with fresh firing rates + weight change magnitudes
            if self.astrocytes is not None:
                self.astrocytes.step(
                    self._firing_rate_report,
                    self._weight_change_by_region or None,
                )
                # Apply excitability gate to each region for next step's current scaling
                for rname, region in self.regions.items():
                    region.excitability_scale = self.astrocytes.get_excitability_gate(rname)

        # Compute total before EMA update (don't count EMA overhead in total)
        _t_end = time.perf_counter()
        _timings["total"] = _t_end - _t0

        # Thread-safe update: step() runs in executor, get_step_timing() on event loop
        alpha = self._timing_ema_alpha
        with self._timing_lock:
            self._step_timing = _timings
            for k, v in _timings.items():
                if k not in self._step_timing_ema:
                    self._step_timing_ema[k] = v
                else:
                    self._step_timing_ema[k] = alpha * v + (1.0 - alpha) * self._step_timing_ema[k]

        return {
            "motor_commands": motor_commands,
            "cognitive_commands": cognitive_commands,
            "speech_commands": speech_commands,
            "reflex_responses": [
                {
                    "name": r.reflex_name,
                    "intensity": r.intensity,
                    "action": r.action,
                    "priority": r.priority,
                }
                for r in reflex_responses
            ],
            "prediction_error": prediction_error,
            "step_count": self._step_count,
        }

    def get_step_timing(self, ema: bool = True) -> dict[str, float]:
        """Return step phase timings in milliseconds.

        Thread-safe: can be called from event loop while step() runs in executor.

        Args:
            ema: If True, return EMA-smoothed timings. Otherwise return last step.

        Timing keys:
            1_brainstem_sensory      — Drive injection, brainstem step, sensory routing + step
            2_reflex_feature         — Reflex arc, feature layer
            3_association            — Association cortex (multi-source routing + step)
            4_concept_predictive     — Concept layer + predictive layer + prediction error
            5_meta_wm_cereb_motor    — Meta-controller, working memory, cerebellum, motor cortex
            6_decode                 — Motor/cognitive/speech decoders
            7_neuromod_phase         — Neuromodulation update, adolescent phase handling
            8_stdp                   — STDP/R-STDP weight updates (zero on non-STDP steps)
            9_eligibility            — Eligibility trace decay + neuromodulatory gating
            10_consolidation_homeostasis — Neighborhood consolidation + homeostatic scaling
            total                    — Full step wall time
        """
        with self._timing_lock:
            src = self._step_timing_ema if ema else self._step_timing
            return {k: v * 1000.0 for k, v in src.items()}

    def _get_syn_regions(self, syn_name: str):
        """Return (pre_region, post_region) for a synapse group name."""
        # Map synapse names to their pre→post region keys
        _map = {
            "sensory_association": ("sensory_cortex", "association_cortex"),
            "association_lateral": ("association_cortex", "association_cortex"),
            "sensory_motor": ("sensory_cortex", "motor_cortex"),
            "brainstem_motor": ("brainstem", "motor_cortex"),
            "sensory_cerebellum": ("sensory_cortex", "cerebellum"),
            "motor_cerebellum": ("motor_cortex", "cerebellum"),
            "cerebellum_motor": ("cerebellum", "motor_cortex"),
            "association_predictive": ("association_cortex", "predictive_layer"),
            "predictive_recurrent": ("predictive_layer", "predictive_layer"),
            "predictive_association": ("predictive_layer", "association_cortex"),
            "association_working": ("association_cortex", "working_memory"),
            "working_recurrent": ("working_memory", "working_memory"),
            "working_motor": ("working_memory", "motor_cortex"),
            "sensory_feature": ("sensory_cortex", "feature_layer"),
            "feature_association": ("feature_layer", "association_cortex"),
            "association_concept": ("association_cortex", "concept_layer"),
            "concept_lateral": ("concept_layer", "concept_layer"),
            "concept_predictive": ("concept_layer", "predictive_layer"),
            "concept_working": ("concept_layer", "working_memory"),
            "predictive_concept": ("predictive_layer", "concept_layer"),
            "association_meta": ("association_cortex", "meta_controller"),
            "association_dg": ("association_cortex", "pattern_separator"),
            "dg_concept": ("pattern_separator", "concept_layer"),
            "brainstem_concept": ("brainstem", "concept_layer"),
            "brainstem_dg": ("brainstem", "pattern_separator"),
            # Global Workspace (CIP-23)
            "association_workspace": ("association_cortex", "global_workspace"),
            "predictive_workspace": ("predictive_layer", "global_workspace"),
            "working_workspace": ("working_memory", "global_workspace"),
            "concept_workspace": ("concept_layer", "global_workspace"),
            "feature_workspace": ("feature_layer", "global_workspace"),
            "meta_workspace": ("meta_controller", "global_workspace"),
            "workspace_lateral": ("global_workspace", "global_workspace"),
            "workspace_association": ("global_workspace", "association_cortex"),
            "workspace_predictive": ("global_workspace", "predictive_layer"),
            "workspace_working": ("global_workspace", "working_memory"),
            "workspace_motor": ("global_workspace", "motor_cortex"),
            "workspace_concept": ("global_workspace", "concept_layer"),
            "workspace_feature": ("global_workspace", "feature_layer"),
        }
        if syn_name not in _map:
            return None, None
        pre_key, post_key = _map[syn_name]
        return self.regions.get(pre_key), self.regions.get(post_key)

    def _compute_compartment_activity(self, syn_name: str, post_spikes: np.ndarray) -> float:
        """Compute compartment activity scaling for STDP credit assignment.

        Returns 1.0 (full credit) when the target compartment was active at spike time,
        0.5 (half credit) when inactive. Returns 1.0 if dendrites are disabled.
        """
        syn = self.synapses[syn_name]
        comp = syn.target_compartment
        if comp is None:
            return 1.0
        _, post_region = self._get_syn_regions(syn_name)
        if post_region is None:
            return 1.0
        pop = post_region.population
        if pop.compartment_active_at_spike is None:
            return 1.0
        # Fraction of spiking post-neurons whose target compartment was active
        post_spiked_idx = np.nonzero(post_spikes)[0]
        if len(post_spiked_idx) == 0:
            return 1.0
        active = pop.compartment_active_at_spike[comp, post_spiked_idx]
        # Scale: 0.5 (inactive) to 1.0 (fully active)
        return float(active.mean()) * 0.5 + 0.5

    def _update_adolescent_signals(self) -> None:
        """Feed signals to neuromodulation for adolescent entry/exit evaluation."""
        cfg = self.config.adolescent_entry

        # Sensory rate variance (G4: deque for O(1) append)
        sen_rate = self._firing_rate_report.get("sensory_cortex", 0.0)
        self._sensory_rate_buffer.append(sen_rate)
        variance = (
            float(np.var(list(self._sensory_rate_buffer)))
            if len(self._sensory_rate_buffer) >= 5
            else 1.0
        )

        # Feature STDP current (mean delta across feature layer synapses)
        feat_delta = 0.0
        if "sensory_feature" in self.synapses:
            feat_delta = self.synapses["sensory_feature"].last_stdp_delta
        # C1: Peak decays slowly — prevents stale early peaks from blocking entry
        self._feature_stdp_peak = max(
            self._feature_stdp_peak * cfg.peak_decay,
            feat_delta,
        )

        # G3: Use cached myelination fraction (recomputed after pruning/myelination rounds)
        myel = self._cached_myel_fraction

        # G5: Record association firing pattern for concept differentiation
        if self._step_count % cfg.concept_sample_interval == 0:
            if hasattr(self.association, "spikes"):
                spikes = self.association.spikes.astype(np.float32)
                self.neuromodulation.concept_tracker.record_pattern(spikes)

        self.neuromodulation.update_external_signals(
            sensory_rate_variance=variance,
            feature_stdp_current=feat_delta,
            feature_stdp_peak=self._feature_stdp_peak,
            myelination_fraction=myel,
        )

    def _handle_adolescent_phase(self) -> None:
        """Handle adolescent phase initialization, STDP widening, pruning, myelination."""
        is_adol = self.neuromodulation.is_adolescent

        if is_adol and not self._adolescent_initialized:
            # Entering adolescent phase — initialize arrays and widen STDP
            self._adolescent_initialized = True
            logger.info("Initializing adolescent phase arrays and widening STDP windows")

            adol_stdp = self.config.adolescent_stdp
            for name, syn in self._plastic_synapses:
                syn.init_adolescent_arrays()
                # Backup original STDP params and apply widened windows
                orig = syn.stdp_params
                self._original_stdp_params[name] = STDPParams(
                    a_plus=orig.a_plus,
                    a_minus=orig.a_minus,
                    tau_plus=orig.tau_plus,
                    tau_minus=orig.tau_minus,
                    w_min=orig.w_min,
                    w_max=orig.w_max,
                    min_dt=orig.min_dt,
                    max_dt=orig.max_dt,
                )
                # A2: a_minus uses its own scale factor (default 1.0 — preserves LTP/LTD asymmetry)
                syn.stdp_params = STDPParams(
                    a_plus=orig.a_plus * adol_stdp.a_plus_scale,
                    a_minus=orig.a_minus * adol_stdp.a_minus_scale,
                    tau_plus=adol_stdp.tau_plus,
                    tau_minus=adol_stdp.tau_minus,
                    w_min=orig.w_min,
                    w_max=orig.w_max,
                    min_dt=orig.min_dt,
                    max_dt=orig.max_dt,
                )

        elif not is_adol and self._adolescent_initialized:
            # Exiting adolescent phase — restore original STDP params, clear masks
            self._adolescent_initialized = False
            logger.info("Exiting adolescent phase — restoring STDP parameters")
            for name, syn in self._plastic_synapses:
                if name in self._original_stdp_params:
                    syn.stdp_params = self._original_stdp_params[name]
                syn._adolescent_plasticity_mask = None
            self._original_stdp_params.clear()

        # During adolescent phase: periodic pruning, myelination, identity
        if is_adol:
            prune_cfg = self.config.pruning
            myel_cfg = self.config.myelination

            # Update stability tracking at configured interval
            if self._step_count % myel_cfg.stability_check_interval == 0:
                for _name, syn in self._plastic_synapses:
                    if syn.stability_counter is not None:
                        syn.update_stability(myel_cfg.stability_tolerance)

            # Pruning at configured interval
            if self._step_count % prune_cfg.prune_interval == 0:
                total_pruned = 0
                for _name, syn in self._plastic_synapses:
                    total_pruned += syn.prune(prune_cfg)
                if total_pruned > 0:
                    logger.info(f"Adolescent pruning: removed {total_pruned:,} synapses")
                    # Rebuild plastic synapse list after pruning changes nnz
                    self._plastic_synapses = [
                        (name, syn) for name, syn in self.synapses.items() if syn.plastic
                    ]
                    # Ensure adaptive interval dict covers new groups (if any)
                    for name, _ in self._plastic_synapses:
                        if name not in self._group_stdp_mult:
                            self._group_stdp_mult[name] = 1

            # Myelination at configured interval
            if self._step_count % prune_cfg.prune_interval == 0:
                total_myel = 0
                for _name, syn in self._plastic_synapses:
                    total_myel += syn.myelinate(myel_cfg)
                if total_myel > 0:
                    logger.info(f"Adolescent myelination: locked {total_myel:,} synapses")

                # Identity tagging (C5: uses alternative stability path)
                total_ident = 0
                for _name, syn in self._plastic_synapses:
                    total_ident += syn.tag_identity(
                        min_survival_rounds=prune_cfg.min_survival_rounds,
                        stability_multiplier=myel_cfg.identity_stability_multiplier,
                        stability_window=myel_cfg.stability_window,
                    )
                if total_ident > 0:
                    logger.info(f"Adolescent identity tagging: {total_ident:,} synapses tagged")

                # G3: Update cached myelination fraction after myelination/pruning
                myel_fracs = [syn.myelination_fraction for _n, syn in self._plastic_synapses]
                self._cached_myel_fraction = (
                    sum(myel_fracs) / len(myel_fracs) if myel_fracs else 0.0
                )

                # Update plasticity masks on all synapse groups (H2)
                for _name, syn in self._plastic_synapses:
                    syn._adolescent_plasticity_mask = syn.get_adolescent_plasticity_mask(myel_cfg)

                # Phase 2a: Structured adolescent summary (cumulative per-group stats)
                _total_nnz = 0
                _total_myel = 0
                _total_ident = 0
                _group_parts = []
                for _gn, _gs in self._plastic_synapses:
                    _nnz = _gs.weights.nnz
                    _total_nnz += _nnz
                    _mf = _gs.myelination_fraction
                    _idf = _gs.identity_fraction
                    _total_myel += int(_mf * _nnz)
                    _total_ident += int(_idf * _nnz)
                    if _mf > 0 or _idf > 0:
                        _group_parts.append(f"{_gn}(m={_mf:.1%},id={_idf:.1%})")
                logger.info(
                    "Adolescent summary step=%d: nnz=%s myel=%s(%.1f%%) ident=%s(%.1f%%) | %s",
                    self._step_count,
                    f"{_total_nnz:,}",
                    f"{_total_myel:,}",
                    100 * _total_myel / max(_total_nnz, 1),
                    f"{_total_ident:,}",
                    100 * _total_ident / max(_total_nnz, 1),
                    " ".join(_group_parts) if _group_parts else "none yet",
                )

    def _update_plasticity(self, modulation: float) -> None:
        """Apply STDP/R-STDP to all plastic connections.

        Uses three performance strategies:
        - Sparse spike gathering (in SynapseGroup) avoids scanning all nnz
        - Adaptive per-group intervals skip stable groups
        - ThreadPoolExecutor parallelizes across synapse groups
        """
        dt = self.config.dt
        t = self._step_count * dt
        step = self._step_count

        # Helper to get spikes and times from a region (returns zeros for missing regions)
        def info(name: str):
            r = self.regions.get(name)
            if r is None:
                return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
            return r.spikes, r.last_spike_time

        bs_sp, bs_t = info("brainstem")
        sen_sp, sen_t = info("sensory_cortex")
        mot_sp, mot_t = info("motor_cortex")
        cer_sp, cer_t = info("cerebellum")
        asc_sp, asc_t = info("association_cortex")
        pred_sp, pred_t = info("predictive_layer")
        wm_sp, wm_t = info("working_memory")
        feat_sp, feat_t = info("feature_layer")
        con_sp, con_t = info("concept_layer")
        dg_sp, dg_t = info("pattern_separator")
        meta_sp, meta_t = info("meta_controller")
        ws_sp, ws_t = info("global_workspace")

        s = self.synapses

        # Standard STDP connections
        # Motor groups with R-STDP enabled are routed to rstdp_pairs instead.
        # Both enabled AND motor_rstdp_enabled must be true for R-STDP routing.
        _mfb = self.config.motor_feedback
        _motor_rstdp = _mfb.enabled and _mfb.motor_rstdp_enabled
        stdp_pairs = [
            ("sensory_association", sen_sp, sen_t, asc_sp, asc_t),
            ("association_lateral", asc_sp, asc_t, asc_sp, asc_t),
            ("brainstem_motor", bs_sp, bs_t, mot_sp, mot_t),
            ("sensory_cerebellum", sen_sp, sen_t, cer_sp, cer_t),
            ("motor_cerebellum", mot_sp, mot_t, cer_sp, cer_t),
            ("predictive_association", pred_sp, pred_t, asc_sp, asc_t),
            ("association_working", asc_sp, asc_t, wm_sp, wm_t),
            ("working_recurrent", wm_sp, wm_t, wm_sp, wm_t),
        ]
        # These motor groups use R-STDP when motor feedback is enabled,
        # otherwise fall back to standard STDP
        _motor_stdp_groups = [
            ("sensory_motor", sen_sp, sen_t, mot_sp, mot_t),
            ("cerebellum_motor", cer_sp, cer_t, mot_sp, mot_t),
            ("working_motor", wm_sp, wm_t, mot_sp, mot_t),
        ]
        if not _motor_rstdp:
            stdp_pairs.extend(_motor_stdp_groups)

        # Add hierarchical STDP connections if present
        if "sensory_feature" in s:
            stdp_pairs.append(("sensory_feature", sen_sp, sen_t, feat_sp, feat_t))
        if "feature_association" in s:
            stdp_pairs.append(("feature_association", feat_sp, feat_t, asc_sp, asc_t))
        if "association_concept" in s:
            stdp_pairs.append(("association_concept", asc_sp, asc_t, con_sp, con_t))
        if "concept_lateral" in s:
            stdp_pairs.append(("concept_lateral", con_sp, con_t, con_sp, con_t))
        if "concept_working" in s:
            stdp_pairs.append(("concept_working", con_sp, con_t, wm_sp, wm_t))
        if "predictive_concept" in s:
            stdp_pairs.append(("predictive_concept", pred_sp, pred_t, con_sp, con_t))
        if "association_dg" in s:
            stdp_pairs.append(("association_dg", asc_sp, asc_t, dg_sp, dg_t))
        if "dg_concept" in s:
            stdp_pairs.append(("dg_concept", dg_sp, dg_t, con_sp, con_t))
        if "association_meta" in s:
            stdp_pairs.append(("association_meta", asc_sp, asc_t, meta_sp, meta_t))
        # Global Workspace STDP (CIP-23)
        if "association_workspace" in s:
            stdp_pairs.append(("association_workspace", asc_sp, asc_t, ws_sp, ws_t))
        if "predictive_workspace" in s:
            stdp_pairs.append(("predictive_workspace", pred_sp, pred_t, ws_sp, ws_t))
        if "working_workspace" in s:
            stdp_pairs.append(("working_workspace", wm_sp, wm_t, ws_sp, ws_t))
        if "concept_workspace" in s:
            stdp_pairs.append(("concept_workspace", con_sp, con_t, ws_sp, ws_t))
        if "feature_workspace" in s:
            stdp_pairs.append(("feature_workspace", feat_sp, feat_t, ws_sp, ws_t))
        if "meta_workspace" in s:
            stdp_pairs.append(("meta_workspace", meta_sp, meta_t, ws_sp, ws_t))
        # Workspace broadcast efferents (learn what to broadcast)
        if "workspace_association" in s:
            stdp_pairs.append(("workspace_association", ws_sp, ws_t, asc_sp, asc_t))
        if "workspace_predictive" in s:
            stdp_pairs.append(("workspace_predictive", ws_sp, ws_t, pred_sp, pred_t))
        if "workspace_working" in s:
            stdp_pairs.append(("workspace_working", ws_sp, ws_t, wm_sp, wm_t))
        if "workspace_motor" in s:
            stdp_pairs.append(("workspace_motor", ws_sp, ws_t, mot_sp, mot_t))
        if "workspace_concept" in s:
            stdp_pairs.append(("workspace_concept", ws_sp, ws_t, con_sp, con_t))
        if "workspace_feature" in s:
            stdp_pairs.append(("workspace_feature", ws_sp, ws_t, feat_sp, feat_t))

        # R-STDP connections (prediction-modulated)
        rstdp_pairs = [
            ("association_predictive", asc_sp, asc_t, pred_sp, pred_t),
            ("predictive_recurrent", pred_sp, pred_t, pred_sp, pred_t),
        ]
        if "concept_predictive" in s:
            rstdp_pairs.append(("concept_predictive", con_sp, con_t, pred_sp, pred_t))
        # Motor R-STDP: prediction error gates motor learning
        if _motor_rstdp:
            rstdp_pairs.extend(_motor_stdp_groups)

        # Strategy 3: Adaptive interval — check if each group is due for update
        base = self._base_stdp_interval
        thresh = self._adaptive_stdp_threshold
        max_mult = self._adaptive_stdp_max_mult
        min_rate = self.config.stdp_min_firing_rate

        def _run_stdp(name, pre_sp, pre_t, post_sp, post_t):
            """Run STDP for one synapse group (thread-safe: writes only to own data)."""
            # Adaptive interval: skip if this group isn't due yet
            mult = self._group_stdp_mult.get(name, 1)
            effective_interval = base * mult
            if step % effective_interval != 0:
                return
            if len(pre_sp) == 0 or len(post_sp) == 0:
                return
            if not (pre_sp.any() or post_sp.any()):
                return
            # Skip groups where both populations fire below minimum rate
            if min_rate > 0:
                pre_rate = pre_sp.sum() / len(pre_sp)
                post_rate = post_sp.sum() / len(post_sp)
                if pre_rate < min_rate and post_rate < min_rate:
                    return
            comp_act = self._compute_compartment_activity(name, post_sp)
            s[name].update_weights_stdp(
                pre_sp, post_sp, pre_t, post_t, t, compartment_activity=comp_act
            )
            # Update adaptive interval based on STDP delta
            delta = s[name].last_stdp_delta
            if delta < thresh:
                # Group is stable — double its interval (up to max)
                self._group_stdp_mult[name] = min(mult * 2, max_mult)
            else:
                # Group is active — reset to base interval
                self._group_stdp_mult[name] = 1

        def _run_rstdp(name, pre_sp, pre_t, post_sp, post_t):
            """Run R-STDP for one synapse group (thread-safe)."""
            mult = self._group_stdp_mult.get(name, 1)
            effective_interval = base * mult
            if step % effective_interval != 0:
                return
            if len(pre_sp) == 0 or len(post_sp) == 0:
                return
            if not (pre_sp.any() or post_sp.any()):
                return
            # Skip groups where both populations fire below minimum rate
            if min_rate > 0:
                pre_rate = pre_sp.sum() / len(pre_sp)
                post_rate = post_sp.sum() / len(post_sp)
                if pre_rate < min_rate and post_rate < min_rate:
                    return
            comp_act = self._compute_compartment_activity(name, post_sp)
            s[name].update_weights_rstdp(
                pre_sp, post_sp, pre_t, post_t, t, modulation, compartment_activity=comp_act
            )
            delta = s[name].last_stdp_delta
            if delta < thresh:
                self._group_stdp_mult[name] = min(mult * 2, max_mult)
            else:
                self._group_stdp_mult[name] = 1

        # Strategy 1: Parallel execution via ThreadPoolExecutor
        if self._stdp_executor is not None:
            futures = []
            for name, pre_sp, pre_t, post_sp, post_t in stdp_pairs:
                futures.append(
                    self._stdp_executor.submit(_run_stdp, name, pre_sp, pre_t, post_sp, post_t)
                )
            for name, pre_sp, pre_t, post_sp, post_t in rstdp_pairs:
                futures.append(
                    self._stdp_executor.submit(_run_rstdp, name, pre_sp, pre_t, post_sp, post_t)
                )
            # Wait for all to complete — exceptions propagate on .result()
            for f in futures:
                f.result()
        else:
            # Sequential fallback (single-threaded / small networks)
            for name, pre_sp, pre_t, post_sp, post_t in stdp_pairs:
                _run_stdp(name, pre_sp, pre_t, post_sp, post_t)
            for name, pre_sp, pre_t, post_sp, post_t in rstdp_pairs:
                _run_rstdp(name, pre_sp, pre_t, post_sp, post_t)

    def apply_motor_echo_da(self, boost: float) -> float:
        """Apply a temporary DA boost after a motor command fires.

        This keeps eligibility traces alive on motor pathways during the
        feedback delay window (50-500ms between motor fire and proprioceptive
        consequence).  The boost is multiplicative on the current DA level.

        Returns the original DA value *before* the boost — caller must pass
        this to ``remove_motor_echo_da`` to restore the exact value and avoid
        float drift from repeated multiply/divide cycles.
        """
        saved = self.neuromodulation.da
        self.neuromodulation.da = saved * boost
        return saved

    def remove_motor_echo_da(self, saved_da: float) -> None:
        """Restore DA to the exact value saved before the echo boost."""
        self.neuromodulation.da = saved_da

    def inject_observation(self, data: Any, provenance: str = "") -> np.ndarray:
        """Encode and inject an observation. Returns the sensory current."""
        current = self.encoder.encode(self.sensory, data, provenance, allocator=self.allocator)
        self.sensory.update_subranges(self.allocator.current_ranges)

        # Apply instinctual gain
        modality = _resolve_modality(provenance)
        ranges = self.allocator.current_ranges
        active = (
            {modality: ranges[modality]}
            if modality in ranges and ranges[modality] != (0, 0)
            else {}
        )
        gain = self.instincts.compute_gain(current, active, self._step_count)
        current = current * gain

        # Track modality phase for oscillatory coherence detection
        if self.oscillators is not None:
            self.oscillators.update_modality_phase(modality)

        return current

    def inject_multimodal(self, inputs: dict[str, Any]) -> np.ndarray:
        """Encode and inject multiple simultaneous observations."""
        current = self.encoder.encode_multimodal(self.sensory, inputs, allocator=self.allocator)
        self.sensory.update_subranges(self.allocator.current_ranges)

        # Apply instinctual gain for all active modalities
        ranges = self.allocator.current_ranges
        active = {}
        for provenance in inputs:
            modality = _resolve_modality(provenance)
            if modality in ranges and ranges[modality] != (0, 0):
                active[modality] = ranges[modality]
        gain = self.instincts.compute_gain(current, active, self._step_count)
        current = current * gain

        # Track all modality phases for oscillatory coherence detection
        if self.oscillators is not None:
            for modality in active:
                self.oscillators.update_modality_phase(modality)

        return current

    def inject_motor_teaching(self, channel: str, success: bool, gain: float) -> None:
        """Inject outcome feedback as a decaying pulse into motor cortex sub-range.

        This creates a fast corollary-discharge pathway: motor outcome reaches
        motor cortex in 1 step (not 4+ hops through sensory pipeline).
        The pulse decays exponentially over teaching_decay_steps brain steps,
        so the teaching signal persists across multiple brain computation cycles.

        Success: excitatory current to the channel's neurons (reinforce).
        Failure: inhibitory current (suppress the pattern that failed).
        """
        sr = self.motor.get_subrange(channel)
        if sr is None:
            return
        n = sr.size
        if success:
            current = np.full(n, gain, dtype=np.float32)
        else:
            current = np.full(n, -gain * 0.5, dtype=np.float32)
        # Store in buffer -- additive with any existing pulse on this channel
        if channel in self._teaching_buffer:
            self._teaching_buffer[channel] = self._teaching_buffer[channel] + current
        else:
            self._teaching_buffer[channel] = current

    def inject_standing_pattern(
        self,
        actuator_intensities: dict[str, float],
        gain: float,
    ) -> None:
        """Inject PD-computed standing torques as weak motor cortex current.

        Maps per-actuator target intensities to corresponding motor neuron
        groups via the PopulationVectorDecoder layout.  This provides an
        explicit "answer key" for standing, biased at low gain so STDP
        learns the pattern through reinforcement rather than being overridden.

        Args:
            actuator_intensities: {actuator_name: target_intensity 0-1}
            gain: Current amplitude (0.3 = gentle hint).
        """
        pop_vec = self.motor_decoder._pop_vec
        if not pop_vec.enabled:
            return
        if pop_vec._group_maps is None:
            pop_vec._group_maps = pop_vec._build_group_map(self.motor)
        for channel, groups in pop_vec._group_maps.items():
            sr = self.motor.get_subrange(channel)
            if sr is None:
                continue
            current = np.zeros(sr.size, dtype=np.float32)
            for g_start, g_end, act_name in groups:
                intensity = actuator_intensities.get(act_name, 0.0)
                # Offset relative to sub-range start
                local_start = g_start - sr.start
                local_end = g_end - sr.start
                current[local_start:local_end] = gain * intensity
            self.motor.inject_current_subrange(channel, current)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def is_converged(self) -> bool:
        """True when STDP deltas have been below threshold for patience checks."""
        return self._convergence_stable_count >= self._convergence_patience

    def reset_convergence(self) -> None:
        """Reset convergence tracking (e.g., after advancing to new stimulus)."""
        self._convergence_history.clear()
        self._convergence_stable_count = 0

    def get_state(self) -> dict[str, Any]:
        """Serialize full network state (with copies — safe for async use)."""
        return self._get_state(copy=True)

    def get_state_zerocopy(self) -> dict[str, Any]:
        """Serialize full network state without copying arrays.

        Only safe when the caller guarantees the arrays will not be mutated
        before the state dict is consumed — e.g. inside a fork-based save
        where ``os.fork()`` is called before releasing the network lock.
        """
        return self._get_state(copy=False)

    def _get_state(self, *, copy: bool) -> dict[str, Any]:
        state: dict[str, Any] = {
            "step_count": self._step_count,
            "drives": self.drives.get_state(),
            "neuromodulation": self.neuromodulation.get_state(),
            "regions": {},
            "synapses": {},
            # Strategy 3: Adaptive STDP interval multipliers per group
            "group_stdp_mult": dict(self._group_stdp_mult),
            # Adolescent phase state (P2: prevent double-widening on restore)
            "adolescent_initialized": self._adolescent_initialized,
            "original_stdp_params": {
                name: {
                    "a_plus": p.a_plus,
                    "a_minus": p.a_minus,
                    "tau_plus": p.tau_plus,
                    "tau_minus": p.tau_minus,
                    "w_min": p.w_min,
                    "w_max": p.w_max,
                    "min_dt": p.min_dt,
                    "max_dt": p.max_dt,
                }
                for name, p in self._original_stdp_params.items()
            },
            # P7: Sensory buffer and caches
            "sensory_buffer": self._sensory_buffer.tolist(),
            "cached_myel_fraction": self._cached_myel_fraction,
            "feature_stdp_peak": self._feature_stdp_peak,
            # Decoder state
            "speech_decoder": (
                self.speech_decoder.get_state() if self.speech_decoder.enabled else {}
            ),
            "cognitive_decoder": (
                self.cognitive_decoder.get_state() if self.cognitive_decoder.enabled else {}
            ),
            # Sensory allocator ranges (prevents range remapping across restarts)
            "sensory_allocator": {
                "ranges": {k: list(v) for k, v in self.allocator.current_ranges.items()},
                "frozen_modalities": list(self.allocator._frozen_modalities),
                "last_active": dict(self.allocator._last_active),
            },
            # Eligibility interval modulator accumulator
            "elig_modulator_accum": self._elig_modulator_accum,
            # Oscillatory dynamics phase state
            "oscillators": self.oscillators.get_state() if self.oscillators is not None else {},
            # Astrocyte state (CIP-22)
            "astrocytes": self.astrocytes.get_state() if self.astrocytes is not None else {},
            # Global Workspace state (CIP-23)
            "workspace": self.workspace.get_workspace_state() if self.workspace is not None else {},
        }
        for name, region in self.regions.items():
            state["regions"][name] = region.get_state(copy=copy)
        for name, syn in self.synapses.items():
            state["synapses"][name] = syn.get_state(copy=copy)
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore full network state."""
        self._step_count = state.get("step_count", 0)
        if "drives" in state:
            self.drives.set_state(state["drives"])
        if "neuromodulation" in state:
            self.neuromodulation.set_state(state["neuromodulation"])
        for name, rstate in state.get("regions", {}).items():
            if name in self.regions:
                self.regions[name].set_state(rstate)
        for name, sstate in state.get("synapses", {}).items():
            if name in self.synapses:
                self.synapses[name].set_state(sstate)

        # Sync region clocks: all regions must share the same _time_base,
        # otherwise cross-region STDP computes dt_spike in the millions.
        # last_spike_time is stored as (_time - _time_base), so mismatched
        # _time_base values make spike times incomparable across regions.
        expected_time = float(self._step_count * self.config.dt)
        # Find the majority _time_base (most regions share the same one)
        _time_bases = {}
        for _r in self.regions.values():
            tb = _r.population._time_base
            _time_bases[tb] = _time_bases.get(tb, 0) + 1
        common_time_base = max(_time_bases, key=_time_bases.get)

        for name, region in self.regions.items():
            pop = region.population
            needs_sync = False
            reason = ""
            if abs(pop._time - expected_time) > 1000.0:
                needs_sync = True
                reason = f"time {pop._time:.0f} behind expected {expected_time:.0f}"
            elif abs(pop._time_base - common_time_base) > 1.0:
                needs_sync = True
                reason = f"time_base {pop._time_base:.0f} != common {common_time_base:.0f}"

            if needs_sync:
                logger.info(
                    "Region %s clock sync: %s (setting time_base=%.0f)",
                    name,
                    reason,
                    common_time_base,
                )
                pop._time = expected_time
                pop._time_base = common_time_base
                pop.last_spike_time[:] = np.float32(-1e6)

        # Force-override non-plastic brainstem weights from config.
        # Non-plastic groups never change via STDP, so saved weights reflect
        # the init_weight at the time of creation.  If config defaults change
        # (e.g. brainstem_association 0.5 -> 1.5), the saved state is stale.
        _conn = self.config.connections
        _brainstem_overrides = {
            "brainstem_sensory": _conn.brainstem_sensory_weight,
            "brainstem_association": _conn.brainstem_association_weight,
            "brainstem_cerebellum": _conn.brainstem_cerebellum_weight,
            "brainstem_feature": _conn.brainstem_feature_weight,
            "brainstem_working": _conn.brainstem_working_weight,
            "brainstem_predictive": _conn.brainstem_predictive_weight,
        }
        # Also include conditional brainstem groups
        if hasattr(_conn, "brainstem_concept_weight"):
            _brainstem_overrides["brainstem_concept"] = _conn.brainstem_concept_weight
        if hasattr(_conn, "brainstem_dg_weight"):
            _brainstem_overrides["brainstem_dg"] = _conn.brainstem_dg_weight
        _brainstem_overrides["brainstem_workspace"] = _conn.brainstem_workspace_weight
        for _bname, _bweight in _brainstem_overrides.items():
            if _bname in self.synapses and not self.synapses[_bname].plastic:
                _syn = self.synapses[_bname]
                if _syn.weights.nnz > 0:
                    _old_mean = float(_syn.weights.data.mean())
                    if abs(_old_mean - _bweight) > 0.01:
                        _syn.weights.data[:] = np.float32(_bweight)
                        logger.info(
                            "Override non-plastic %s weights: %.3f -> %.3f (config)",
                            _bname,
                            _old_mean,
                            _bweight,
                        )

        # Strategy 3: Restore adaptive STDP interval multipliers
        saved_mult = state.get("group_stdp_mult", {})
        max_mult = self._adaptive_stdp_max_mult
        for name in self._group_stdp_mult:
            if name in saved_mult:
                self._group_stdp_mult[name] = saved_mult[name]
        # Reset maxed-out multipliers: groups stuck at max likely had zero
        # learning signal due to a transient issue (e.g. clock desync).
        # Give them a fresh start to discover if they can learn now.
        reset_names = [n for n, m in self._group_stdp_mult.items() if m >= max_mult]
        if reset_names:
            for n in reset_names:
                self._group_stdp_mult[n] = 1
            logger.info(
                "Reset %d maxed-out STDP interval multipliers: %s",
                len(reset_names),
                ", ".join(reset_names),
            )

        # P2: Restore adolescent phase state (prevent double-widening)
        self._adolescent_initialized = state.get("adolescent_initialized", False)
        saved_originals = state.get("original_stdp_params", {})
        self._original_stdp_params = {}
        for name, params in saved_originals.items():
            self._original_stdp_params[name] = STDPParams(
                a_plus=params["a_plus"],
                a_minus=params["a_minus"],
                tau_plus=params["tau_plus"],
                tau_minus=params["tau_minus"],
                w_min=params["w_min"],
                w_max=params["w_max"],
                min_dt=params.get("min_dt", 1.0),
                max_dt=params.get("max_dt", 50.0),
            )

        # P7: Restore sensory buffer and caches
        if "sensory_buffer" in state:
            buf = np.array(state["sensory_buffer"], dtype=np.float32)
            if len(buf) == len(self._sensory_buffer):
                self._sensory_buffer = buf
        self._cached_myel_fraction = state.get("cached_myel_fraction", 0.0)
        self._feature_stdp_peak = state.get("feature_stdp_peak", 0.0)
        self._elig_modulator_accum = state.get("elig_modulator_accum", 0.0)

        # Decoder state
        if self.speech_decoder.enabled and "speech_decoder" in state:
            self.speech_decoder.set_state(state["speech_decoder"])
        if self.cognitive_decoder.enabled and "cognitive_decoder" in state:
            self.cognitive_decoder.set_state(state["cognitive_decoder"])

        # Oscillatory dynamics phase restoration
        if self.oscillators is not None and state.get("oscillators"):
            self.oscillators.set_state(state["oscillators"])

        # Astrocyte state restoration (CIP-22)
        if self.astrocytes is not None and state.get("astrocytes"):
            self.astrocytes.set_state(state["astrocytes"])

        # Global Workspace state restoration (CIP-23)
        if self.workspace is not None and state.get("workspace"):
            self.workspace.set_workspace_state(state["workspace"])

        # Restore allocator ranges if saved (prevents range remapping).
        # If not in saved state, fall back to pending_freeze discovery.
        alloc_state = state.get("sensory_allocator")
        if alloc_state and alloc_state.get("ranges"):
            for mod, rng in alloc_state["ranges"].items():
                self.allocator._current_ranges[mod] = tuple(rng)
            frozen = set(alloc_state.get("frozen_modalities", []))
            if frozen:
                self.allocator._frozen = True
                self.allocator._frozen_modalities = frozen
                logger.info(
                    "Restored allocator ranges: %s (frozen: %s)",
                    {k: v for k, v in self.allocator._current_ranges.items() if v != (0, 0)},
                    frozen,
                )
            else:
                # Ranges saved but not frozen yet — set pending freeze
                self.allocator._pending_freeze = True

            # Restore _last_active so gain system and recompute work correctly.
            saved_last_active = alloc_state.get("last_active")
            if saved_last_active:
                self.allocator._last_active = {k: int(v) for k, v in saved_last_active.items()}
            else:
                # Old state format without last_active — initialize all frozen
                # modalities as recently active so gain system treats them as live.
                step = self._step_count
                self.allocator._last_active = {mod: step for mod in frozen}
                logger.info(
                    "Initialized _last_active for %d frozen modalities (old state format)",
                    len(frozen),
                )
        elif self._step_count > 0:
            # Legacy state without allocator — fall back to auto-discovery
            self.allocator._pending_freeze = True

        # Rebuild plasticity masks if in adolescent phase
        if self._adolescent_initialized:
            myel_cfg = self.config.myelination
            for _name, syn in self._plastic_synapses:
                if syn.myelinated is not None:
                    syn._adolescent_plasticity_mask = syn.get_adolescent_plasticity_mask(myel_cfg)

    def get_firing_rates(self) -> dict[str, float]:
        """Get current per-region firing rates (fraction, 0.0-1.0)."""
        return dict(self._firing_rate_report)

    def get_synapse_groups(self) -> list[tuple[str, SynapseGroup]]:
        """Get list of (name, SynapseGroup) for all plastic synapse groups."""
        return list(self._plastic_synapses)

    def get_metrics(self) -> dict[str, Any]:
        """Get current network metrics for monitoring."""
        firing_rates = dict(self._firing_rate_report)

        synapse_stats = {}
        stdp_deltas = {}
        for name, syn in self.synapses.items():
            if syn.nnz > 0:
                synapse_stats[name] = {
                    "nnz": syn.nnz,
                    "mean_weight": float(syn.weights.data.mean()),
                    "max_weight": float(syn.weights.data.max()),
                }
                if syn.plastic:
                    stdp_deltas[name] = syn.last_stdp_delta

        return {
            "step_count": self._step_count,
            "total_neurons": self.config.populations.total,
            "firing_rates": firing_rates,
            "synapse_stats": synapse_stats,
            "stdp_deltas": stdp_deltas,
            "drives": self.drives.get_state(),
            "drives_critical": self.drives.is_critical(),
            "convergence": {
                "is_converged": self.is_converged,
                "stable_count": self._convergence_stable_count,
                "patience": self._convergence_patience,
                "mean_delta": self._convergence_history[-1] if self._convergence_history else 0.0,
            },
            "neuromodulation": {
                "phase": self.neuromodulation.phase,
                "da": round(self.neuromodulation.da, 3),
                "ach": round(self.neuromodulation.ach, 3),
                "ne": round(self.neuromodulation.ne, 3),
                "serotonin": round(self.neuromodulation.serotonin, 3),
                "plasticity_multiplier": round(self.neuromodulation.plasticity_multiplier, 3),
            },
            "cross_modal": self._get_cross_modal_metrics(),
            "oscillatory": (
                {
                    "enabled": True,
                    **{k: round(v, 3) for k, v in self.oscillators.get_metrics().items()},
                }
                if self.oscillators is not None
                else {
                    "enabled": False,
                    "gamma_phase": 0.0,
                    "theta_phase": 0.0,
                    "gamma_gain": 1.0,
                    "theta_factor": 1.0,
                    "coherence": 0.0,
                }
            ),
            "step_timing_ms": self.get_step_timing(ema=True),
        }

    def _get_cross_modal_metrics(self) -> dict[str, Any]:
        """Return cross-modal probe results, throttled to avoid excessive cost.

        At 1M scale the probe involves sparse column slicing (~50ms).
        Running every metrics call (default 10s) while holding _net_lock
        would block training unnecessarily.  Instead, run the full probe
        every Nth call and return the cached result in between.
        """
        self._crossmodal_probe_counter += 1
        if self._crossmodal_probe_counter >= self._crossmodal_probe_interval:
            self._crossmodal_probe_counter = 0
            self._crossmodal_last_result = self.cross_modal_probe.probe_network(self).to_dict()
        return self._crossmodal_last_result
