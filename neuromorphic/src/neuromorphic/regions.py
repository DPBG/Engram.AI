"""Brain regions — each wraps a NeuronPopulation with region-specific params."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neuromorphic.config import (
    DendriticCompartmentConfig,
    DualInhibitionConfig,
    InhibitoryConfig,
    LIFParams,
    NeuromorphicConfig,
    NMDAConfig,
)
from neuromorphic.neurons import NeuronPopulation


@dataclass
class SubRange:
    """A named sub-range within a region's neuron population."""

    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

    def slice(self) -> slice:
        return slice(self.start, self.end)


class BrainRegion:
    """Base class for all brain regions."""

    name: str = "base"

    def __init__(
        self,
        n: int,
        lif_params: LIFParams,
        rng: np.random.Generator | None = None,
        inhibitory_config: InhibitoryConfig | None = None,
        dendrite_config: DendriticCompartmentConfig | None = None,
        dual_inhibition_config: DualInhibitionConfig | None = None,
        nmda_config: NMDAConfig | None = None,
    ):
        self.n = n
        self.population = NeuronPopulation(
            n,
            lif_params,
            rng=rng,
            inhibitory_config=inhibitory_config,
            dendrite_config=dendrite_config,
            dual_inhibition_config=dual_inhibition_config,
            nmda_config=nmda_config,
        )
        self.sub_ranges: list[SubRange] = []
        self._external_current = np.zeros(n, dtype=np.float32)
        # Astrocyte excitability gate: scales total input current (1.0 = no effect)
        self.excitability_scale: float = 1.0

    def inject_current(self, current: np.ndarray) -> None:
        """Inject external current into this region (additive)."""
        self._external_current += current

    def inject_compartment_current(self, compartment: int, current: np.ndarray) -> None:
        """Route current to a specific dendritic compartment."""
        self.population.inject_compartment_current(compartment, current)

    def inject_current_subrange(self, sub_name: str, current: np.ndarray) -> None:
        """Inject current into a named sub-range."""
        sr = self.get_subrange(sub_name)
        if sr is None:
            return
        self._external_current[sr.slice()] += current[: sr.size]

    def pre_step(self) -> None:
        """Hook called before step — override for region-specific logic."""
        pass

    def step(self, synaptic_current: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Step the region's neurons. Returns spike array."""
        self.pre_step()
        total_current = synaptic_current + self._external_current
        if self.excitability_scale != 1.0:
            total_current *= np.float32(self.excitability_scale)
        spikes = self.population.step(total_current, dt)
        # Clear external current after use
        self._external_current[:] = 0.0
        return spikes

    def get_subrange(self, name: str) -> SubRange | None:
        for sr in self.sub_ranges:
            if sr.name == name:
                return sr
        return None

    @property
    def spikes(self) -> np.ndarray:
        return self.population.spikes

    @property
    def last_spike_time(self) -> np.ndarray:
        return self.population.last_spike_time

    @property
    def time(self) -> float:
        return self.population.time

    def get_state(self, *, copy: bool = True) -> dict[str, Any]:
        return self.population.get_state(copy=copy)

    def set_state(self, state: dict[str, Any]) -> None:
        self.population.set_state(state)


class Brainstem(BrainRegion):
    """
    Homeostatic drives — energy, damage, temperature, fatigue.
    Slow dynamics (high tau). Always active.
    """

    name = "brainstem"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.brainstem
        super().__init__(n, config.brainstem_lif, rng=rng)
        quarter = n // 4
        self.sub_ranges = [
            SubRange("energy", 0, quarter),
            SubRange("damage", quarter, quarter * 2),
            SubRange("temperature", quarter * 2, quarter * 3),
            SubRange("fatigue", quarter * 3, n),
        ]


class ReflexArc(BrainRegion):
    """
    Fast hardwired responses — pain withdrawal, grip, startle, flinch.
    Fast dynamics (low tau, low threshold).
    """

    name = "reflex_arc"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.reflex_arc
        super().__init__(n, config.reflex_lif, rng=rng)
        quarter = n // 4
        self.sub_ranges = [
            SubRange("pain_withdrawal", 0, quarter),
            SubRange("grip", quarter, quarter * 2),
            SubRange("startle", quarter * 2, quarter * 3),
            SubRange("flinch", quarter * 3, n),
        ]


class SensoryCortex(BrainRegion):
    """
    Encodes raw sensory input. Sub-ranges: visual, auditory, tactile, proprioceptive.
    """

    name = "sensory_cortex"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.sensory_cortex
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            config.lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )
        enc = config.encoding
        vis_end = int(n * enc.visual_end)
        aud_end = int(n * enc.auditory_end)
        tac_end = int(n * enc.tactile_end)
        self.sub_ranges = [
            SubRange("visual", 0, vis_end),
            SubRange("auditory", vis_end, aud_end),
            SubRange("tactile", aud_end, tac_end),
            SubRange("proprioceptive", tac_end, n),
        ]

    def update_subranges(self, ranges: dict[str, tuple[int, int]]) -> None:
        """Update subranges from dynamic allocator output."""
        self.sub_ranges = [SubRange(name, start, end) for name, (start, end) in ranges.items()]


class MotorCortex(BrainRegion):
    """
    Generates movement commands.
    Sub-ranges: locomotion, manipulation, head, [speech], expression, [cognitive].
    """

    name = "motor_cortex"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.motor_cortex
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            config.lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )
        dec = config.decoding
        loc_end = int(n * dec.locomotion_end)
        man_end = int(n * dec.manipulation_end)
        head_end = int(n * dec.head_end)
        speech_end = int(n * dec.speech_end)
        expr_end = int(n * dec.expression_end)
        self.sub_ranges = [
            SubRange("locomotion", 0, loc_end),
            SubRange("manipulation", loc_end, man_end),
            SubRange("head", man_end, head_end),
        ]
        # Speech sub-range: brain-native language output (learned via STDP)
        # Only created when speech_end > head_end
        if speech_end > head_end:
            self.sub_ranges.append(SubRange("speech", head_end, speech_end))
            self.sub_ranges.append(SubRange("expression", speech_end, expr_end))
        else:
            self.sub_ranges.append(SubRange("expression", head_end, expr_end))
        # Cognitive sub-range: non-physical actions (LLM queries, memory ops)
        # Only created when expression_end < 1.0 (i.e., cognitive channel enabled)
        if dec.expression_end < 1.0:
            self.sub_ranges.append(SubRange("cognitive", expr_end, n))


class Cerebellum(BrainRegion):
    """Motor coordination via efference copy feedback + STDP."""

    name = "cerebellum"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.cerebellum
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            config.lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )


class AssociationCortex(BrainRegion):
    """
    Multi-modal binding. Simultaneous inputs from different senses → STDP
    links them → unified concept representations emerge.
    Has lateral (self→self) connections for cross-modal integration.
    """

    name = "association_cortex"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.association_cortex
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            config.lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )


class PredictiveLayer(BrainRegion):
    """
    Temporal sequence learning + world model.
    Learns "after A comes B". Prediction errors drive stronger learning (R-STDP).
    Recurrent connections for sequence memory.
    """

    name = "predictive_layer"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.predictive_layer
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            config.lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )
        # Track last-step activity for prediction error computation
        self._prev_spikes = np.zeros(n, dtype=bool)

    def pre_step(self) -> None:
        self._prev_spikes = self.population.spikes.copy()

    @property
    def prev_spikes(self) -> np.ndarray:
        return self._prev_spikes


class WorkingMemory(BrainRegion):
    """
    Sustained attention via strong recurrent connections.
    Holds active representations for deliberate processing.
    Smaller by design (bottleneck, like real working memory).
    """

    name = "working_memory"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.working_memory
        dend = config.dendrites if config.dendrites.enabled else None
        nmda = config.nmda if config.nmda.enabled else None
        super().__init__(
            n,
            config.working_memory_lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
            nmda_config=nmda,
        )


class FeatureLayer(BrainRegion):
    """
    V4/IT-like intermediate feature integration layer.
    Pools simple sensory features into complex feature combinations
    before multi-modal binding in association cortex.
    """

    name = "feature_layer"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.feature_layer
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            config.lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )


class ConceptLayer(BrainRegion):
    """
    Information bottleneck via k-Winners-Take-All (k-WTA).
    Forces 50:1 compression from association cortex into sparse distributed
    representations (SDRs). Concepts emerge from compression, not engineering.
    """

    name = "concept_layer"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.concept_layer
        cl = config.concept_layer
        concept_lif = LIFParams(
            tau=cl.tau,
            threshold=cl.threshold,
            reset=-70.0,
            resting=-65.0,
            noise_std=0.3,
        )
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            concept_lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )
        self._k = min(cl.k_winners, n)  # k cannot exceed population size

    def step(self, synaptic_current: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Step with k-WTA: only top-k neurons fire each step."""
        self.pre_step()
        total_current = synaptic_current + self._external_current
        spikes = self.population.step(total_current, dt)

        # k-WTA: keep only top-k spikes by drive strength
        if spikes.sum() > self._k:
            spike_idx = np.nonzero(spikes)[0]
            # Use _pre_spike_drive which includes dendritic contribution
            drive = self.population._pre_spike_drive[spike_idx]
            if len(spike_idx) > self._k:
                # argpartition deterministically picks top-k (handles ties)
                top_k_local = np.argpartition(drive, -self._k)[-self._k :]
                keep = spike_idx[top_k_local]
                suppress = np.setdiff1d(spike_idx, keep)
                spikes[suppress] = False
                self.population.spikes = spikes
                # Reset suppressed neurons back to resting (they didn't "really" fire)
                self.population.v_membrane[suppress] = self.population.params.resting
                self.population.refractory_timer[suppress] = 0.0
                self.population.last_spike_time[suppress] = -1e6
                # Undo SFA increment — suppressed neurons didn't "really" fire
                self.population._sfa_offset[suppress] = np.maximum(
                    self.population._sfa_offset[suppress] - self.population._sfa_increment,
                    0.0,
                )
                # Reset suppressed dendrites to prevent phantom buildup
                if self.population.v_dendrite is not None:
                    for c in range(self.population.n_compartments):
                        self.population.v_dendrite[c, suppress] = self.population._dend_cfg.resting[
                            c
                        ]
                    # Clear activity flags so suppressed neurons don't get STDP credit
                    self.population.compartment_active_at_spike[:, suppress] = False

        self._external_current[:] = 0.0
        return spikes


class PatternSeparator(BrainRegion):
    """Dentate gyrus analog — sparse expansion for episodic pattern separation.

    Transforms similar association cortex inputs into dissimilar sparse
    representations before concept layer storage.  Uses k-WTA with high
    sparsity (~2%) to create orthogonal codes that prevent catastrophic
    interference when storing similar memories.

    Architecture: association -> pattern_separator -> concept

    Biological basis: dentate gyrus granule cells have very high thresholds
    and massive inhibition, producing extremely sparse outputs (~1-5%
    active at any time).  This forces dissimilar codes for similar inputs.
    """

    name = "pattern_separator"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.pattern_separator
        dg = config.pattern_separator
        dg_lif = LIFParams(
            tau=dg.tau,
            threshold=dg.threshold,
            reset=-70.0,
            resting=-65.0,
            noise_std=0.2,
        )
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            dg_lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )
        self._k = min(dg.k_winners, n)

    def step(self, synaptic_current: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Step with k-WTA: very sparse output (~2% active)."""
        self.pre_step()
        total_current = synaptic_current + self._external_current
        spikes = self.population.step(total_current, dt)

        # k-WTA: keep only top-k spikes
        if spikes.sum() > self._k:
            spike_idx = np.nonzero(spikes)[0]
            drive = self.population._pre_spike_drive[spike_idx]
            if len(spike_idx) > self._k:
                top_k_local = np.argpartition(drive, -self._k)[-self._k :]
                keep = spike_idx[top_k_local]
                suppress = np.setdiff1d(spike_idx, keep)
                spikes[suppress] = False
                self.population.spikes = spikes
                self.population.v_membrane[suppress] = self.population.params.resting
                self.population.refractory_timer[suppress] = 0.0
                self.population.last_spike_time[suppress] = -1e6
                # Undo SFA for suppressed neurons
                self.population._sfa_offset[suppress] = np.maximum(
                    self.population._sfa_offset[suppress] - self.population._sfa_increment,
                    0.0,
                )
                # Reset suppressed dendrites
                if self.population.v_dendrite is not None:
                    for c in range(self.population.n_compartments):
                        self.population.v_dendrite[c, suppress] = self.population._dend_cfg.resting[
                            c
                        ]
                    self.population.compartment_active_at_spike[:, suppress] = False

        self._external_current[:] = 0.0
        return spikes


class MetaControllerRegion(BrainRegion):
    """
    Neuromodulatory hub — monitors network-wide statistics and outputs
    4 neuromodulatory signals (DA, ACh, NE, 5-HT).
    """

    name = "meta_controller"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.meta_controller
        mc = config.meta_controller
        meta_lif = LIFParams(
            tau=mc.tau,
            threshold=-55.0,
            reset=-70.0,
            resting=-65.0,
            noise_std=0.3,
            sfa_increment=2.0,  # ~7x cortical — primary rate control for meta
            sfa_decay=0.99,
        )
        super().__init__(
            n,
            meta_lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dual_inhibition_config=config.dual_inhibition,
        )

        # Sub-populations
        pos = 0
        self._monitor_end = pos + int(n * mc.monitor_frac)
        pos = self._monitor_end
        self._da_end = pos + int(n * mc.da_frac)
        pos = self._da_end
        self._ach_end = pos + int(n * mc.ach_frac)
        pos = self._ach_end
        self._ne_end = pos + int(n * mc.ne_frac)
        pos = self._ne_end
        self._serotonin_end = pos + int(n * mc.serotonin_frac)
        pos = self._serotonin_end

        self.sub_ranges = [
            SubRange("monitor", 0, self._monitor_end),
            SubRange("da", self._monitor_end, self._da_end),
            SubRange("ach", self._da_end, self._ach_end),
            SubRange("ne", self._ach_end, self._ne_end),
            SubRange("serotonin", self._ne_end, self._serotonin_end),
            SubRange("integrator", self._serotonin_end, n),
        ]
        self._mc_config = mc

    def read_neuromodulators(self) -> dict[str, float]:
        """Read current neuromodulatory outputs as firing-rate-based signals."""
        mc = self._mc_config
        result = {}
        for name, sr_name, (lo, hi) in [
            ("da", "da", mc.da_range),
            ("ach", "ach", mc.ach_range),
            ("ne", "ne", mc.ne_range),
            ("serotonin", "serotonin", mc.serotonin_range),
        ]:
            sr = self.get_subrange(sr_name)
            if sr is None or sr.size == 0:
                result[name] = (lo + hi) / 2
                continue
            rate = float(self.spikes[sr.slice()].mean())
            # Map firing rate [0, 1] → output range [lo, hi]
            result[name] = lo + rate * (hi - lo)
        return result


class GlobalWorkspace(BrainRegion):
    """
    Global Neuronal Workspace (Dehaene-Changeux GNW Theory, CIP-23).

    A dedicated cortical workspace where neural coalitions from higher-order
    regions compete via strong lateral inhibition.  When a coalition's firing
    rate crosses a nonlinear ignition threshold, its signal is broadcast to
    all regions, creating a unified "conscious access" moment.

    Provides principled multi-goal arbitration for embodied agents: when the
    body must walk + avoid obstacle + maintain balance, GWT determines which
    goal gets priority motor access.

    The workspace is intentionally small (bottleneck) to force competition.
    """

    name = "global_workspace"

    def __init__(self, config: NeuromorphicConfig, rng: np.random.Generator | None = None):
        n = config.populations.global_workspace
        gw = config.global_workspace
        dend = config.dendrites if config.dendrites.enabled else None
        super().__init__(
            n,
            config.lif,
            rng=rng,
            inhibitory_config=config.inhibitory,
            dendrite_config=dend,
            dual_inhibition_config=config.dual_inhibition,
        )
        self._gw_cfg = gw
        self._refractory_counter: int = 0
        self._ignition_active: bool = False
        self._ignition_count: int = 0
        self._firing_rate_history: list[float] = []
        self._history_len: int = 10  # rolling window for rate smoothing

    @property
    def ignition_active(self) -> bool:
        """Whether the workspace is currently in an ignition state."""
        return self._ignition_active

    @property
    def ignition_strength(self) -> float:
        """Sigmoid-gated ignition strength [0, 1]. Used for broadcast gain."""
        if not self._firing_rate_history:
            return 0.0
        rate = sum(self._firing_rate_history) / len(self._firing_rate_history)
        gw = self._gw_cfg
        return float(1.0 / (1.0 + np.exp(-gw.sigmoid_slope * (rate - gw.ignition_threshold))))

    def step(self, external_input: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Step workspace, detect ignition, manage refractory."""
        gw = self._gw_cfg

        # During post-ignition refractory, suppress all input
        # (both direct external_input AND current accumulated via inject_current)
        if self._refractory_counter > 0:
            self._refractory_counter -= 1
            self._ignition_active = False
            external_input = np.zeros_like(external_input)
            self._external_current[:] = 0.0

        spikes = super().step(external_input, dt)

        # Track firing rate
        rate = float(spikes.mean())
        self._firing_rate_history.append(rate)
        if len(self._firing_rate_history) > self._history_len:
            self._firing_rate_history.pop(0)

        # Ignition detection
        strength = self.ignition_strength
        if strength > 0.5 and self._refractory_counter == 0:
            self._ignition_active = True
            self._ignition_count += 1
            self._refractory_counter = gw.refractory_steps
        elif self._refractory_counter == 0:
            self._ignition_active = False

        return spikes

    def get_broadcast_gain(self) -> float:
        """Multiplicative gain for efferent broadcast synapses.

        Returns broadcast_gain during ignition, 1.0 otherwise.
        """
        if self._ignition_active:
            return self._gw_cfg.broadcast_gain
        return 1.0

    def get_workspace_state(self) -> dict:
        """Workspace-specific state for persistence."""
        return {
            "refractory_counter": self._refractory_counter,
            "ignition_active": self._ignition_active,
            "ignition_count": self._ignition_count,
            "firing_rate_history": list(self._firing_rate_history),
        }

    def set_workspace_state(self, state: dict) -> None:
        """Restore workspace-specific state."""
        self._refractory_counter = state.get("refractory_counter", 0)
        self._ignition_active = state.get("ignition_active", False)
        self._ignition_count = state.get("ignition_count", 0)
        self._firing_rate_history = state.get("firing_rate_history", [])


def create_all_regions(
    config: NeuromorphicConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, BrainRegion]:
    """Instantiate all brain regions with independent RNGs."""
    rng = rng or np.random.default_rng()

    # Count how many regions we need
    n_regions = 8  # base regions
    if config.populations.feature_layer > 0:
        n_regions += 1
    if config.populations.concept_layer > 0:
        n_regions += 1
    if config.populations.pattern_separator > 0:
        n_regions += 1
    if config.populations.meta_controller > 0:
        n_regions += 1
    if config.populations.global_workspace > 0:
        n_regions += 1

    child_rngs = rng.spawn(n_regions)
    ri = iter(child_rngs)

    regions: dict[str, BrainRegion] = {
        "brainstem": Brainstem(config, next(ri)),
        "reflex_arc": ReflexArc(config, next(ri)),
        "sensory_cortex": SensoryCortex(config, next(ri)),
        "motor_cortex": MotorCortex(config, next(ri)),
        "cerebellum": Cerebellum(config, next(ri)),
        "association_cortex": AssociationCortex(config, next(ri)),
        "predictive_layer": PredictiveLayer(config, next(ri)),
        "working_memory": WorkingMemory(config, next(ri)),
    }

    if config.populations.feature_layer > 0:
        regions["feature_layer"] = FeatureLayer(config, next(ri))
    if config.populations.concept_layer > 0:
        regions["concept_layer"] = ConceptLayer(config, next(ri))
    if config.populations.pattern_separator > 0:
        regions["pattern_separator"] = PatternSeparator(config, next(ri))
    if config.populations.meta_controller > 0:
        regions["meta_controller"] = MetaControllerRegion(config, next(ri))
    if config.populations.global_workspace > 0:
        regions["global_workspace"] = GlobalWorkspace(config, next(ri))

    return regions
