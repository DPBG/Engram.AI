"""Leaky integrate-and-fire neuron populations — vectorized NumPy."""

from __future__ import annotations

import numpy as np

from neuromorphic.config import (
    DendriticCompartmentConfig,
    DualInhibitionConfig,
    InhibitoryConfig,
    LIFParams,
    NMDAConfig,
)


class NeuronPopulation:
    """
    A population of N leaky integrate-and-fire (LIF) neurons.

    All state is stored as parallel float32 NumPy arrays for efficient
    vectorized computation. Supports E/I split: inhibitory neurons have
    faster tau, shorter refractory, and produce negative output current.
    """

    def __init__(
        self,
        n: int,
        params: LIFParams,
        rng: np.random.Generator | None = None,
        inhibitory_config: InhibitoryConfig | None = None,
        dendrite_config: DendriticCompartmentConfig | None = None,
        dual_inhibition_config: DualInhibitionConfig | None = None,
        nmda_config: NMDAConfig | None = None,
    ):
        self.n = n
        self.params = params
        self._rng = rng or np.random.default_rng()

        # E/I split: build per-neuron parameter arrays
        self.is_inhibitory = np.zeros(n, dtype=bool)
        if inhibitory_config is not None and inhibitory_config.inhibitory_fraction > 0:
            n_inh = int(n * inhibitory_config.inhibitory_fraction)
            self.is_inhibitory[n - n_inh :] = True  # last n_inh neurons are inhibitory
            self._inh_cfg = inhibitory_config
        else:
            self._inh_cfg = None

        # Dual inhibition: PV-fast (GABA-A) vs SST-slow (GABA-B) subtypes (CIP-21)
        self._dual_inh_cfg = dual_inhibition_config
        self.is_sst = np.zeros(n, dtype=bool)
        if (
            dual_inhibition_config is not None
            and dual_inhibition_config.enabled
            and self._inh_cfg is not None
        ):
            n_inh = int(self.is_inhibitory.sum())
            n_pv = int(n_inh * dual_inhibition_config.pv_fraction)
            n_sst = n_inh - n_pv
            if n_sst > 0:
                # SST neurons are the LAST n_sst of the inhibitory pool
                inh_indices = np.nonzero(self.is_inhibitory)[0]
                self.is_sst[inh_indices[-n_sst:]] = True
            self._gaba_b_enabled = True
        else:
            self._gaba_b_enabled = False

        # Per-neuron LIF parameters (allows E/I heterogeneity)
        self._tau = np.full(n, params.tau, dtype=np.float32)
        self._threshold = np.full(n, params.threshold, dtype=np.float32)
        self._refractory_ms = np.full(n, params.refractory_ms, dtype=np.float32)

        if self._inh_cfg is not None:
            inh_mask = self.is_inhibitory
            self._tau[inh_mask] = self._inh_cfg.inhibitory_tau
            self._threshold[inh_mask] = self._inh_cfg.inhibitory_threshold
            self._refractory_ms[inh_mask] = self._inh_cfg.inhibitory_refractory_ms

        # Override SST-specific params (slower tau, different threshold)
        if self._gaba_b_enabled:
            sst_mask = self.is_sst
            self._tau[sst_mask] = dual_inhibition_config.sst_tau
            self._threshold[sst_mask] = dual_inhibition_config.sst_threshold
            self._refractory_ms[sst_mask] = dual_inhibition_config.sst_refractory_ms

        # State arrays (float32 to halve memory)
        self.v_membrane = np.full(n, params.resting, dtype=np.float32)
        self.refractory_timer = np.zeros(n, dtype=np.float32)
        self.spikes = np.zeros(n, dtype=bool)
        self.last_spike_time = np.full(n, -1e6, dtype=np.float32)

        # Output sign: +1 for excitatory, -1 for inhibitory
        self.output_sign = np.ones(n, dtype=np.float32)
        if self._inh_cfg is not None:
            self.output_sign[self.is_inhibitory] = self._inh_cfg.inhibitory_weight_scale

        # GABA-B slow inhibitory conductance (CIP-21)
        # Accumulates when SST neurons fire (via lateral synapses), decays with
        # tau ~150ms. Applied as slow hyperpolarizing current each step.
        if self._gaba_b_enabled:
            di = dual_inhibition_config
            self._gaba_b_conductance = np.zeros(n, dtype=np.float32)
            self._gaba_b_decay = np.float32(np.exp(-1.0 / di.gaba_b_tau))
            self._gaba_b_reversal = np.float32(di.gaba_b_reversal)
            self._gaba_b_increment = np.float32(di.gaba_b_conductance_increment)
        else:
            self._gaba_b_conductance = None

        # NMDA slow excitatory conductance (CIP-24)
        # Accumulates from recurrent WM spikes, decays with tau ~100ms.
        # Voltage-dependent Mg2+ block creates bistable attractor states.
        self._nmda_cfg = nmda_config
        if nmda_config is not None and nmda_config.enabled:
            self._nmda_conductance = np.zeros(n, dtype=np.float32)
            self._nmda_decay = np.float32(np.exp(-1.0 / nmda_config.tau_nmda))
            self._nmda_e_rev = np.float32(nmda_config.e_nmda)
            self._nmda_mg = np.float32(nmda_config.mg_concentration)
            self._nmda_slope = np.float32(nmda_config.mg_block_slope)
            self._nmda_scale = np.float32(nmda_config.conductance_scale)
        else:
            self._nmda_conductance = None

        # Spike Frequency Adaptation (SFA): per-neuron threshold offset
        # Increases on each spike, decays exponentially each step.
        self._sfa_offset = np.zeros(n, dtype=np.float32)
        self._sfa_increment = np.float32(params.sfa_increment)
        self._sfa_decay = np.float32(params.sfa_decay)

        # Simulation clock
        self._time: float = 0.0
        self._time_base: float = 0.0  # rebase offset for float32 precision

        # Pre-spike drive (used by k-WTA ranking in ConceptLayer)
        self._pre_spike_drive: np.ndarray = np.zeros(n, dtype=np.float32)

        # Dendritic compartments (None = point neuron, backward compatible)
        self._dend_cfg = dendrite_config
        if dendrite_config is not None and dendrite_config.enabled:
            nc = dendrite_config.n_compartments
            self.n_compartments = nc
            # Per-compartment membrane potential: shape (nc, n)
            self.v_dendrite = np.zeros((nc, n), dtype=np.float32)
            for c in range(nc):
                self.v_dendrite[c, :] = dendrite_config.resting[c]
            # Per-compartment tau and soma coupling (small arrays, indexed per-compartment)
            self._tau_dend = np.array(dendrite_config.tau, dtype=np.float32)
            self._soma_coupling = np.array(dendrite_config.soma_coupling, dtype=np.float32)
            # Dendritic spike absolute thresholds (mV)
            self._dend_spike_thresh: list[float | None] = [
                dendrite_config.resting[c] + t if t is not None else None
                for c, t in enumerate(dendrite_config.dend_spike_threshold)
            ]
            self._dend_spike_boost = np.float32(dendrite_config.dend_spike_boost)
            self._dend_refractory = np.zeros((nc, n), dtype=np.float32)
            self._dend_refractory_ms = np.float32(dendrite_config.dend_spike_refractory_ms)
            self._plateau_fraction = np.float32(dendrite_config.dend_spike_plateau_fraction)
            self._dend_voltage_floor = np.float32(dendrite_config.dend_voltage_floor)
            # Track which compartments were active at somatic spike time (for STDP credit)
            self.compartment_active_at_spike = np.zeros((nc, n), dtype=bool)
            # Accumulated compartment input buffer (zeroed each step)
            self._compartment_input = np.zeros((nc, n), dtype=np.float32)
            # Reusable buffer for dendritic spike tracking (avoids per-step allocation)
            self._dend_spike_fired = np.zeros((nc, n), dtype=bool)
        else:
            self.n_compartments = 0
            self.v_dendrite = None
            self.compartment_active_at_spike = None
            self._compartment_input = None

    def inject_compartment_current(self, compartment: int, current: np.ndarray) -> None:
        """Accumulate current into a specific dendritic compartment.

        Called by network step() to route each synapse group's output
        to the correct dendritic domain. No-op if dendrites are disabled
        or compartment index is out of range.
        """
        if self._compartment_input is not None and 0 <= compartment < self.n_compartments:
            self._compartment_input[compartment] += current

    def step_dendrites(self, dt: float = 1.0) -> np.ndarray:
        """Phase 1: integrate dendritic compartments and compute somatic current.

        Each compartment: dV/dt = (-(V - V_rest) + I) / tau_dend
        If dendritic spike threshold is crossed, apply supralinear boost.
        Returns total somatic current from all compartments (shape: (n,)).
        """
        if self.v_dendrite is None:
            return np.zeros(self.n, dtype=np.float32)

        cfg = self._dend_cfg
        soma_current = np.zeros(self.n, dtype=np.float32)
        compartment_input = self._compartment_input

        # Reuse pre-allocated buffer for dendritic spike tracking
        dend_spike_fired = self._dend_spike_fired
        dend_spike_fired[:] = False

        for c in range(self.n_compartments):
            v = self.v_dendrite[c]
            rest = np.float32(cfg.resting[c])
            tau = self._tau_dend[c]
            I_c = compartment_input[c]  # noqa: N806

            # Decrease dendritic refractory timers
            np.maximum(
                self._dend_refractory[c] - np.float32(dt),
                np.float32(0.0),
                out=self._dend_refractory[c],
            )
            active = self._dend_refractory[c] <= 0.0

            # Leaky integration (in-place to avoid temporary arrays)
            dv = (-(v - rest) + I_c) * (dt / tau)
            dv *= active
            v += dv
            # Voltage floor — prevents runaway hyperpolarization from
            # inhibitory-dominated input (biological K+ reversal ≈ -90mV)
            np.maximum(v, self._dend_voltage_floor, out=v)

            # Dendritic spike detection
            boost = np.ones(self.n, dtype=np.float32)
            thresh = self._dend_spike_thresh[c]
            _spiked = None
            if thresh is not None:
                dend_spiked = (v >= thresh) & active
                if dend_spiked.any():
                    boost[dend_spiked] = self._dend_spike_boost
                    dend_spike_fired[c] = dend_spiked
                    _spiked = dend_spiked
                    self._dend_refractory[c, dend_spiked] = self._dend_refractory_ms

            # Couple to soma BEFORE voltage reset (Patent Claim 5: supralinear
            # amplification). Previous code reset voltage first, making boost * 0 = 0.
            # Clamp to non-negative: sub-resting dendrites don't actively hyperpolarize
            # soma. Inhibition comes through lateral connections in the target region.
            dend_drive = np.maximum(self.v_dendrite[c] - rest, np.float32(0.0))
            soma_current += self._soma_coupling[c] * dend_drive * boost

            # Reset spiked compartments to plateau level (not rest).
            # During refractory, leaky integration is disabled (active=False),
            # so voltage stays at plateau — sustained depolarization like real
            # dendritic calcium spikes (50-100ms plateau potentials).
            if _spiked is not None:
                plateau_v = rest + self._plateau_fraction * (thresh - rest)
                self.v_dendrite[c, _spiked] = plateau_v

        # Track compartment activity for STDP credit assignment
        # A compartment is "active" if it fired a dendritic spike OR if its
        # voltage is significantly above resting (even without a spike).
        for c in range(self.n_compartments):
            thresh = self._dend_spike_thresh[c]
            rest_c = cfg.resting[c]
            if thresh is not None:
                half = rest_c + (thresh - rest_c) * 0.5
                # Include dendritic spike — voltage was reset but compartment was active
                self.compartment_active_at_spike[c] = dend_spike_fired[c] | (
                    self.v_dendrite[c] > half
                )
            else:
                self.compartment_active_at_spike[c] = self.v_dendrite[c] > (rest_c + 2.0)

        # Clear input buffer for next step
        compartment_input[:] = 0.0

        return soma_current

    def step(self, input_current: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Advance one timestep. Returns boolean spike array.

        When dendrites are enabled, step_dendrites() is called first to compute
        dendritic contribution. input_current here is direct somatic injection
        (external current, sensory buffer, etc.).

        Args:
            input_current: External/direct somatic current injection (n,) array.
            dt: Timestep in ms.
        """
        p = self.params

        # Phase 0: GABA-B slow inhibitory current (CIP-21)
        # Decay conductance, then apply as hyperpolarizing current.
        # Applied BEFORE dendritic integration so it affects all compartments.
        if self._gaba_b_conductance is not None:
            self._gaba_b_conductance *= self._gaba_b_decay
            # I_gaba_b = g_b * (V - E_gaba_b); E_gaba_b = -90mV, V ~ -65mV
            # → negative current (hyperpolarizing) since V > E_gaba_b
            gaba_b_current = self._gaba_b_conductance * (self.v_membrane - self._gaba_b_reversal)
            input_current = input_current - gaba_b_current

        # Phase 0b: NMDA slow excitatory current (CIP-24)
        # Decay conductance, then compute voltage-dependent Mg2+ block and
        # apply as depolarizing current. The Mg2+ block creates positive
        # feedback: depolarized neurons get more NMDA current, sustaining
        # attractor states in working memory.
        if self._nmda_conductance is not None:
            self._nmda_conductance *= self._nmda_decay
            # B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-slope * V))
            mg_block = np.float32(1.0) / (
                np.float32(1.0)
                + (self._nmda_mg / np.float32(3.57))
                * np.exp(-self._nmda_slope * self.v_membrane, dtype=np.float32)
            )
            # I_nmda = g_nmda * B(V) * (E_nmda - V); E_nmda=0mV, V~-65mV → positive (depolarizing)
            nmda_current = self._nmda_conductance * mg_block * (self._nmda_e_rev - self.v_membrane)
            input_current = input_current + nmda_current

        # Phase 1: dendritic integration (if enabled)
        if self.v_dendrite is not None:
            dendrite_soma_current = self.step_dendrites(dt)
            input_current = input_current + dendrite_soma_current

        # Store pre-spike drive for k-WTA ranking (includes dendritic contribution)
        self._pre_spike_drive = input_current

        # Decrease refractory timers FIRST (before integration)
        np.maximum(
            self.refractory_timer - np.float32(dt), np.float32(0.0), out=self.refractory_timer
        )

        # Which neurons are NOT in refractory period
        active = self.refractory_timer <= 0.0

        # Leaky integration with per-neuron tau
        # dV/dt = (-(V - V_rest) + I) / tau
        dv = (-(self.v_membrane - p.resting) + input_current) * (dt / self._tau)

        # Add noise
        if p.noise_std > 0:
            dv += self._rng.normal(0, p.noise_std, size=self.n).astype(np.float32) * np.sqrt(
                dt / self._tau
            )

        # Only update active neurons
        self.v_membrane += dv * active

        # Spike Frequency Adaptation: decay adaptation offset, then detect with
        # adapted threshold.  Fires only if membrane >= (base_threshold + sfa_offset).
        self._sfa_offset *= self._sfa_decay

        # Detect spikes with per-neuron adapted thresholds
        self.spikes = (self.v_membrane >= (self._threshold + self._sfa_offset)) & active

        # Record spike times (store relative to _time_base for float32 precision)
        spike_indices = np.nonzero(self.spikes)[0]
        if len(spike_indices) > 0:
            self.last_spike_time[spike_indices] = np.float32(self._time - self._time_base)
            # SFA: raise threshold for neurons that just spiked
            self._sfa_offset[spike_indices] += self._sfa_increment

        # Reset spiking neurons and set per-neuron refractory
        self.v_membrane[self.spikes] = p.reset
        self.refractory_timer[self.spikes] = self._refractory_ms[self.spikes]

        # Advance clock
        self._time += dt

        # Periodically rebase time to maintain float32 precision.
        # At 1ms/step, float32 loses sub-ms precision after ~46 hours (~166M ms).
        # Rebase every ~11 hours (40M ms) to stay well within safe range.
        elapsed = self._time - self._time_base
        if elapsed > 40_000_000.0:
            # Shift all spike times so they remain correct relative to new base
            self.last_spike_time -= np.float32(elapsed)
            self._time_base = self._time

        return self.spikes

    @property
    def signed_spikes(self) -> np.ndarray:
        """Spikes with E/I sign: +1 for excitatory, -1 for inhibitory.

        When dual inhibition is enabled (CIP-21), SST neuron spikes are
        excluded from this property (they contribute zero). SST inhibition
        acts through slow GABA-B conductance instead of instant current.
        PV neuron spikes retain their -1 output as normal fast inhibition.
        """
        result = self.spikes.astype(np.float32) * self.output_sign
        if self._gaba_b_enabled:
            result[self.is_sst] = 0.0
        return result

    @property
    def sst_spikes(self) -> np.ndarray:
        """Boolean spike array for SST (slow inhibitory) neurons only.

        Used by network.py to route SST spikes through lateral synapses
        into the GABA-B conductance accumulator of post-synaptic neurons.
        Returns all-False when dual inhibition is disabled.
        """
        if not self._gaba_b_enabled:
            return np.zeros(self.n, dtype=bool)
        return self.spikes & self.is_sst

    def accumulate_gaba_b(self, sst_current: np.ndarray) -> None:
        """Accumulate slow GABA-B conductance from SST lateral synaptic input.

        Called by network.py after routing SST spikes through the lateral
        weight matrix. The magnitude of sst_current at each post-synaptic
        neuron increases gaba_b_conductance, which then decays with tau_gaba_b.

        Args:
            sst_current: Absolute value of post-synaptic current from SST spikes
                         routed through lateral weight matrix. Shape (n,).
        """
        if self._gaba_b_conductance is not None:
            self._gaba_b_conductance += np.abs(sst_current) * self._gaba_b_increment

    def accumulate_nmda(self, excitatory_current: np.ndarray) -> None:
        """Accumulate slow NMDA conductance from recurrent excitatory input.

        Called by network.py after routing WM recurrent spikes through the
        weight matrix. The magnitude of excitatory_current increases the
        NMDA conductance, which decays with tau_nmda (~100ms).

        The actual NMDA current is computed in step() with voltage-dependent
        Mg2+ block: I = g * B(V) * (E_rev - V).

        Args:
            excitatory_current: Post-synaptic current from recurrent spikes
                                routed through weight matrix. Shape (n,).
        """
        if self._nmda_conductance is not None:
            self._nmda_conductance += np.abs(excitatory_current) * self._nmda_scale

    def reset(self) -> None:
        """Reset all neurons to resting state."""
        self.v_membrane[:] = self.params.resting
        self.refractory_timer[:] = 0.0
        self.spikes[:] = False
        self.last_spike_time[:] = -1e6
        self._time = 0.0
        self._time_base = 0.0
        self._pre_spike_drive[:] = 0.0
        self._sfa_offset[:] = 0.0
        if self._gaba_b_conductance is not None:
            self._gaba_b_conductance[:] = 0.0
        if self._nmda_conductance is not None:
            self._nmda_conductance[:] = 0.0
        if self.v_dendrite is not None:
            cfg = self._dend_cfg
            for c in range(self.n_compartments):
                self.v_dendrite[c, :] = cfg.resting[c]
            self._dend_refractory[:] = 0.0
            self.compartment_active_at_spike[:] = False
            self._compartment_input[:] = 0.0
            self._dend_spike_fired[:] = False

    @property
    def time(self) -> float:
        return self._time

    def get_state(self, *, copy: bool = True) -> dict[str, np.ndarray]:
        """Serialize neuron state to dict of arrays.

        Args:
            copy: If True (default), returns copies (safe for concurrent
                  mutation).  If False, returns direct references (zero-copy).
        """
        _c = (lambda a: a.copy()) if copy else (lambda a: a)
        state = {
            "v_membrane": _c(self.v_membrane),
            "refractory_timer": _c(self.refractory_timer),
            "last_spike_time": _c(self.last_spike_time),
            "time": np.array([self._time], dtype=np.float64),
            "time_base": np.array([self._time_base], dtype=np.float64),
            "is_inhibitory": _c(self.is_inhibitory),
            "sfa_offset": _c(self._sfa_offset),
        }
        if self.v_dendrite is not None:
            state["v_dendrite"] = _c(self.v_dendrite)
            state["dend_refractory"] = _c(self._dend_refractory)
        if self._gaba_b_conductance is not None:
            state["gaba_b_conductance"] = _c(self._gaba_b_conductance)
            state["is_sst"] = _c(self.is_sst)
        if self._nmda_conductance is not None:
            state["nmda_conductance"] = _c(self._nmda_conductance)
        return state

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore neuron state from dict of arrays.

        Validates shapes — if config changed since save, logs a warning
        and re-initializes instead of crashing with a numpy shape error.
        """
        saved_n = len(state["v_membrane"])
        if saved_n != self.n:
            import logging

            logging.getLogger(__name__).warning(
                f"Neuron population size changed ({saved_n} → {self.n}), "
                "re-initializing instead of loading saved state"
            )
            self.reset()
            if "time" in state:
                self._time = float(state["time"][0])
            return

        self.v_membrane[:] = state["v_membrane"]
        self.refractory_timer[:] = state["refractory_timer"]
        self.last_spike_time[:] = state["last_spike_time"]
        self._time = float(state["time"][0])
        self._time_base = float(state["time_base"][0]) if "time_base" in state else 0.0
        self.spikes[:] = False
        if "is_inhibitory" in state:
            self.is_inhibitory[:] = state["is_inhibitory"]
            inh_scale = self._inh_cfg.inhibitory_weight_scale if self._inh_cfg else -1.0
            self.output_sign[:] = np.where(self.is_inhibitory, inh_scale, 1.0)

        # Restore SFA state (backward compat: old saves may not have it)
        if "sfa_offset" in state and len(state["sfa_offset"]) == self.n:
            self._sfa_offset[:] = state["sfa_offset"]

        # Restore GABA-B state (backward compat: old saves may not have it)
        if self._gaba_b_conductance is not None and "gaba_b_conductance" in state:
            if len(state["gaba_b_conductance"]) == self.n:
                self._gaba_b_conductance[:] = state["gaba_b_conductance"]
        if self._gaba_b_enabled and "is_sst" in state and len(state["is_sst"]) == self.n:
            self.is_sst[:] = state["is_sst"]

        # Restore NMDA state (backward compat: old saves may not have it)
        if self._nmda_conductance is not None and "nmda_conductance" in state:
            if len(state["nmda_conductance"]) == self.n:
                self._nmda_conductance[:] = state["nmda_conductance"]

        # Restore dendritic state
        if self.v_dendrite is not None and "v_dendrite" in state:
            saved = state["v_dendrite"]
            if saved.shape == self.v_dendrite.shape:
                self.v_dendrite[:] = saved
                if "dend_refractory" in state:
                    self._dend_refractory[:] = state["dend_refractory"]
            else:
                # Shape mismatch (compartment count changed) — reset dendrites
                import logging

                logging.getLogger(__name__).warning(
                    f"Dendrite shape mismatch ({saved.shape} → {self.v_dendrite.shape}), resetting"
                )
                for c in range(self.n_compartments):
                    self.v_dendrite[c, :] = self._dend_cfg.resting[c]
                self._dend_refractory[:] = 0.0
                self.compartment_active_at_spike[:] = False
