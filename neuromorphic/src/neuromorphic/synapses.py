"""Sparse synapse connections with STDP and R-STDP learning rules."""

from __future__ import annotations

import numpy as np
from scipy import sparse

import logging

from neuromorphic.config import (
    STDPParams, RSTDPParams, EligibilityTraceConfig, BCMConfig,
    PruningConfig, MyelinationConfig, NeighborhoodConsolidationConfig,
)

logger = logging.getLogger(__name__)


class SynapseGroup:
    """
    A group of synaptic connections between two neuron populations.

    Uses scipy.sparse CSR matrices for memory-efficient storage and
    fast matrix-vector multiplication.

    Convention: weights shape is (n_post, n_pre) so that
    current = weights @ pre_spikes gives post-synaptic current.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        sparsity: float,
        init_weight: float,
        plastic: bool = True,
        stdp_params: STDPParams | None = None,
        rstdp_params: RSTDPParams | None = None,
        rng: np.random.Generator | None = None,
        name: str = "",
        eligibility_config: EligibilityTraceConfig | None = None,
        target_compartment: int | None = None,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.sparsity = sparsity
        self.plastic = plastic
        self.stdp_params = stdp_params or STDPParams()
        self.rstdp_params = rstdp_params
        self.name = name
        self._rng = rng or np.random.default_rng()
        self.target_compartment = target_compartment  # which dendrite compartment to target (None = direct somatic)

        # Build sparse connectivity
        self.weights = self._build_sparse_weights(init_weight)
        self._cache_structure()

        # Eligibility traces — shadow array matching weights.data
        self._eligibility_cfg = eligibility_config
        self.eligibility: np.ndarray | None = None
        # Active eligibility index set: sorted int32 array of indices with nonzero traces.
        # Phase 17 only processes these entries instead of scanning all nnz.
        self._elig_active: np.ndarray | None = None
        # Adaptive significance threshold: fraction of a_plus, not hardcoded.
        # concept_lateral has a_plus=0.005 so threshold = 0.0005 (not 1e-5 = 500x too low).
        self._elig_significance: float = 1e-5  # default fallback
        self._base_elig_significance: float = 1e-5  # original value before dynamic boost
        self._elig_sig_boosted: bool = False  # True when significance doubled due to active set > 30%
        self._elig_prune_threshold: float = 1e-6  # for alive check in apply_neuromodulation
        self._elig_untracked_interval: int = 100
        self._elig_step_counter: int = 0  # counts apply_neuromodulation calls
        if eligibility_config is not None and plastic and self.weights.nnz > 0:
            self.eligibility = np.zeros(self.weights.nnz, dtype=np.float32)
            self._elig_active = np.empty(0, dtype=np.int32)
            a_plus = self.stdp_params.a_plus
            if self.rstdp_params is not None:
                a_plus = max(a_plus, self.rstdp_params.a_plus)
            self._elig_significance = a_plus * eligibility_config.significance_ratio
            self._base_elig_significance = self._elig_significance
            self._elig_prune_threshold = self._elig_significance * 0.01
            self._elig_untracked_interval = eligibility_config.untracked_decay_interval

        # BCM metaplasticity — per-postsynaptic-neuron modification threshold
        self._bcm_config: BCMConfig | None = None
        self.bcm_theta: np.ndarray | None = None

        # Adolescent phase arrays (lazily initialized when phase begins)
        # Per-synapse stability counter: steps with weight change < tolerance
        self.stability_counter: np.ndarray | None = None
        # Per-synapse boolean: True if myelinated (plasticity locked)
        self.myelinated: np.ndarray | None = None
        # Per-synapse boolean: True if identity-tagged (near-permanent)
        self.identity: np.ndarray | None = None
        # Per-synapse: rounds surviving pruning
        self.prune_survival_count: np.ndarray | None = None
        # Previous weights snapshot for stability tracking
        self._prev_weights: np.ndarray | None = None

        # Adolescent plasticity mask (set externally when phase is active)
        self._adolescent_plasticity_mask: np.ndarray | None = None

        # Convergence tracking — mean |dw| from last STDP update
        self.last_stdp_delta: float = 0.0

    def _cache_structure(self) -> None:
        """Cache COO row/col indices for in-place STDP updates.

        CSR .data is stored in row-major order. tocoo() produces indices
        in the same order, so coo.data[i] == csr.data[i]. We cache the
        row/col arrays once and reuse them for every STDP update, avoiding
        repeated O(nnz) allocations and CSR rebuilds.
        """
        if self.weights.nnz == 0:
            self._cached_rows = np.empty(0, dtype=np.int32)
            self._cached_cols = np.empty(0, dtype=np.int32)
            self._csc_valid = False
            return
        coo = self.weights.tocoo()
        self._cached_rows = coo.row.copy()
        self._cached_cols = coo.col.copy()
        # Invalidate CSC — will be rebuilt on next compute_current()
        self._csc_valid = False

    def _ensure_csc(self) -> None:
        """Rebuild CSC copy if invalid. CSC enables sparse column-gather SpMV."""
        if getattr(self, "_csc_valid", False):
            return
        if self.weights.nnz == 0:
            self._csc_indptr = np.zeros(self.n_pre + 1, dtype=np.int32)
            self._csc_indices = np.empty(0, dtype=np.int32)
            self._csc_data = np.empty(0, dtype=np.float32)
            self._csc_valid = True
            return
        csc = self.weights.tocsc()
        self._csc_indptr = csc.indptr.copy()
        self._csc_indices = csc.indices.copy()  # row indices
        self._csc_data = csc.data  # shares data with csc, but we'll rebuild from CSR
        # Build permutation: csc_data[i] corresponds to csr_data[perm[i]]
        # We need this to read current CSR weights through CSC structure
        # Since tocsc() reorders data, we build a mapping.
        # For each CSC entry at position k, find the CSR position:
        #   CSC entry k is at (row=csc_indices[k], col=j) where j is the column
        #   In CSR, that entry is at indptr[row] + offset where indices[offset]==col
        # Instead of this expensive lookup, just store a reference and rebuild
        # CSC data from CSR data on each use. The key insight: we only need
        # the CSC *structure* (indptr, indices) — actual weight values come from
        # CSR data via the permutation.
        #
        # Build perm: for each CSC data slot, which CSR data slot has the same value.
        # Since tocsc() is deterministic, we use argsort on (row, col) in both orderings.
        coo = self.weights.tocoo()
        # CSR order: row-major (same as COO from CSR)
        csr_key = coo.row.astype(np.int64) * self.n_pre + coo.col.astype(np.int64)
        csr_order = np.argsort(csr_key, kind="mergesort")
        # CSC order: col-major
        csc_key = coo.col.astype(np.int64) * self.n_post + coo.row.astype(np.int64)
        csc_order = np.argsort(csc_key, kind="mergesort")
        # perm[csc_position] = csr_position
        # csc_order[i] gives the COO index that goes to CSC position i
        # csr_order[i] gives the COO index that goes to CSR position i
        # We need: for CSC position i, what CSR position has the same (row,col)?
        # csc_order maps CSC-pos → COO-pos, inv_csr maps COO-pos → CSR-pos
        inv_csr = np.empty_like(csr_order)
        inv_csr[csr_order] = np.arange(len(csr_order), dtype=np.intp)
        self._csc_to_csr_perm = inv_csr[csc_order]
        self._csc_valid = True

    def _build_sparse_weights(self, init_weight: float) -> sparse.csr_matrix:
        """Create a random sparse weight matrix with target sparsity."""
        nnz = int(self.n_pre * self.n_post * self.sparsity)
        if nnz == 0:
            return sparse.csr_matrix((self.n_post, self.n_pre), dtype=np.float32)

        # Random row/col indices (with replacement is fine for large matrices)
        rows = self._rng.integers(0, self.n_post, size=nnz)
        cols = self._rng.integers(0, self.n_pre, size=nnz)
        data = np.full(nnz, init_weight, dtype=np.float32)

        # Small random variation around init weight
        data += self._rng.normal(0, init_weight * 0.1, size=nnz).astype(np.float32)
        # Non-plastic groups (e.g. brainstem arousal) may use init_weight > w_max.
        # Only enforce STDP bounds on plastic synapses.
        if self.plastic:
            data = np.clip(data, self.stdp_params.w_min, self.stdp_params.w_max)
        else:
            data = np.clip(data, 0.001, np.inf)

        mat = sparse.coo_matrix((data, (rows, cols)), shape=(self.n_post, self.n_pre))
        # Convert to CSR (sums duplicates), then clip to enforce weight bounds
        mat = mat.tocsr()
        if self.plastic:
            mat.data = np.clip(mat.data, self.stdp_params.w_min, self.stdp_params.w_max).astype(np.float32)
        else:
            mat.data = np.clip(mat.data, 0.001, np.inf).astype(np.float32)
        mat.eliminate_zeros()
        return mat

    def compute_current(
        self,
        pre_spikes: np.ndarray,
        pre_output_sign: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute post-synaptic current from pre-synaptic spikes.

        Uses CSC column-gather: only reads synapse columns for spiking
        pre-neurons. At ~5% firing rate, reads ~5% of nnz instead of 100%.

        Args:
            pre_spikes: boolean array (n_pre,) of which pre neurons spiked.
            pre_output_sign: optional float32 array (n_pre,) with +1/-1 per neuron
                for E/I balance. If None, all excitatory (+1).

        Returns:
            float32 array (n_post,) of input currents.
        """
        nnz = self.weights.nnz
        if nnz == 0:
            return np.zeros(self.n_post, dtype=np.float32)

        spiking = np.flatnonzero(pre_spikes)
        if len(spiking) == 0:
            return np.zeros(self.n_post, dtype=np.float32)

        # CSR SpMV via SciPy C code is faster for large matrices.
        # CSC column-gather only wins for very sparse small matrices.
        if nnz < 100_000 or len(spiking) > self.n_pre * 0.3 or nnz > 10_000_000:
            if pre_output_sign is not None:
                spike_vec = pre_spikes.astype(np.float32) * pre_output_sign
            else:
                spike_vec = pre_spikes.astype(np.float32)
            return np.asarray(self.weights @ spike_vec).flatten()

        # CSC sparse column-gather path
        self._ensure_csc()
        csc_indptr = self._csc_indptr
        csc_indices = self._csc_indices  # row (post-neuron) indices

        # Gather column ranges for spiking pre-neurons
        starts = csc_indptr[spiking]
        ends = csc_indptr[spiking + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(self.n_post, dtype=np.float32)

        # Build flat index array into CSC data
        nonempty = lengths > 0
        s_ne = starts[nonempty]
        l_ne = lengths[nonempty]
        spiking_ne = spiking[nonempty]

        base = np.repeat(s_ne, l_ne)
        grp_offsets = np.arange(total, dtype=np.intp)
        grp_cumlen = np.repeat(
            np.concatenate(([np.intp(0)], np.cumsum(l_ne[:-1]))),
            l_ne,
        )
        flat_csc_idx = base + (grp_offsets - grp_cumlen)

        # Map CSC positions back to CSR data positions for current weights
        flat_csr_idx = self._csc_to_csr_perm[flat_csc_idx]
        flat_weights = self.weights.data[flat_csr_idx]

        # Post-neuron (row) indices for each gathered entry
        flat_rows = csc_indices[flat_csc_idx]

        # Apply E/I sign per spiking pre-neuron
        if pre_output_sign is not None:
            flat_sign = np.repeat(pre_output_sign[spiking_ne], l_ne)
            flat_weights = flat_weights * flat_sign
        # else: all excitatory, flat_weights unchanged

        # Accumulate into result using bincount (faster than np.add.at)
        return np.bincount(
            flat_rows, weights=flat_weights, minlength=self.n_post,
        ).astype(np.float32)

    def _expand_ranges(self, starts, lengths):
        """Vectorized expansion of multiple [start, start+length) ranges into flat indices."""
        nonempty = lengths > 0
        if not nonempty.any():
            return np.empty(0, dtype=np.intp)
        s_ne = starts[nonempty]
        l_ne = lengths[nonempty]
        total = int(l_ne.sum())
        base = np.repeat(s_ne, l_ne)
        grp_offsets = np.arange(total, dtype=np.intp)
        grp_cumlen = np.repeat(
            np.concatenate(([np.intp(0)], np.cumsum(l_ne[:-1]))),
            l_ne,
        )
        return base + (grp_offsets - grp_cumlen)

    def _gather_active_synapses(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> np.ndarray:
        """Gather CSR data indices of synapses where pre OR post neuron spiked.

        Post-neuron path uses CSR indptr (row ranges).
        Pre-neuron path uses CSC indptr (column ranges) + perm mapping to
        CSR positions. This avoids scanning all nnz column indices.

        At ~5% firing rate this visits ~10% of synapses instead of 100%.
        Returns array of data indices into weights.data / cached COO.
        """
        active_post = np.flatnonzero(post_spikes)
        active_pre = np.flatnonzero(pre_spikes)

        if len(active_post) == 0 and len(active_pre) == 0:
            return np.empty(0, dtype=np.intp)

        nnz = self.weights.nnz
        if nnz == 0:
            return np.empty(0, dtype=np.intp)

        # For small matrices, fall back to simple boolean mask (less overhead)
        # For large matrices (>10M nnz), CSC overhead outweighs gains
        if nnz < 100_000 or nnz > 10_000_000:
            mask = np.zeros(nnz, dtype=np.bool_)
            indptr = self.weights.indptr
            if len(active_post) > 0:
                flat_idx = self._expand_ranges(
                    indptr[active_post], indptr[active_post + 1] - indptr[active_post])
                if len(flat_idx) > 0:
                    mask[flat_idx] = True
            if len(active_pre) > 0:
                pre_mask = np.zeros(self.n_pre, dtype=np.bool_)
                pre_mask[active_pre] = True
                mask |= pre_mask[self.weights.indices]
            return np.flatnonzero(mask)

        # Large matrix path: use CSR for post, CSC for pre
        csr_indices_post = np.empty(0, dtype=np.intp)
        csr_indices_pre = np.empty(0, dtype=np.intp)

        # Post-neuron path: CSR row ranges
        if len(active_post) > 0:
            indptr = self.weights.indptr
            csr_indices_post = self._expand_ranges(
                indptr[active_post], indptr[active_post + 1] - indptr[active_post])

        # Pre-neuron path: CSC column ranges → map to CSR positions
        if len(active_pre) > 0:
            self._ensure_csc()
            csc_indptr = self._csc_indptr
            csc_flat = self._expand_ranges(
                csc_indptr[active_pre], csc_indptr[active_pre + 1] - csc_indptr[active_pre])
            if len(csc_flat) > 0:
                # Map CSC positions to CSR data positions
                csr_indices_pre = self._csc_to_csr_perm[csc_flat]

        # Union of both sets (deduplicated)
        if len(csr_indices_post) == 0:
            return np.unique(csr_indices_pre)
        if len(csr_indices_pre) == 0:
            return np.unique(csr_indices_post)
        combined = np.concatenate([csr_indices_post, csr_indices_pre])
        return np.unique(combined)

    def update_weights_stdp(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray,
        current_time: float,
        compartment_activity: float = 1.0,
    ) -> None:
        """
        Apply STDP weight update in-place on existing CSR data.

        Uses sparse spike gathering via CSR indptr to avoid scanning all nnz
        entries. Only visits synapses connected to neurons that actually spiked.
        """
        if not self.plastic:
            return

        p = self.stdp_params
        rows = self._cached_rows
        cols = self._cached_cols
        data = self.weights.data

        if len(data) == 0:
            return

        # Guard: cached indices must match current CSR data length.
        # If this fires, weights were replaced without calling _cache_structure().
        if len(rows) != len(data):
            self._cache_structure()
            rows = self._cached_rows
            cols = self._cached_cols

        # Sparse spike gathering — only visit synapses connected to active neurons
        active_idx = self._gather_active_synapses(pre_spikes, post_spikes)
        if len(active_idx) == 0:
            return

        # Time differences
        dt_spike = post_spike_times[rows[active_idx]] - pre_spike_times[cols[active_idx]]

        # H8: Spike-rate filter — ignore pairs outside biologically meaningful window
        abs_dt = np.abs(dt_spike)
        in_window = (abs_dt >= p.min_dt) & (abs_dt <= p.max_dt)
        if not in_window.any():
            self.last_stdp_delta = 0.0
            return
        active_idx = active_idx[in_window]
        dt_spike = dt_spike[in_window]

        # LTP: post fires after pre (dt > 0), or simultaneous (dt == 0, Hebbian coincidence)
        ltp_mask = dt_spike >= 0
        dw = np.zeros(len(active_idx), dtype=np.float32)

        # BCM scaling: modulate A_plus AND A_minus per postsynaptic neuron
        bcm_scale = self._get_bcm_scaling()
        if bcm_scale is not None and ltp_mask.any():
            # Per-synapse scaling based on postsynaptic neuron's BCM threshold
            a_plus_vec = np.float32(p.a_plus) * bcm_scale[rows[active_idx[ltp_mask]]]
            dw[ltp_mask] = a_plus_vec * np.exp(-dt_spike[ltp_mask] / p.tau_plus).astype(np.float32)
        else:
            dw[ltp_mask] = p.a_plus * np.exp(-dt_spike[ltp_mask] / p.tau_plus).astype(np.float32)

        # LTD: pre fires after post (dt < 0)
        # H3: BCM also modulates LTD — active neurons resist BOTH potentiation and depression
        ltd_mask = dt_spike < 0
        if bcm_scale is not None and ltd_mask.any():
            # Inverse BCM for LTD: active neurons have reduced depression
            a_minus_vec = np.float32(p.a_minus) * bcm_scale[rows[active_idx[ltd_mask]]]
            dw[ltd_mask] = -a_minus_vec * np.exp(dt_spike[ltd_mask] / p.tau_minus).astype(np.float32)
        else:
            dw[ltd_mask] = -p.a_minus * np.exp(dt_spike[ltd_mask] / p.tau_minus).astype(np.float32)

        # Compartment-aware credit assignment: scale dw by compartment activity
        if compartment_activity < 1.0:
            dw *= np.float32(compartment_activity)

        # Track mean absolute weight change for convergence detection
        self.last_stdp_delta = float(np.abs(dw).mean()) if len(dw) > 0 else 0.0

        if self.eligibility is not None:
            # Three-factor: STDP updates eligibility traces, not weights directly
            # (plasticity mask is applied later in apply_neuromodulation, not here)
            self.eligibility[active_idx] += dw
            # D2: Cap eligibility traces — only clip modified entries, not all nnz
            self.eligibility[active_idx] = np.clip(
                self.eligibility[active_idx], -p.w_max, p.w_max)
            # Merge only significant updates into active eligibility index set.
            # Filter by |dw| > adaptive threshold to prevent the active set from
            # bloating with trivially small eligibility values that don't affect weights.
            significant = np.abs(dw) > self._elig_significance
            if significant.any():
                self._merge_elig_active(active_idx[significant])
        else:
            # Classic two-factor: STDP directly updates weights
            # H2: Apply plasticity mask in two-factor path (three-factor uses apply_neuromodulation)
            if self._adolescent_plasticity_mask is not None and len(self._adolescent_plasticity_mask) == len(data):
                dw *= self._adolescent_plasticity_mask[active_idx]
            data[active_idx] = np.clip(data[active_idx] + dw, p.w_min, p.w_max)

        # Only eliminate zeros if w_min allows zero weights
        if p.w_min <= 0.0 and self.eligibility is None:
            self.weights.eliminate_zeros()
            self._cache_structure()

    def update_weights_rstdp(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray,
        current_time: float,
        modulation: float,
        compartment_activity: float = 1.0,
    ) -> None:
        """
        Apply reward-modulated STDP (R-STDP) in-place on existing CSR data.

        Same as STDP but weight changes are scaled by a modulation signal:
        - modulation ~1.0: baseline (neutral)
        - modulation < 1.0: suppress learning (prediction matched)
        - modulation > 1.0: amplify learning (prediction error / surprise)
        """
        if not self.plastic or self.rstdp_params is None:
            return

        p = self.rstdp_params
        rows = self._cached_rows
        cols = self._cached_cols
        data = self.weights.data

        if len(data) == 0:
            return

        if len(rows) != len(data):
            self._cache_structure()
            rows = self._cached_rows
            cols = self._cached_cols

        # Sparse spike gathering — only visit synapses connected to active neurons
        active_idx = self._gather_active_synapses(pre_spikes, post_spikes)
        if len(active_idx) == 0:
            return

        dt_spike = post_spike_times[rows[active_idx]] - pre_spike_times[cols[active_idx]]

        # LTP: post fires after pre or simultaneous (Hebbian coincidence)
        ltp_mask = dt_spike >= 0
        dw = np.zeros(len(active_idx), dtype=np.float32)
        dw[ltp_mask] = p.a_plus * np.exp(-dt_spike[ltp_mask] / p.tau_plus).astype(np.float32)

        ltd_mask = dt_spike < 0
        dw[ltd_mask] = -p.a_minus * np.exp(dt_spike[ltd_mask] / p.tau_minus).astype(np.float32)

        # Modulate by reward signal
        dw *= np.float32(modulation)

        # Compartment-aware credit assignment (before surprise bonus, which is global)
        if compartment_activity < 1.0:
            dw *= np.float32(compartment_activity)

        # H4: Surprise bonus — correct predictions strengthen the pathway
        # Applied after compartment scaling: surprise is a global neuromodulatory
        # signal, not gated by local compartment activity
        if modulation > p.modulation_mismatch:
            dw += np.float32(p.surprise_bonus)

        # Track mean absolute weight change for convergence detection
        self.last_stdp_delta = float(np.abs(dw).mean()) if len(dw) > 0 else 0.0

        if self.eligibility is not None:
            # Three-factor: R-STDP also goes into eligibility traces
            # (plasticity mask is applied later in apply_neuromodulation, not here)
            self.eligibility[active_idx] += dw
            # D2: Cap eligibility traces — only clip modified entries, not all nnz
            self.eligibility[active_idx] = np.clip(
                self.eligibility[active_idx], -p.w_max, p.w_max)
            # Merge only significant updates into active eligibility index set
            significant = np.abs(dw) > self._elig_significance
            if significant.any():
                self._merge_elig_active(active_idx[significant])
        else:
            # Two-factor: directly update weights
            # H2: Apply plasticity mask in two-factor path only
            if self._adolescent_plasticity_mask is not None and len(self._adolescent_plasticity_mask) == len(data):
                dw *= self._adolescent_plasticity_mask[active_idx]
            data[active_idx] = np.clip(data[active_idx] + dw, p.w_min, p.w_max)

        # Only eliminate zeros if w_min allows zero weights
        if p.w_min <= 0.0 and self.eligibility is None:
            self.weights.eliminate_zeros()
            self._cache_structure()

    def enable_bcm(self, bcm_config: BCMConfig) -> None:
        """Enable BCM metaplasticity for this synapse group."""
        if not self.plastic:
            return
        self._bcm_config = bcm_config
        self.bcm_theta = np.full(self.n_post, bcm_config.theta_init, dtype=np.float32)

    def update_bcm_threshold(self, post_firing_rates: np.ndarray) -> None:
        """Update per-neuron BCM modification threshold from recent activity.

        theta_m = EMA[post_rate^2] — active neurons get higher threshold.
        """
        if self.bcm_theta is None or self._bcm_config is None:
            return
        alpha = 1.0 / self._bcm_config.theta_tau
        rate_sq = post_firing_rates ** 2
        self.bcm_theta += np.float32(alpha) * (rate_sq - self.bcm_theta)

    def _get_bcm_scaling(self) -> np.ndarray | None:
        """Get per-postsynaptic-neuron scaling from BCM threshold.

        Patent Claim 1c: BCM scales BOTH a_plus AND a_minus per postsynaptic neuron.
        Active neurons (high theta) are harder to potentiate/depress.

        Formula: scaling = 2 * theta_init / (theta + theta_init)
          theta = theta_init → scaling = 1.0  (baseline: normal STDP)
          theta = 2x init   → scaling ≈ 0.67 (active: reduced plasticity)
          theta = 4x init   → scaling = 0.4  (very active: strongly reduced)
          theta → 0         → scaling → 2.0  (quiet: enhanced plasticity)
        """
        if self.bcm_theta is None or self._bcm_config is None:
            return None
        theta_init = np.float32(self._bcm_config.theta_init)
        return 2.0 * theta_init / (self.bcm_theta + theta_init + np.float32(1e-8))

    def apply_neuromodulation(
        self,
        modulator_signal: float,
        plasticity_mask: np.ndarray | None = None,
    ) -> None:
        """
        Apply neuromodulatory signal to eligibility traces → weight updates.

        This is the third factor in three-factor learning:
        dw = M(t) * e(t) * mask  where M is the neuromodulator, e is eligibility,
        and mask is per-synapse plasticity scaling (1.0/0.1/0.01 for normal/myelinated/identity).
        """
        if self.eligibility is None or not self.plastic:
            return

        p = self.stdp_params
        data = self.weights.data
        if len(data) == 0:
            return

        # Weight change = modulator * eligibility * per-synapse mask
        dw = np.float32(modulator_signal) * self.eligibility
        if plasticity_mask is not None and len(plasticity_mask) == len(dw):
            dw *= plasticity_mask
        data[:] = np.clip(data + dw, p.w_min, p.w_max)

    def decay_eligibility(self) -> None:
        """Decay eligibility traces by one step."""
        if self.eligibility is None:
            return
        self.eligibility *= np.float32(self._eligibility_cfg.trace_decay)

    def _merge_elig_active(self, new_idx: np.ndarray) -> None:
        """Merge new active indices into the sorted _elig_active set.

        Uses searchsorted for O(n+m) sorted merge instead of O((n+m) log(n+m))
        union1d. At 1B+ synapses with ~30% active, this saves hundreds of ms.

        Identity-tagged synapses are NOT filtered at merge time -- Patent
        Claim 1 ("Never disable any mechanism") and Claim 2 ("identity
        tagging plasticity->1%") require three-factor learning to remain
        active at 1% strength.  The plasticity_mask in
        apply_neuromodulation_and_decay handles the 0.01 scaling.  If
        identity entries push the active set past 80% of nnz, the
        full-array fallback (faster than fancy indexing) kicks in
        automatically.
        """
        if self._elig_active is None:
            return
        if len(new_idx) == 0:
            return
        new_int32 = np.unique(new_idx.astype(np.int32))

        if len(self._elig_active) == 0:
            self._elig_active = new_int32
            return
        # If active set already covers >80% of nnz, abandon tracking --
        # full-array ops will be cheaper than fancy indexing overhead.
        # Dynamic significance: when active set > 30% of nnz, double the
        # significance threshold to slow growth and prevent 80% fallback.
        nnz = self.weights.nnz
        if nnz > 0 and len(self._elig_active) > nnz * 0.8:
            self._elig_active = None
            return
        frac = len(self._elig_active) / nnz if nnz > 0 else 0.0
        if frac > 0.3:
            if not self._elig_sig_boosted:
                self._elig_significance *= 2.0
                self._elig_sig_boosted = True
        elif frac < 0.10 and self._elig_sig_boosted:
            self._elig_significance = self._base_elig_significance
            self._elig_sig_boosted = False

        # Sorted merge: find which new entries are not already in active set
        positions = np.searchsorted(self._elig_active, new_int32)
        # Clamp positions to valid range for comparison
        clamped = np.minimum(positions, len(self._elig_active) - 1)
        novel = self._elig_active[clamped] != new_int32
        if not novel.any():
            return  # all already present
        # Insert novel entries in sorted position
        to_insert = new_int32[novel]
        self._elig_active = np.sort(np.concatenate([self._elig_active, to_insert]))

    def apply_neuromodulation_and_decay(
        self,
        modulator_signal: float,
        plasticity_mask: np.ndarray | None = None,
        interval: int = 1,
    ) -> None:
        """Fused neuromodulation + decay operating on active eligibility set.

        When _elig_active is available (sparse tracking), only processes those
        entries. When _elig_active is None (>80% active, abandoned sparse
        tracking), operates on the full contiguous array which is faster than
        fancy indexing at that density.

        Args:
            interval: Number of steps being batched.  Decay is compensated:
                ``decay^interval`` instead of ``decay^1``.  Modulator signal
                should be pre-averaged over the interval by the caller.

        After decay, entries that fall below epsilon are removed from the active set.
        """
        if self.eligibility is None or not self.plastic:
            return

        p = self.stdp_params
        data = self.weights.data
        if len(data) == 0:
            return

        # Compensated decay: decay^interval is mathematically equivalent to
        # applying decay once per step for `interval` steps.
        decay = np.float32(self._eligibility_cfg.trace_decay ** interval)

        # Full-array path: _elig_active abandoned (>80% active) or never initialized
        if self._elig_active is None:
            dw = np.float32(modulator_signal) * self.eligibility
            if plasticity_mask is not None and len(plasticity_mask) == len(self.eligibility):
                dw *= plasticity_mask
            data[:] = np.clip(data + dw, p.w_min, p.w_max)
            self.eligibility *= decay
            return

        # Periodic sweep: must run even when _elig_active is empty, to clean
        # stale traces left over from before the adaptive threshold fix.
        # Moved BEFORE the empty-active-set early return so old saturated
        # eligibility entries get zeroed out instead of lingering forever.
        self._elig_step_counter += 1
        if self._elig_step_counter >= self._elig_untracked_interval:
            self._elig_step_counter = 0
            if self._elig_active is not None and len(self._elig_active) < self.weights.nnz:
                # Build mask of all untracked entries with nonzero eligibility
                all_nonzero = np.flatnonzero(np.abs(self.eligibility) > self._elig_prune_threshold)
                if len(all_nonzero) > len(self._elig_active):
                    # Untracked entries exist -- zero them out (they've been decaying
                    # without neuromodulation application, so they're stale)
                    tracked_set = set(self._elig_active.tolist()) if len(self._elig_active) < 100000 else None
                    if tracked_set is not None:
                        untracked = np.array([i for i in all_nonzero if i not in tracked_set], dtype=np.int32)
                    else:
                        # For large active sets, use sorted merge to find difference
                        mask = np.ones(len(all_nonzero), dtype=bool)
                        positions = np.searchsorted(self._elig_active, all_nonzero)
                        valid = positions < len(self._elig_active)
                        mask[valid] = self._elig_active[positions[valid]] != all_nonzero[valid]
                        untracked = all_nonzero[mask]
                    if len(untracked) > 0:
                        self.eligibility[untracked] = 0.0

        if len(self._elig_active) == 0:
            return

        idx = self._elig_active

        # Identity-tagged synapses are NOT filtered here -- Patent Claim 1
        # ("Never disable any mechanism") requires three-factor learning to
        # remain active.  The plasticity_mask (passed from network.py via
        # get_adolescent_plasticity_mask) scales identity synapses to 0.01,
        # satisfying Claim 2's "plasticity->1%" requirement.

        elig_slice = self.eligibility[idx]

        # Apply neuromodulation: dw = modulator * eligibility * mask
        dw = np.float32(modulator_signal) * elig_slice
        if plasticity_mask is not None and len(plasticity_mask) == len(self.eligibility):
            dw *= plasticity_mask[idx]
        data[idx] = np.clip(data[idx] + dw, p.w_min, p.w_max)

        # Decay eligibility traces (compensated for interval)
        elig_slice *= decay
        self.eligibility[idx] = elig_slice

        # Prune near-zero entries from active set
        alive = np.abs(elig_slice) > self._elig_prune_threshold
        if not alive.all():
            self._elig_active = idx[alive]

    def normalize_weights(
        self,
        target_frac: float = 0.5,
        base_rate: float = 0.01,
        plasticity_multiplier: float = 1.0,
        adolescent_bypass: bool = False,
        plasticity_mask: np.ndarray | None = None,
    ) -> None:
        """
        Synaptic scaling homeostasis -- per-postsynaptic-neuron normalization.

        For each postsynaptic neuron, scale all incoming weights so their
        sum stays near a target. This creates competition: if one synapse
        strengthens via STDP, others must weaken, preventing saturation
        while preserving learned relative differences.

        Uses multiplicative scaling (biological synaptic scaling) rather
        than divisive normalization, applying a soft pull toward the target
        sum each call. Fully vectorized using np.add.reduceat.

        H1: scaling_rate is inversely proportional to plasticity_multiplier --
        during high-plasticity phases (infant), homeostasis is gentler
        so STDP can "win". During mature phase, homeostasis strengthens.

        H2 (adolescent bypass): During adolescent phase, the widened STDP
        windows (1.5x a_plus) create strong net-positive weight pressure.
        When adolescent_bypass=True, skip the inverse-plasticity reduction
        and use base_rate directly so homeostasis can counterbalance STDP.
        """
        if not self.plastic or self.weights.nnz == 0:
            return

        p = self.stdp_params
        data = self.weights.data
        indptr = self.weights.indptr

        target_mean = np.float32(target_frac * p.w_max)
        if adolescent_bypass:
            # H2: During adolescent phase, use base_rate directly.
            # The widened STDP windows need stronger homeostatic counterbalance.
            scaling_rate = np.float32(min(base_rate, 0.05))
        else:
            # H1: Inverse scaling for non-adolescent phases.
            effective_rate = base_rate / max(plasticity_multiplier, 0.1)
            scaling_rate = np.float32(min(effective_rate, 0.05))

        # Per-row fan-in counts
        fan_in = np.diff(indptr)

        # Find rows with nonzero fan-in
        nonempty = fan_in > 0
        if not nonempty.any():
            return

        # Row-wise sums using reduceat (vectorized, no Python loop)
        # reduceat needs the start indices of each nonempty row
        starts = indptr[:-1][nonempty]
        row_sums = np.add.reduceat(data, starts)

        # Target sum per row
        target_sums = target_mean * fan_in[nonempty].astype(np.float32)

        # Per-row scale factor (skip rows with non-positive sums)
        valid = row_sums > 0
        scale_per_row = np.ones(len(starts), dtype=np.float32)
        scale_per_row[valid] = np.float32(1.0) + scaling_rate * (
            target_sums[valid] / row_sums[valid] - np.float32(1.0)
        )

        # Broadcast per-row scale to per-synapse using np.repeat
        # Build full scale array (all rows including empty ones)
        counts = fan_in[nonempty]
        per_synapse_scale = np.repeat(scale_per_row, counts)

        # Myelination-aware: myelinated/identity synapses resist homeostatic
        # correction.  Without this, homeostasis treats all synapses equally,
        # so unmyelinated minority can't compensate for the locked majority,
        # causing mean weight to keep climbing toward the emergency ceiling.
        if plasticity_mask is not None and len(plasticity_mask) == len(data):
            correction = per_synapse_scale - np.float32(1.0)
            correction *= plasticity_mask
            per_synapse_scale = np.float32(1.0) + correction

        # Apply multiplicative scaling
        data *= per_synapse_scale

        # Enforce hard bounds
        np.clip(data, p.w_min, p.w_max, out=data)

    # ==================================================================
    # Adolescent phase mechanisms
    # ==================================================================

    def init_adolescent_arrays(self) -> None:
        """Initialize per-synapse arrays for adolescent phase mechanisms."""
        if not self.plastic or self.weights.nnz == 0:
            return
        nnz = self.weights.nnz
        if self.stability_counter is None:
            self.stability_counter = np.zeros(nnz, dtype=np.int32)
        if self.myelinated is None:
            self.myelinated = np.zeros(nnz, dtype=np.bool_)
        if self.identity is None:
            self.identity = np.zeros(nnz, dtype=np.bool_)
        if self.prune_survival_count is None:
            self.prune_survival_count = np.zeros(nnz, dtype=np.int32)
        self._prev_weights = self.weights.data.copy()

    def update_stability(self, tolerance: float = 0.02) -> None:
        """Track per-synapse weight stability (steps with < tolerance change)."""
        if self._prev_weights is None or self.stability_counter is None:
            return
        data = self.weights.data
        if len(data) != len(self._prev_weights):
            # Topology changed (e.g., after pruning), re-init
            self._prev_weights = data.copy()
            self.stability_counter = np.zeros(len(data), dtype=np.int32)
            return
        delta = np.abs(data - self._prev_weights)
        stable = delta < tolerance
        self.stability_counter[stable] += 1
        self.stability_counter[~stable] = 0
        self._prev_weights = data.copy()

    def prune(self, config: PruningConfig) -> int:
        """Remove weak synapses. Returns count of pruned connections."""
        if not self.plastic or self.weights.nnz == 0:
            return 0

        data = self.weights.data
        # Never prune myelinated or identity synapses
        prune_eligible = data < config.weight_threshold
        if self.myelinated is not None:
            prune_eligible &= ~self.myelinated
        if self.identity is not None:
            prune_eligible &= ~self.identity

        n_eligible = int(prune_eligible.sum())
        if n_eligible == 0:
            return 0

        max_prune = int(len(data) * config.max_prune_fraction)
        n_to_prune = min(n_eligible, max(1, max_prune))

        if n_to_prune == 0:
            return 0

        # Select weakest among eligible
        eligible_idx = np.nonzero(prune_eligible)[0]
        eligible_weights = data[eligible_idx]
        if n_to_prune >= len(eligible_weights):
            # Prune all eligible
            weakest = np.arange(len(eligible_weights))
        else:
            weakest = np.argpartition(eligible_weights, n_to_prune)[:n_to_prune]
        prune_idx = eligible_idx[weakest]

        # Zero out pruned synapses
        data[prune_idx] = 0.0

        # Update survival count: only eligible synapses that survived (faced real risk)
        if self.prune_survival_count is not None:
            survived_risk = prune_eligible.copy()
            survived_risk[prune_idx] = False  # these were pruned, not survivors
            self.prune_survival_count[survived_risk] += 1

        # Eliminate zeros and rebuild structure
        self.weights.eliminate_zeros()
        self._cache_structure()

        # Resize adolescent arrays to match new nnz
        new_nnz = self.weights.nnz
        self._resize_adolescent_arrays(new_nnz, prune_idx)

        return n_to_prune

    def _resize_adolescent_arrays(self, new_nnz: int, removed_idx: np.ndarray) -> None:
        """Resize per-synapse adolescent arrays after pruning removes entries.

        NOTE: This assumes eliminate_zeros() only removed the synapses we zeroed
        in prune() — no pre-existing zeros.  This holds when w_min > 0 (production
        uses w_min=0.01).  If w_min ever becomes 0, the old_nnz calculation would
        be wrong and arrays would misalign.
        """
        # Build keep mask
        old_nnz = new_nnz + len(removed_idx)
        keep_mask = np.ones(old_nnz, dtype=np.bool_)
        keep_mask[removed_idx] = False

        if self.stability_counter is not None and len(self.stability_counter) == old_nnz:
            self.stability_counter = self.stability_counter[keep_mask]
        elif self.stability_counter is not None:
            self.stability_counter = np.zeros(new_nnz, dtype=np.int32)

        if self.myelinated is not None and len(self.myelinated) == old_nnz:
            self.myelinated = self.myelinated[keep_mask]
        elif self.myelinated is not None:
            self.myelinated = np.zeros(new_nnz, dtype=np.bool_)

        if self.identity is not None and len(self.identity) == old_nnz:
            self.identity = self.identity[keep_mask]
        elif self.identity is not None:
            self.identity = np.zeros(new_nnz, dtype=np.bool_)

        if self.prune_survival_count is not None and len(self.prune_survival_count) == old_nnz:
            self.prune_survival_count = self.prune_survival_count[keep_mask]
        elif self.prune_survival_count is not None:
            self.prune_survival_count = np.zeros(new_nnz, dtype=np.int32)

        if self.eligibility is not None and len(self.eligibility) == old_nnz:
            self.eligibility = self.eligibility[keep_mask]
        elif self.eligibility is not None:
            self.eligibility = np.zeros(new_nnz, dtype=np.float32)

        # Rebuild _elig_active: remap old indices through keep_mask
        if self._elig_active is not None:
            if len(self._elig_active) > 0:
                # Build old→new index mapping
                new_indices = np.cumsum(keep_mask) - 1  # new position for each old position
                # Filter to entries that survived pruning
                survived = keep_mask[self._elig_active]
                self._elig_active = new_indices[self._elig_active[survived]].astype(np.int32)
            # else: already empty, nothing to remap

        self._prev_weights = self.weights.data.copy() if self._prev_weights is not None else None

    def myelinate(self, config: MyelinationConfig) -> int:
        """Mark stable, high-weight synapses as myelinated. Returns newly myelinated count."""
        if not self.plastic or self.stability_counter is None or self.myelinated is None:
            return 0

        data = self.weights.data
        if len(data) == 0:
            return 0

        eligible = (
            (self.stability_counter >= config.stability_window)
            & (data >= config.weight_threshold)
            & (~self.myelinated)
        )
        n_new = int(eligible.sum())
        if n_new > 0:
            self.myelinated[eligible] = True
        return n_new

    def tag_identity(self, min_survival_rounds: int = 3, stability_multiplier: int = 2,
                     stability_window: int = 5000) -> int:
        """Tag synapses as identity via two paths:

        Path 1 (original): myelinated + survived N pruning rounds
        Path 2 (C5 alt):   myelinated + extremely stable (stability > N × stability_window)

        This ensures stable high-weight synapses that were never weak enough
        to face pruning can still become identity-tagged.
        """
        if (self.myelinated is None or self.identity is None or
                self.prune_survival_count is None):
            return 0

        # Path 1: pruning survival
        path1 = (
            self.myelinated
            & (self.prune_survival_count >= min_survival_rounds)
            & (~self.identity)
        )

        # Path 2: extreme stability (C5)
        path2 = np.zeros_like(self.identity)
        if self.stability_counter is not None:
            alt_threshold = stability_multiplier * stability_window
            path2 = (
                self.myelinated
                & (self.stability_counter >= alt_threshold)
                & (~self.identity)
            )

        eligible = path1 | path2
        n_new = int(eligible.sum())
        if n_new > 0:
            self.identity[eligible] = True
        return n_new

    @property
    def myelination_fraction(self) -> float:
        """Fraction of synapses that are myelinated."""
        if self.myelinated is None or len(self.myelinated) == 0:
            return 0.0
        return float(self.myelinated.sum()) / len(self.myelinated)

    @property
    def identity_fraction(self) -> float:
        """Fraction of synapses tagged as identity."""
        if self.identity is None or len(self.identity) == 0:
            return 0.0
        return float(self.identity.sum()) / len(self.identity)

    def get_adolescent_plasticity_mask(self, config: MyelinationConfig) -> np.ndarray:
        """Return per-synapse plasticity scaling for adolescent phase.

        Normal synapses: 1.0
        Myelinated: config.plasticity_reduction (default 0.1)
        Identity: config.identity_plasticity_reduction (default 0.01)
        """
        n = self.weights.nnz
        if n == 0:
            return np.ones(0, dtype=np.float32)
        mask = np.ones(n, dtype=np.float32)
        if self.myelinated is not None:
            if len(self.myelinated) == n:
                mask[self.myelinated] = config.plasticity_reduction
            else:
                logger.warning(
                    f"Synapse '{self.name}' myelinated array length {len(self.myelinated)} "
                    f"!= nnz {n} — skipping myelination mask"
                )
        if self.identity is not None:
            if len(self.identity) == n:
                mask[self.identity] = config.identity_plasticity_reduction
            else:
                logger.warning(
                    f"Synapse '{self.name}' identity array length {len(self.identity)} "
                    f"!= nnz {n} — skipping identity mask"
                )
        return mask

    def apply_neighborhood_consolidation(
        self,
        da_level: float,
        config: NeighborhoodConsolidationConfig,
        current_time: float,
        spike_times_pre: np.ndarray,
        spike_times_post: np.ndarray,
    ) -> None:
        """When DA bursts, rescue nearby eligibility traces (synaptic tagging & capture)."""
        if self.eligibility is None or da_level < config.da_burst_threshold:
            return

        rows = self._cached_rows
        cols = self._cached_cols
        if len(rows) == 0 or len(rows) != len(self.eligibility):
            return

        # Find synapses with recent pre/post spikes within rescue window
        pre_recent = (current_time - spike_times_pre[cols]) < config.rescue_radius_ms
        post_recent = (current_time - spike_times_post[rows]) < config.rescue_radius_ms
        nearby = pre_recent | post_recent

        # Boost eligibility of nearby synapses (H5: ensure minimum rescue level)
        if nearby.any():
            orig = self.eligibility[nearby].copy()
            boosted = orig * (1.0 + config.rescue_strength)
            # Rescue floor: only for traces that were nonzero before boosting
            # (prevents injecting signal into synapses that never had STDP activity)
            floor = np.float32(config.rescue_floor)
            was_nonzero = np.abs(orig) > 0
            too_small = was_nonzero & (np.abs(boosted) < floor)
            boosted[too_small] = np.where(
                boosted[too_small] >= 0, floor, -floor
            )
            self.eligibility[nearby] = boosted
            # Track newly active eligibility entries
            self._merge_elig_active(np.flatnonzero(nearby))

    @property
    def nnz(self) -> int:
        """Number of non-zero connections."""
        return self.weights.nnz

    def get_state(self, *, copy: bool = True) -> dict:
        """Serialize weights, eligibility traces, and adolescent arrays for persistence.

        Args:
            copy: If True (default), returns copies of arrays (safe for
                  concurrent mutation).  If False, returns direct references
                  (zero-copy — only safe when caller guarantees immutability,
                  e.g. inside a fork-based snapshot).
        """
        _c = (lambda a: a.copy()) if copy else (lambda a: a)
        state = {
            "weights_data": _c(self.weights.data),
            "weights_indices": _c(self.weights.indices),
            "weights_indptr": _c(self.weights.indptr),
            "shape": self.weights.shape,
        }
        if self.eligibility is not None:
            state["eligibility"] = _c(self.eligibility)
        if self._elig_active is not None:
            state["elig_active"] = _c(self._elig_active)
        if self.bcm_theta is not None:
            state["bcm_theta"] = _c(self.bcm_theta)
        if self.stability_counter is not None:
            state["stability_counter"] = _c(self.stability_counter)
        if self.myelinated is not None:
            state["myelinated"] = _c(self.myelinated)
        if self.identity is not None:
            state["identity"] = _c(self.identity)
        if self.prune_survival_count is not None:
            state["prune_survival_count"] = _c(self.prune_survival_count)
        # Dynamic significance boosting state (avoids threshold reset on restart)
        if self._elig_sig_boosted:
            state["elig_sig_boosted"] = True
        return state

    def set_state(self, state: dict) -> None:
        """Restore weights, eligibility traces, and BCM state from serialized state.

        Validates shapes — if config changed since save, logs a warning
        and keeps current (freshly initialized) weights.
        """
        import logging
        _log = logging.getLogger(__name__)

        saved_shape = state["shape"]
        if saved_shape != (self.n_post, self.n_pre):
            _log.warning(
                f"Synapse group '{self.name}' shape changed "
                f"({saved_shape} → ({self.n_post}, {self.n_pre})), "
                "keeping freshly initialized weights"
            )
            return

        self.weights = sparse.csr_matrix(
            (state["weights_data"].copy(), state["weights_indices"].copy(), state["weights_indptr"].copy()),
            shape=saved_shape,
        )
        self._cache_structure()

        # Restore eligibility traces (match against current nnz, not old)
        if "eligibility" in state and self.eligibility is not None:
            saved = state["eligibility"]
            if len(saved) == self.weights.nnz:
                self.eligibility = saved.copy()
                # Restore or rebuild active eligibility index set
                if "elig_active" in state:
                    ea = state["elig_active"]
                    # Validate all indices are within bounds [0, nnz)
                    if len(ea) == 0 or (ea.min() >= 0 and ea.max() < self.weights.nnz):
                        self._elig_active = ea.copy()
                    else:
                        self._elig_active = np.flatnonzero(np.abs(self.eligibility) > self._elig_prune_threshold).astype(np.int32)
                else:
                    # Old state without elig_active — rebuild from eligibility
                    self._elig_active = np.flatnonzero(np.abs(self.eligibility) > self._elig_prune_threshold).astype(np.int32)
            else:
                _log.debug(f"Synapse '{self.name}' eligibility length mismatch, re-initializing")
                self.eligibility = np.zeros(self.weights.nnz, dtype=np.float32)
                self._elig_active = np.empty(0, dtype=np.int32)
        elif "eligibility" not in state and self.eligibility is not None:
            # Old state without eligibility saved — re-init to match loaded weights nnz
            if len(self.eligibility) != self.weights.nnz:
                self.eligibility = np.zeros(self.weights.nnz, dtype=np.float32)
                self._elig_active = np.empty(0, dtype=np.int32)
        if "bcm_theta" in state and self.bcm_theta is not None:
            saved = state["bcm_theta"]
            if len(saved) == len(self.bcm_theta):
                self.bcm_theta[:] = saved
            else:
                self.bcm_theta = np.full(self.n_post, self._bcm_config.theta_init, dtype=np.float32)
        # Restore adolescent arrays (size must match current nnz)
        nnz = self.weights.nnz
        # Default dtypes for re-initialization on length mismatch
        _defaults = {
            "stability_counter": (np.int32, 0),
            "myelinated": (np.bool_, False),
            "identity": (np.bool_, False),
            "prune_survival_count": (np.int32, 0),
        }
        for key, attr in [
            ("stability_counter", "stability_counter"),
            ("myelinated", "myelinated"),
            ("identity", "identity"),
            ("prune_survival_count", "prune_survival_count"),
        ]:
            if key in state and getattr(self, attr) is not None:
                saved = state[key]
                if len(saved) == nnz:
                    setattr(self, attr, saved.copy())
                else:
                    _log.debug(f"Synapse '{self.name}' {key} length mismatch, re-initializing")
                    dtype, fill = _defaults[key]
                    setattr(self, attr, np.full(nnz, fill, dtype=dtype))
        # Restore dynamic significance boosting state
        if state.get("elig_sig_boosted", False):
            self._elig_sig_boosted = True
            self._elig_significance = self._base_elig_significance * 2.0
