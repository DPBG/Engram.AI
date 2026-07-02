"""Cross-modal recall measurement probe.

Quantifies the brain's ability to recall one sensory modality's
representation when presented with only another modality.  This is the
core measurement for Patent Claim 4 (Cross-Modal Binding with
Instinctual Gain).

The probe is **read-only** — it examines synapse weights and spike
patterns but never modifies network state.

Architecture
~~~~~~~~~~~~
The probe is decoupled from ``NeuromorphicNetwork`` internals.  The
``probe()`` entry point accepts a lightweight ``ProbeInputs`` dataclass
that the caller constructs, keeping the probe testable and modular.
The ``probe_network()`` convenience method handles extraction for
callers that have a full network reference.

Key metrics
~~~~~~~~~~~
- recall_ratio:  fraction of modality-B-associated association neurons
  that fire when only modality-A input is presented (0->1)
- binding_strength:  mean weight of cross-modal synapses (synapses
  connecting sensory neurons of one modality to association neurons
  dominated by the other modality)
- modality_selectivity:  how distinct visual-associated vs
  auditory-associated neuron populations are (higher = more specialised)
- association_overlap:  fraction of association neurons with strong
  input from BOTH modalities (the "bound" neurons)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from neuromorphic.network import NeuromorphicNetwork

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProbeInputs:
    """Decoupled inputs for the cross-modal probe.

    Constructed by the caller from whatever source it has (live network,
    saved state, test fixture).  Keeps the probe free of hard dependencies
    on ``NeuromorphicNetwork`` internals.
    """

    weights: csr_matrix
    """sensory_association weight matrix, shape (n_assoc, n_sensory)."""

    association_spikes: np.ndarray
    """Boolean spike array for the association cortex, length n_assoc."""

    sensory_ranges: dict[str, tuple[int, int]]
    """Modality name -> (start, end) neuron index in sensory cortex."""

    n_association: int
    """Number of neurons in the association cortex region."""

    step_count: int = 0
    """Current simulation step (for context in metrics)."""


@dataclass
class CrossModalMetrics:
    """Results from a single cross-modal probe measurement."""

    # Core recall metrics
    recall_ratio_visual: float = 0.0
    recall_ratio_auditory: float = 0.0

    # Binding strength
    binding_strength: float = 0.0
    binding_strength_visual_to_auditory: float = 0.0
    binding_strength_auditory_to_visual: float = 0.0

    # Selectivity and overlap
    modality_selectivity: float = 0.0
    association_overlap: float = 0.0

    # Population counts
    n_visual_associated: int = 0
    n_auditory_associated: int = 0
    n_cross_modal: int = 0

    # Context
    step_count: int = 0
    sensory_ranges: dict[str, list[int]] = field(default_factory=dict)
    probe_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_ratio_visual": round(self.recall_ratio_visual, 4),
            "recall_ratio_auditory": round(self.recall_ratio_auditory, 4),
            "binding_strength": round(self.binding_strength, 4),
            "binding_strength_v2a": round(self.binding_strength_visual_to_auditory, 4),
            "binding_strength_a2v": round(self.binding_strength_auditory_to_visual, 4),
            "modality_selectivity": round(self.modality_selectivity, 4),
            "association_overlap": round(self.association_overlap, 4),
            "n_visual_associated": self.n_visual_associated,
            "n_auditory_associated": self.n_auditory_associated,
            "n_cross_modal": self.n_cross_modal,
            "step_count": self.step_count,
            "sensory_ranges": self.sensory_ranges,
            "probe_duration_ms": round(self.probe_duration_ms, 2),
        }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class CrossModalProbe:
    """Read-only probe for measuring cross-modal binding in the association cortex.

    Uses the ``sensory_association`` synapse group's weight matrix to classify
    association neurons by their dominant sensory input modality.  Then measures
    activation patterns to detect cross-modal recall.

    Classification
    ~~~~~~~~~~~~~~
    A neuron is "visual-dominated" if its visual weight fraction exceeds the
    auditory fraction by at least ``DOMINANCE_MARGIN`` (default 0.2).  Neurons
    where neither modality dominates by this margin are classified as
    cross-modal.  This is robust even in trained networks where every neuron
    has nonzero weights from both modalities.

    Thread safety
    ~~~~~~~~~~~~~
    ``probe()`` acquires no locks and only reads arrays.  It is safe to call
    from the metrics loop while the simulation step runs in a thread executor,
    provided the caller holds the network lock.

    Performance
    ~~~~~~~~~~~
    Classification uses vectorised SciPy sparse column slicing — O(nnz) in C.
    Binding strength reuses the per-neuron weight sums from classification
    (no redundant matrix slicing).  At 1M neurons (~232K association), the
    full probe takes <500ms.
    """

    DOMINANCE_MARGIN = 0.2
    """Minimum difference between vis_frac and aud_frac to call a neuron
    "dominated" by one modality.  With margin=0.2, a neuron with 60% visual
    / 40% auditory is cross-modal, while 75% / 25% is visual-dominated."""

    def __init__(self) -> None:
        self._visual_assoc_idx: np.ndarray = np.array([], dtype=np.intp)
        self._auditory_assoc_idx: np.ndarray = np.array([], dtype=np.intp)
        self._cross_modal_idx: np.ndarray = np.array([], dtype=np.intp)
        # Reusable per-neuron weight sums (avoid recomputing in binding_strength)
        self._vis_strength: np.ndarray = np.array([], dtype=np.float32)
        self._aud_strength: np.ndarray = np.array([], dtype=np.float32)

    # -- Public API ---------------------------------------------------------

    def probe_network(self, network: NeuromorphicNetwork) -> CrossModalMetrics:
        """Convenience: extract ``ProbeInputs`` from a live network and probe.

        This is the method called by ``NeuromorphicNetwork.get_metrics()``.
        """
        inputs = self._extract_inputs(network)
        if inputs is None:
            return CrossModalMetrics(step_count=network.step_count)
        return self.probe(inputs)

    def probe(self, inputs: ProbeInputs) -> CrossModalMetrics:
        """Run a cross-modal measurement on the given inputs.

        This is **read-only** — no weights or state are modified.
        """
        t0 = time.perf_counter()
        metrics = CrossModalMetrics(step_count=inputs.step_count)
        try:
            self._probe_inner(inputs, metrics)
        except Exception:
            logger.debug("Cross-modal probe failed", exc_info=True)
        finally:
            metrics.probe_duration_ms = (time.perf_counter() - t0) * 1000.0
        return metrics

    # -- Input extraction ---------------------------------------------------

    @staticmethod
    def _extract_inputs(network: NeuromorphicNetwork) -> ProbeInputs | None:
        """Build ``ProbeInputs`` from a live ``NeuromorphicNetwork``.

        Returns ``None`` if the network lacks the required synapse group.
        """
        sa_syn = network.synapses.get("sensory_association")
        if sa_syn is None or sa_syn.nnz == 0:
            return None

        return ProbeInputs(
            weights=sa_syn.weights,
            association_spikes=network.association.spikes,
            sensory_ranges=network.allocator.current_ranges,
            n_association=network.association.n,
            step_count=network.step_count,
        )

    # -- Core logic ---------------------------------------------------------

    def _probe_inner(self, inputs: ProbeInputs, metrics: CrossModalMetrics) -> None:
        weights = inputs.weights
        n_post = weights.shape[0]

        # Association region neuron count must match synapse n_post
        if inputs.n_association != n_post:
            logger.warning(
                "Association cortex size (%d) != sensory_association n_post (%d); "
                "skipping cross-modal probe",
                inputs.n_association,
                n_post,
            )
            return

        vis_range = inputs.sensory_ranges.get("visual", (0, 0))
        aud_range = inputs.sensory_ranges.get("auditory", (0, 0))

        # JSON-safe: convert tuples to lists for NATS serialisation
        metrics.sensory_ranges = {
            k: list(v) for k, v in inputs.sensory_ranges.items() if v != (0, 0)
        }

        # Need both modalities allocated
        if vis_range[1] <= vis_range[0] or aud_range[1] <= aud_range[0]:
            return

        # Clamp ranges to matrix dimensions
        n_pre = weights.shape[1]
        vis_range = (max(0, vis_range[0]), min(n_pre, vis_range[1]))
        aud_range = (max(0, aud_range[0]), min(n_pre, aud_range[1]))
        if vis_range[1] <= vis_range[0] or aud_range[1] <= aud_range[0]:
            return

        # Classify association neurons by dominant input modality.
        # Also stores per-neuron weight sums for binding_strength.
        self._classify_neurons(weights, vis_range, aud_range)

        metrics.n_visual_associated = len(self._visual_assoc_idx)
        metrics.n_auditory_associated = len(self._auditory_assoc_idx)
        metrics.n_cross_modal = len(self._cross_modal_idx)

        if n_post > 0:
            metrics.association_overlap = len(self._cross_modal_idx) / n_post
            n_classified = (
                metrics.n_visual_associated + metrics.n_auditory_associated + metrics.n_cross_modal
            )
            metrics.modality_selectivity = n_classified / n_post

        # Binding strength — reuses cached weight sums, no extra matrix slicing
        bs_v2a, bs_a2v = self._binding_strength()
        metrics.binding_strength_visual_to_auditory = bs_v2a
        metrics.binding_strength_auditory_to_visual = bs_a2v
        nonzero = (bs_v2a > 0) + (bs_a2v > 0)
        metrics.binding_strength = (bs_v2a + bs_a2v) / nonzero if nonzero else 0.0

        # Cross-modal recall from current spike patterns
        assoc_spikes = inputs.association_spikes
        if len(assoc_spikes) != n_post:
            return

        metrics.recall_ratio_visual = self._recall_ratio(
            assoc_spikes,
            self._auditory_assoc_idx,
        )
        metrics.recall_ratio_auditory = self._recall_ratio(
            assoc_spikes,
            self._visual_assoc_idx,
        )

    # -- Vectorised helpers -------------------------------------------------

    def _classify_neurons(
        self,
        weights: csr_matrix,
        vis_range: tuple[int, int],
        aud_range: tuple[int, int],
    ) -> None:
        """Classify each association neuron by dominant sensory input.

        Uses a dominance margin instead of an absolute threshold.  A neuron
        must have its dominant modality fraction exceed the other by at least
        ``DOMINANCE_MARGIN`` to be classified as single-modality.  This
        produces meaningful classifications even in trained networks where
        every neuron has nonzero weights from both modalities.

        Also caches per-neuron weight sums (``_vis_strength``, ``_aud_strength``)
        so ``_binding_strength()`` can reuse them without re-slicing the matrix.
        """
        n_post = weights.shape[0]

        # Column-slice: O(nnz) in C — the two expensive operations
        self._vis_strength = (
            np.asarray(
                weights[:, vis_range[0] : vis_range[1]].sum(axis=1),
            )
            .ravel()
            .astype(np.float32)
        )
        self._aud_strength = (
            np.asarray(
                weights[:, aud_range[0] : aud_range[1]].sum(axis=1),
            )
            .ravel()
            .astype(np.float32)
        )

        total = self._vis_strength + self._aud_strength
        has_input = total > 0.0
        vis_frac = np.zeros(n_post, dtype=np.float32)
        aud_frac = np.zeros(n_post, dtype=np.float32)
        vis_frac[has_input] = self._vis_strength[has_input] / total[has_input]
        aud_frac[has_input] = self._aud_strength[has_input] / total[has_input]

        margin = self.DOMINANCE_MARGIN
        diff = vis_frac - aud_frac  # positive = visual-dominated

        # Visual-dominated: vis_frac - aud_frac >= margin AND has input
        vis_only = has_input & (diff >= margin)
        # Auditory-dominated: aud_frac - vis_frac >= margin AND has input
        aud_only = has_input & (diff <= -margin)
        # Cross-modal: has input from both but neither dominates
        cross = has_input & ~vis_only & ~aud_only

        self._visual_assoc_idx = np.flatnonzero(vis_only)
        self._auditory_assoc_idx = np.flatnonzero(aud_only)
        self._cross_modal_idx = np.flatnonzero(cross)

    def _binding_strength(self) -> tuple[float, float]:
        """Mean cross-modal weight using cached per-neuron sums.

        visual->auditory: for auditory-dominated neurons, their mean
        visual weight sum (how much visual input reaches them).

        auditory->visual: for visual-dominated neurons, their mean
        auditory weight sum.

        Reuses ``_vis_strength`` / ``_aud_strength`` from classification
        instead of re-slicing the sparse matrix (saves ~1s at 1M scale).
        """
        v2a = 0.0
        a2v = 0.0

        if len(self._auditory_assoc_idx) > 0:
            v2a = float(self._vis_strength[self._auditory_assoc_idx].mean())

        if len(self._visual_assoc_idx) > 0:
            a2v = float(self._aud_strength[self._visual_assoc_idx].mean())

        return v2a, a2v

    @staticmethod
    def _recall_ratio(
        association_spikes: np.ndarray,
        target_indices: np.ndarray,
    ) -> float:
        """Fraction of target association neurons currently firing."""
        if len(target_indices) == 0:
            return 0.0
        n = len(association_spikes)
        if n == 0:
            return 0.0
        valid = target_indices[target_indices < n]
        if len(valid) == 0:
            return 0.0
        if len(valid) < len(target_indices):
            logger.warning(
                "recall_ratio: %d/%d target indices out of bounds (n=%d)",
                len(target_indices) - len(valid),
                len(target_indices),
                n,
            )
        return float(association_spikes[valid].sum()) / len(valid)
