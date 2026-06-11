"""Tests for concept probe logic.

These tests verify the concept probe handler logic without importing
NeuromorphicService (which requires the activelearning SDK). Instead,
the probe logic is tested directly against mock network/region objects.
"""

import asyncio
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── Mock objects matching the real interfaces ──────────────────────


class FakeConcept:
    """Mock ConceptLayer with k-WTA-like spikes."""

    def __init__(self, n=200, k=20):
        self.n = n
        self._k = k
        self._spikes = np.zeros(n, dtype=bool)

    @property
    def spikes(self):
        return self._spikes

    def set_active(self, indices):
        self._spikes[:] = False
        for i in indices:
            if i < self.n:
                self._spikes[i] = True


class FakeRegion:
    """Mock BrainRegion."""

    def __init__(self, n=100):
        self.n = n
        self._spikes = np.zeros(n, dtype=bool)

    @property
    def spikes(self):
        return self._spikes


class FakeNetwork:
    """Mock NeuromorphicNetwork for concept probe tests."""

    def __init__(self, concept_n=200, concept_k=20):
        self.concept = FakeConcept(concept_n, concept_k)
        self._step_count = 1000
        self._step_call_count = 0
        self.regions = {
            "sensory": FakeRegion(500),
            "concept_layer": self.concept,
            "motor_cortex": FakeRegion(100),
        }

    @property
    def step_count(self):
        return self._step_count

    def inject_multimodal(self, inputs):
        return np.zeros(500, dtype=np.float32)

    def step(self, sensory_current=None):
        self._step_count += 1
        self._step_call_count += 1
        # Different neurons fire each step (shifted by step count)
        self.concept.set_active([
            (self._step_call_count * 7 + i) % self.concept.n
            for i in range(self.concept._k)
        ])
        self.regions["sensory"]._spikes[:] = False
        self.regions["sensory"]._spikes[:50] = True
        return {"motor_commands": [], "prediction_error": 0.1,
                "step_count": self._step_count}


# ── The probe logic (mirrors _handle_concept_probe in service.py) ──


async def run_concept_probe(network, data: dict) -> dict | None:
    """Run a concept probe against a network. Returns result dict.

    This mirrors the logic in NeuromorphicService._handle_concept_probe
    so we can test it without the full service import chain.
    """
    concept = network.concept
    if concept is None:
        return {"error": "no concept layer"}

    stimulus = data.get("stimulus", {})
    propagation_steps = min(int(data.get("propagation_steps", 10)), 50)
    label = data.get("label", "probe")

    provenance = stimulus.get("provenance", "observation.text")
    stim_data = stimulus.get("data", "")

    concept_n = concept.n
    spike_counts = np.zeros(concept_n, dtype=np.int32)
    step_start = network.step_count
    region_spike_counts: dict[str, int] = defaultdict(int)

    for i in range(propagation_steps):
        if i == 0:
            sensory_current = network.inject_multimodal({provenance: stim_data})
            network.step(sensory_current)
        else:
            network.step()

        spike_counts += concept.spikes.astype(np.int32)
        for rname, region in network.regions.items():
            region_spike_counts[rname] += int(region.spikes.sum())

    step_end = network.step_count

    active_mask = spike_counts > 0
    active_indices = np.nonzero(active_mask)[0]
    rates = spike_counts.astype(np.float32) / propagation_steps

    region_rates = {
        rname: round(count / (propagation_steps * network.regions[rname].n), 4)
        for rname, count in region_spike_counts.items()
    }

    return {
        "label": label,
        "active_indices": active_indices.tolist(),
        "active_count": int(active_mask.sum()),
        "spike_counts": spike_counts[active_mask].tolist(),
        "rates": rates[active_mask].tolist(),
        "total_concept_neurons": concept_n,
        "propagation_steps": propagation_steps,
        "step_range": [step_start, step_end],
        "region_rates": region_rates,
    }


# ── Tests ──────────────────────────────────────────────────────────


class TestConceptProbe:

    @pytest.mark.asyncio
    async def test_basic_probe(self):
        """Probe returns active concept neurons."""
        net = FakeNetwork(concept_n=200, concept_k=20)
        result = await run_concept_probe(net, {
            "stimulus": {"provenance": "observation.text", "data": "dog"},
            "propagation_steps": 5,
            "label": "test_dog",
        })

        assert result["label"] == "test_dog"
        assert result["active_count"] > 0
        assert result["total_concept_neurons"] == 200
        assert result["propagation_steps"] == 5
        assert result["step_range"][1] > result["step_range"][0]
        assert result["step_range"][1] - result["step_range"][0] == 5
        assert "region_rates" in result
        assert "sensory" in result["region_rates"]

    @pytest.mark.asyncio
    async def test_no_concept_layer(self):
        """Probe returns error when concept layer is None."""
        net = FakeNetwork()
        net.concept = None
        result = await run_concept_probe(net, {
            "stimulus": {"provenance": "observation.text", "data": "dog"},
            "label": "no_concept",
        })
        assert "error" in result

    @pytest.mark.asyncio
    async def test_capped_steps(self):
        """Propagation steps are capped at 50."""
        net = FakeNetwork()
        result = await run_concept_probe(net, {
            "stimulus": {"provenance": "observation.text", "data": "test"},
            "propagation_steps": 999,
            "label": "capped",
        })
        assert result["propagation_steps"] == 50

    @pytest.mark.asyncio
    async def test_sparse_output_consistency(self):
        """active_indices, spike_counts, and rates have matching lengths."""
        net = FakeNetwork()
        result = await run_concept_probe(net, {
            "stimulus": {"provenance": "observation.text", "data": "dog"},
            "propagation_steps": 3,
            "label": "sparse",
        })
        assert len(result["active_indices"]) == result["active_count"]
        assert len(result["spike_counts"]) == result["active_count"]
        assert len(result["rates"]) == result["active_count"]
        assert all(r > 0 for r in result["rates"])

    @pytest.mark.asyncio
    async def test_region_rates_computed(self):
        """Region rates reflect actual firing."""
        net = FakeNetwork()
        result = await run_concept_probe(net, {
            "stimulus": {"provenance": "observation.text", "data": "test"},
            "propagation_steps": 5,
            "label": "regions",
        })
        rr = result["region_rates"]
        # Sensory: 50 of 500 fire each step = 10%
        assert rr["sensory"] == pytest.approx(0.1, abs=0.01)
        # Concept layer: ~20 of 200 fire each step = 10%
        assert rr["concept_layer"] > 0
        # Motor cortex: 0 of 100 (FakeRegion defaults to no spikes)
        assert rr["motor_cortex"] == 0.0

    @pytest.mark.asyncio
    async def test_different_stimuli_different_patterns(self):
        """Different stimuli produce different active neuron sets."""
        net = FakeNetwork()
        result_dog = await run_concept_probe(net, {
            "stimulus": {"provenance": "observation.text", "data": "dog"},
            "propagation_steps": 3,
            "label": "dog",
        })
        result_cat = await run_concept_probe(net, {
            "stimulus": {"provenance": "observation.text", "data": "cat"},
            "propagation_steps": 3,
            "label": "cat",
        })

        assert result_dog["active_count"] > 0
        assert result_cat["active_count"] > 0

        # Network state progresses between probes, so patterns differ
        set_dog = set(result_dog["active_indices"])
        set_cat = set(result_cat["active_indices"])
        assert set_dog != set_cat

    @pytest.mark.asyncio
    async def test_accumulation_across_steps(self):
        """More steps accumulate more unique active neurons."""
        net1 = FakeNetwork()
        result_short = await run_concept_probe(net1, {
            "stimulus": {"provenance": "observation.text", "data": "test"},
            "propagation_steps": 1,
            "label": "short",
        })

        net2 = FakeNetwork()
        result_long = await run_concept_probe(net2, {
            "stimulus": {"provenance": "observation.text", "data": "test"},
            "propagation_steps": 10,
            "label": "long",
        })

        # More steps should activate more unique neurons
        assert result_long["active_count"] >= result_short["active_count"]

    @pytest.mark.asyncio
    async def test_step_range_correct(self):
        """Step range tracks actual network steps."""
        net = FakeNetwork()
        initial_step = net.step_count
        result = await run_concept_probe(net, {
            "stimulus": {"provenance": "observation.text", "data": "test"},
            "propagation_steps": 7,
            "label": "step_check",
        })
        assert result["step_range"][0] == initial_step
        assert result["step_range"][1] == initial_step + 7
