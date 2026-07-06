"""Tests for scale-adaptive parallel synaptic routing (ADR 0002 §1)."""

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.network import NeuromorphicNetwork


@pytest.fixture
def small_config():
    cfg = NeuromorphicConfig()
    cfg.populations.brainstem = 100
    cfg.populations.reflex_arc = 80
    cfg.populations.sensory_cortex = 400
    cfg.populations.motor_cortex = 200
    cfg.populations.cerebellum = 200
    cfg.populations.association_cortex = 400
    cfg.populations.predictive_layer = 200
    cfg.populations.working_memory = 50
    cfg.connections.sensory_motor_sparsity = 0.05
    cfg.connections.sensory_motor_weight = 0.8
    return cfg


class TestScaleAdaptiveParallelRouting:
    def test_small_network_disables_parallel_routing_by_default(
        self, small_config, monkeypatch
    ):
        monkeypatch.setenv("NEURO_STDP_THREADS", "4")
        monkeypatch.delenv("NEURO_PARALLEL_ROUTE_MIN_NEURONS", raising=False)
        monkeypatch.delenv("NEURO_PARALLEL_ROUTE", raising=False)
        net = NeuromorphicNetwork(small_config, seed=42)
        assert net._parallel_routing_enabled is False
        assert net._stdp_executor is not None

    def test_large_network_enables_parallel_routing(self, monkeypatch):
        monkeypatch.setenv("NEURO_STDP_THREADS", "4")
        monkeypatch.setenv("NEURO_PARALLEL_ROUTE_MIN_NEURONS", "1000")
        cfg = NeuromorphicConfig()
        cfg.populations.brainstem = 500
        cfg.populations.reflex_arc = 400
        cfg.populations.sensory_cortex = 2000
        cfg.populations.motor_cortex = 1000
        cfg.populations.cerebellum = 1000
        cfg.populations.association_cortex = 2000
        cfg.populations.predictive_layer = 1000
        cfg.populations.working_memory = 500
        net = NeuromorphicNetwork(cfg, seed=42)
        assert cfg.populations.total >= 1000
        assert net._parallel_routing_enabled is True

    def test_never_mode_disables_routing_even_for_large_net(self, monkeypatch):
        monkeypatch.setenv("NEURO_STDP_THREADS", "4")
        monkeypatch.setenv("NEURO_PARALLEL_ROUTE", "never")
        cfg = NeuromorphicConfig()
        cfg.populations.sensory_cortex = 50_000
        net = NeuromorphicNetwork(cfg, seed=42)
        assert net._parallel_routing_enabled is False

    def test_always_mode_enables_routing_even_for_small_net(
        self, small_config, monkeypatch
    ):
        monkeypatch.setenv("NEURO_STDP_THREADS", "4")
        monkeypatch.setenv("NEURO_PARALLEL_ROUTE", "always")
        net = NeuromorphicNetwork(small_config, seed=42)
        assert net._parallel_routing_enabled is True

    def test_serial_and_parallel_routes_produce_identical_state(
        self, small_config, monkeypatch
    ):
        """Numerical equivalence: routing mode must not change simulation math."""
        monkeypatch.setenv("NEURO_STDP_THREADS", "4")

        def _run_steps(parallel_mode: str) -> dict:
            monkeypatch.setenv("NEURO_PARALLEL_ROUTE", parallel_mode)
            net = NeuromorphicNetwork(small_config, seed=99)
            rng = np.random.default_rng(7)
            vis = rng.random(200).astype(np.float32)
            aud = rng.random(100).astype(np.float32)
            for _ in range(30):
                current = net.inject_multimodal({
                    "sensor.videofile.profile": vis,
                    "sensor.audiofile.profile": aud,
                })
                net.step(current)
            return {
                "motor": net.motor.spikes.copy(),
                "sensory_v": net.sensory.population.v_membrane.copy(),
                "step_count": net.step_count,
                "weights": net.synapses["sensory_association"].weights.data.copy(),
            }

        serial = _run_steps("never")
        parallel = _run_steps("always")
        np.testing.assert_array_equal(serial["motor"], parallel["motor"])
        np.testing.assert_array_equal(serial["sensory_v"], parallel["sensory_v"])
        assert serial["step_count"] == parallel["step_count"]
        np.testing.assert_allclose(serial["weights"], parallel["weights"])
