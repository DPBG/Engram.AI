"""Tests for Global Neuronal Workspace competition and broadcast (CIP-23)."""

import numpy as np

from neuromorphic.config import GlobalWorkspaceConfig, NeuromorphicConfig
from neuromorphic.regions import GlobalWorkspace


def _small_ws_config(**overrides):
    """Small config with workspace enabled for fast tests."""
    cfg = NeuromorphicConfig(**overrides)
    cfg.populations.brainstem = 100
    cfg.populations.reflex_arc = 80
    cfg.populations.sensory_cortex = 400
    cfg.populations.motor_cortex = 200
    cfg.populations.cerebellum = 200
    cfg.populations.association_cortex = 400
    cfg.populations.predictive_layer = 200
    cfg.populations.working_memory = 50
    cfg.populations.global_workspace = 100
    return cfg


class TestGlobalWorkspaceConfig:
    """Test GlobalWorkspaceConfig defaults and env parsing."""

    def test_defaults(self):
        cfg = GlobalWorkspaceConfig()
        assert cfg.enabled is False
        assert cfg.n_neurons == 5000
        assert cfg.ignition_threshold == 0.3
        assert cfg.broadcast_gain == 3.0
        assert cfg.refractory_steps == 50

    def test_neuromorphic_config_has_workspace(self):
        cfg = NeuromorphicConfig()
        assert hasattr(cfg, "global_workspace")
        assert cfg.global_workspace.enabled is False

    def test_population_default_zero(self):
        cfg = NeuromorphicConfig()
        assert cfg.populations.global_workspace == 0


class TestGlobalWorkspaceRegion:
    """Test the GlobalWorkspace region class."""

    def _make_workspace(self, n=100, **gw_overrides):
        cfg = NeuromorphicConfig(
            global_workspace=GlobalWorkspaceConfig(enabled=True, **gw_overrides)
        )
        cfg.populations.global_workspace = n
        return GlobalWorkspace(cfg)

    def test_creation(self):
        ws = self._make_workspace()
        assert ws.n == 100
        assert ws.name == "global_workspace"
        assert not ws.ignition_active

    def test_ignition_off_with_no_input(self):
        """No input should mean no ignition."""
        ws = self._make_workspace(ignition_threshold=0.2)
        for _ in range(20):
            ws.step(np.zeros(100, dtype=np.float32))
        assert not ws.ignition_active
        assert ws.ignition_strength < 0.15  # sigmoid(slope * (0 - 0.2)) = 0.12

    def test_ignition_with_strong_input(self):
        """Strong sustained input should trigger ignition."""
        ws = self._make_workspace(ignition_threshold=0.1, refractory_steps=5)
        ignited = False
        for _ in range(50):
            ws.step(np.ones(100, dtype=np.float32) * 25.0)
            if ws.ignition_active:
                ignited = True
                break
        assert ignited, "Strong input should trigger ignition"
        assert ws._ignition_count >= 1

    def test_refractory_prevents_rapid_ignition(self):
        """After ignition, workspace should be silent for refractory_steps."""
        ws = self._make_workspace(ignition_threshold=0.1, refractory_steps=20)
        # Drive to ignition
        for _ in range(30):
            ws.step(np.ones(100, dtype=np.float32) * 25.0)
        ignition_count_before = ws._ignition_count

        # During refractory, input is suppressed
        for _ in range(15):
            ws.step(np.ones(100, dtype=np.float32) * 25.0)

        # Should not have re-ignited during refractory
        assert ws._ignition_count == ignition_count_before or ws._refractory_counter > 0

    def test_broadcast_gain_during_ignition(self):
        """broadcast_gain should be > 1 during ignition, 1.0 otherwise."""
        ws = self._make_workspace(ignition_threshold=0.1, broadcast_gain=5.0, refractory_steps=3)
        assert ws.get_broadcast_gain() == 1.0

        # Drive ignition
        for _ in range(50):
            ws.step(np.ones(100, dtype=np.float32) * 25.0)
            if ws.ignition_active:
                assert ws.get_broadcast_gain() == 5.0
                break

    def test_workspace_state_persistence(self):
        """get_workspace_state/set_workspace_state should roundtrip."""
        ws = self._make_workspace(ignition_threshold=0.1, refractory_steps=10)
        for _ in range(30):
            ws.step(np.ones(100, dtype=np.float32) * 25.0)

        state = ws.get_workspace_state()
        assert "refractory_counter" in state
        assert "ignition_active" in state
        assert "ignition_count" in state
        assert "firing_rate_history" in state

        ws2 = self._make_workspace(ignition_threshold=0.1, refractory_steps=10)
        ws2.set_workspace_state(state)
        assert ws2._ignition_count == ws._ignition_count
        assert ws2._refractory_counter == ws._refractory_counter
        assert ws2._ignition_active == ws._ignition_active

    def test_ignition_strength_sigmoid(self):
        """ignition_strength should be a smooth sigmoid of firing rate."""
        ws = self._make_workspace(ignition_threshold=0.3, sigmoid_slope=10.0)
        # Simulate with known firing rate
        ws._firing_rate_history = [0.0]
        assert ws.ignition_strength < 0.1  # well below threshold

        ws._firing_rate_history = [0.3]  # at threshold
        strength_at_thresh = ws.ignition_strength
        assert 0.4 < strength_at_thresh < 0.6  # sigmoid(0) = 0.5

        ws._firing_rate_history = [0.6]  # above threshold
        assert ws.ignition_strength > 0.9  # well above


class TestWorkspaceNetwork:
    """Test workspace integration in the full network."""

    def test_workspace_synapse_groups_exist(self):
        """Network should have workspace synapse groups when enabled."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        net = NeuromorphicNetwork(cfg, seed=42)

        # Base afferent groups
        assert "association_workspace" in net.synapses
        assert "predictive_workspace" in net.synapses
        assert "working_workspace" in net.synapses

        # Base efferent groups
        assert "workspace_association" in net.synapses
        assert "workspace_predictive" in net.synapses
        assert "workspace_working" in net.synapses
        assert "workspace_motor" in net.synapses

        # Infrastructure
        assert "workspace_lateral" in net.synapses
        assert "brainstem_workspace" in net.synapses

    def test_workspace_region_exists(self):
        """Network should have global_workspace region."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        net = NeuromorphicNetwork(cfg, seed=42)
        assert "global_workspace" in net.regions
        assert net.workspace is not None
        assert net.workspace.n == 100

    def test_step_with_workspace(self):
        """Network step should work with workspace enabled."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        net = NeuromorphicNetwork(cfg, seed=42)
        result = net.step()
        assert "motor_commands" in result

    def test_multiple_steps_with_workspace(self):
        """Multiple steps should work without error."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        net = NeuromorphicNetwork(cfg, seed=42)
        for _ in range(20):
            result = net.step()
        assert result["step_count"] == 20

    def test_no_workspace_without_population(self):
        """When global_workspace=0, no workspace region or synapses should exist."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = NeuromorphicConfig()
        cfg.populations.brainstem = 100
        cfg.populations.reflex_arc = 80
        cfg.populations.sensory_cortex = 400
        cfg.populations.motor_cortex = 200
        cfg.populations.cerebellum = 200
        cfg.populations.association_cortex = 400
        cfg.populations.predictive_layer = 200
        cfg.populations.working_memory = 50
        net = NeuromorphicNetwork(cfg, seed=42)
        assert "global_workspace" not in net.regions
        assert net.workspace is None
        assert "association_workspace" not in net.synapses

    def test_workspace_with_concept_layer(self):
        """Workspace should have concept synapse groups when concept layer exists."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        cfg.populations.concept_layer = 50
        net = NeuromorphicNetwork(cfg, seed=42)
        assert "concept_workspace" in net.synapses
        assert "workspace_concept" in net.synapses

    def test_workspace_with_feature_layer(self):
        """Workspace should have feature synapse groups when feature layer exists."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        cfg.populations.feature_layer = 100
        net = NeuromorphicNetwork(cfg, seed=42)
        assert "feature_workspace" in net.synapses
        assert "workspace_feature" in net.synapses

    def test_workspace_with_meta(self):
        """Workspace should have meta afferent when meta-controller exists."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        cfg.populations.meta_controller = 50
        net = NeuromorphicNetwork(cfg, seed=42)
        assert "meta_workspace" in net.synapses

    def test_workspace_lateral_not_plastic(self):
        """Lateral inhibition in workspace should be non-plastic."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        net = NeuromorphicNetwork(cfg, seed=42)
        assert not net.synapses["workspace_lateral"].plastic

    def test_workspace_state_in_network(self):
        """Network state should include workspace state."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = _small_ws_config()
        net = NeuromorphicNetwork(cfg, seed=42)
        for _ in range(5):
            net.step()
        state = net.get_state()
        assert "workspace" in state
        assert "ignition_count" in state["workspace"]
        assert "ignition_active" in state["workspace"]
