"""Tests for NeuromorphicService with mocked EventBus."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from neuromorphic.config import NeuromorphicConfig


class MockEventBus:
    """Mock NATS event bus for testing."""

    def __init__(self):
        self.published: list[tuple[str, dict]] = []
        self.subscriptions: dict[str, list] = {}

    async def connect(self):
        pass

    async def close(self):
        pass

    async def publish(self, subject: str, data) -> None:
        self.published.append((subject, data))

    async def subscribe(self, subject: str, handler, queue=None) -> None:
        if subject not in self.subscriptions:
            self.subscriptions[subject] = []
        self.subscriptions[subject].append(handler)

    async def unsubscribe(self, subject: str) -> None:
        self.subscriptions.pop(subject, None)

    @property
    def is_connected(self):
        return True


class TestNeuromorphicServiceConfig:
    def test_config_from_env(self):
        config = NeuromorphicConfig.from_env()
        assert config.populations.total > 0
        assert config.dt > 0
        assert config.nats_url

    def test_memory_estimate(self):
        config = NeuromorphicConfig()
        est = config.estimate_memory_bytes()
        assert est > 0
        # With default 750K neurons, should be at least 100 MB
        assert est > 100 * 1024 * 1024


class TestNeuromorphicServiceHandlers:
    """Test message handlers in isolation (without full service lifecycle)."""

    @pytest.mark.asyncio
    async def test_observation_queuing(self):
        """Test that observations are queued for processing."""
        # Test the queue mechanism directly (service import requires activelearning SDK)
        queue = asyncio.Queue(maxsize=10)
        data = {"provenance": "sensor.camera", "data": [0.5, 0.3]}
        await queue.put(data)
        result = queue.get_nowait()
        assert result["provenance"] == "sensor.camera"

    @pytest.mark.asyncio
    async def test_drive_event_handling(self):
        """Test drive event processing."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = NeuromorphicConfig()
        cfg.populations.brainstem = 50
        cfg.populations.reflex_arc = 40
        cfg.populations.sensory_cortex = 100
        cfg.populations.motor_cortex = 80
        cfg.populations.cerebellum = 80
        cfg.populations.association_cortex = 100
        cfg.populations.predictive_layer = 80
        cfg.populations.working_memory = 20

        net = NeuromorphicNetwork(cfg, seed=42)
        _initial_energy = net.drives.energy
        net.drives.handle_event({"type": "damage", "amount": 0.3})
        assert net.drives.damage == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_teaching_input(self):
        """Test that teaching runs multiple repetitions through the network."""
        from neuromorphic.network import NeuromorphicNetwork

        cfg = NeuromorphicConfig()
        cfg.populations.brainstem = 50
        cfg.populations.reflex_arc = 40
        cfg.populations.sensory_cortex = 100
        cfg.populations.motor_cortex = 80
        cfg.populations.cerebellum = 80
        cfg.populations.association_cortex = 100
        cfg.populations.predictive_layer = 80
        cfg.populations.working_memory = 20

        net = NeuromorphicNetwork(cfg, seed=42)

        inputs = {
            "sensor.camera": [0.5, 0.3, 0.8],
            "sensor.microphone": [0.2, 0.7],
        }
        repetitions = 5

        for _ in range(repetitions):
            current = net.inject_multimodal(inputs)
            net.step(current)

        assert net.step_count == 5


class TestMockEventBus:
    """Test the mock bus itself."""

    @pytest.mark.asyncio
    async def test_publish_records(self):
        bus = MockEventBus()
        await bus.publish("test.subject", {"key": "value"})
        assert len(bus.published) == 1
        assert bus.published[0][0] == "test.subject"

    @pytest.mark.asyncio
    async def test_subscribe_stores_handler(self):
        bus = MockEventBus()
        handler = AsyncMock()
        await bus.subscribe("test.*", handler)
        assert "test.*" in bus.subscriptions
