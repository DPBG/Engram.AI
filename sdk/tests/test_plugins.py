"""Tests for plugin interfaces."""

import pytest

from activelearning.core import ActionProposal
from activelearning.plugins import (
    ActuatorPlugin,
    PluginCapability,
    RiskClass,
    SensorPlugin,
    get_actuator,
    get_sensor,
    list_actuators,
    list_sensors,
    register_actuator,
    register_sensor,
)


class MockSensor(SensorPlugin[dict]):
    """Mock sensor for testing."""

    def __init__(self, sensor_id: str = "test.sensor"):
        super().__init__(
            sensor_id=sensor_id,
            name="Test Sensor",
            description="A test sensor",
            rate_limit_hz=10.0,
        )
        self.capture_count = 0

    async def capture(self) -> dict:
        self.capture_count += 1
        return {"frame": self.capture_count}


class MockActuator(ActuatorPlugin[dict]):
    """Mock actuator for testing."""

    def __init__(self, actuator_id: str = "test.actuator"):
        super().__init__(
            actuator_id=actuator_id,
            name="Test Actuator",
            description="A test actuator",
            envelope={"speed": (0.0, 100.0), "angle": (0.0, 180.0)},
        )
        self.executed_actions = []

    async def _do_execute(self, action: dict) -> bool:
        self.executed_actions.append(action)
        return True


class TestPluginCapability:
    """Tests for PluginCapability."""

    def test_capability_basic(self):
        cap = PluginCapability(
            name="capture_frame",
            description="Capture a single frame",
            parameters={"resolution": "str"},
        )
        assert cap.name == "capture_frame"
        assert cap.risk_class == RiskClass.LOW

    def test_capability_with_risk(self):
        cap = PluginCapability(
            name="move_arm",
            description="Move robotic arm",
            parameters={"x": "float", "y": "float"},
            risk_class=RiskClass.HIGH,
        )
        assert cap.risk_class == RiskClass.HIGH


class TestSensorPlugin:
    """Tests for SensorPlugin base class."""

    def test_sensor_metadata(self):
        sensor = MockSensor()
        metadata = sensor.describe()
        assert metadata.sensor_id == "test.sensor"
        assert metadata.name == "Test Sensor"
        assert metadata.rate_limit_hz == 10.0

    def test_sensor_add_capability(self):
        sensor = MockSensor()
        cap = PluginCapability(name="test", description="test")
        sensor.add_capability(cap)
        assert len(sensor.get_capabilities()) == 1

    @pytest.mark.asyncio
    async def test_sensor_capture(self):
        sensor = MockSensor()
        data = await sensor.capture()
        assert data == {"frame": 1}
        assert sensor.capture_count == 1

    @pytest.mark.asyncio
    async def test_sensor_emit_without_start(self):
        sensor = MockSensor()
        with pytest.raises(RuntimeError, match="not started"):
            await sensor.emit({"test": "data"})


class TestActuatorPlugin:
    """Tests for ActuatorPlugin base class."""

    def test_actuator_metadata(self):
        actuator = MockActuator()
        metadata = actuator.describe()
        assert metadata.actuator_id == "test.actuator"
        assert "speed" in metadata.envelope
        assert "angle" in metadata.envelope

    def test_actuator_validate_envelope_valid(self):
        actuator = MockActuator()
        is_valid, error = actuator.validate_envelope({"speed": 50.0, "angle": 90.0})
        assert is_valid
        assert error is None

    def test_actuator_validate_envelope_invalid_range(self):
        actuator = MockActuator()
        is_valid, error = actuator.validate_envelope({"speed": 150.0})
        assert not is_valid
        assert "outside range" in error

    def test_actuator_validate_envelope_invalid_type(self):
        actuator = MockActuator()
        is_valid, error = actuator.validate_envelope({"speed": "fast"})
        assert not is_valid
        assert "must be numeric" in error

    @pytest.mark.asyncio
    async def test_actuator_execute_without_start(self):
        actuator = MockActuator()
        proposal = ActionProposal(
            trace_id="test",
            provenance="test",
            action={"speed": 50},
        )
        with pytest.raises(RuntimeError, match="not started"):
            await actuator.execute(proposal)


class TestPluginRegistry:
    """Tests for plugin registration."""

    def test_register_sensor(self):
        sensor = MockSensor("registry.test.sensor")
        registered = register_sensor(sensor)
        assert registered is sensor
        assert get_sensor("registry.test.sensor") is sensor

    def test_register_actuator(self):
        actuator = MockActuator("registry.test.actuator")
        registered = register_actuator(actuator)
        assert registered is actuator
        assert get_actuator("registry.test.actuator") is actuator

    def test_list_sensors(self):
        sensor = MockSensor("list.test.sensor")
        register_sensor(sensor)
        sensors = list_sensors()
        sensor_ids = [s.sensor_id for s in sensors]
        assert "list.test.sensor" in sensor_ids

    def test_list_actuators(self):
        actuator = MockActuator("list.test.actuator")
        register_actuator(actuator)
        actuators = list_actuators()
        actuator_ids = [a.actuator_id for a in actuators]
        assert "list.test.actuator" in actuator_ids

    def test_get_nonexistent_sensor(self):
        result = get_sensor("nonexistent.sensor")
        assert result is None

    def test_get_nonexistent_actuator(self):
        result = get_actuator("nonexistent.actuator")
        assert result is None
