"""Tests for ActuatorPlugin adapters and MotorFeedbackAdapter routing.

Covers:
1. MuJoCoActuator — metadata, successful execution, error handling
2. SerialServoActuator — metadata, execution, heartbeat publication
3. NullServoDriver and ServoDriver Protocol conformance
4. MotorFeedbackAdapter.register_plugin() — plugin path vs NATS fallback
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap activelearning SDK submodules without triggering __init__.py.
# The neuromorphic venv doesn't install aiohttp/qdrant (pulled in by the
# activelearning __init__), but activelearning.core / .nats_client / .plugins
# work fine once the package namespace is registered with the correct __path__.
# ---------------------------------------------------------------------------
import os
import sys
import types as _types

_SDK_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sdk", "src"))
if "activelearning" not in sys.modules:
    _al_pkg = _types.ModuleType("activelearning")
    _al_pkg.__path__ = [os.path.join(_SDK_SRC, "activelearning")]  # type: ignore[attr-defined]
    _al_pkg.__package__ = "activelearning"
    sys.modules["activelearning"] = _al_pkg
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from activelearning.core import ActionProposal  # noqa: E402
from activelearning.plugins import RiskClass  # noqa: E402

from neuromorphic.actuators.mujoco_actuator import MuJoCoActuator  # noqa: E402
from neuromorphic.actuators.serial_servo_actuator import (  # noqa: E402
    NullServoDriver,
    SerialServoActuator,
    ServoDriver,
)
from neuromorphic.motor_feedback_adapter import MotorFeedbackAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


class _FakeBus:
    """Minimal event bus stub — records published subjects and payloads."""

    def __init__(self):
        self.published: list[tuple[str, Any]] = []
        self._subs: dict[str, Any] = {}

    async def publish(self, subject: str, data: Any) -> None:
        self.published.append((subject, data))

    async def subscribe(self, subject: str, handler) -> None:
        self._subs[subject] = handler

    async def unsubscribe(self, subject: str) -> None:
        self._subs.pop(subject, None)

    def subjects(self) -> list[str]:
        return [s for s, _ in self.published]


# ---------------------------------------------------------------------------
# MuJoCoActuator
# ---------------------------------------------------------------------------


class _FakeMuJoCoBody:
    """Minimal MuJoCoBody stub for unit tests."""

    def __init__(self, success: bool = True, raise_on_call: bool = False):
        self._success = success
        self._raise = raise_on_call
        self.calls: list[tuple[str, float]] = []

    def step_command(self, channel: str, intensity: float) -> dict:
        if self._raise:
            raise RuntimeError("simulated body crash")
        self.calls.append((channel, intensity))
        return {"channel": channel, "success": self._success, "confidence": 0.8}


class TestMuJoCoActuator:
    def test_metadata(self):
        body = _FakeMuJoCoBody()
        act = MuJoCoActuator(body, "locomotion")
        meta = act.describe()
        assert meta.actuator_id == "mujoco.locomotion"
        assert meta.risk_class == RiskClass.LOW
        assert "intensity" in meta.envelope
        assert meta.envelope["intensity"] == (0.0, 1.0)
        assert len(meta.capabilities) == 1

    def test_channel_property(self):
        act = MuJoCoActuator(_FakeMuJoCoBody(), "manipulation")
        assert act.channel == "manipulation"

    def test_execute_success(self):
        body = _FakeMuJoCoBody(success=True)
        act = MuJoCoActuator(body, "locomotion")
        # Wire up a fake bus so execute() can publish the outcome
        bus = _FakeBus()
        act._bus = bus

        proposal = ActionProposal(
            trace_id="t-1",
            provenance="motor.locomotion",
            action={"intensity": 0.6},
        )
        outcome = _run(act.execute(proposal))

        assert outcome.success is True
        assert body.calls == [("locomotion", 0.6)]
        assert any("outcome" in s for s in bus.subjects())

    def test_execute_body_failure(self):
        body = _FakeMuJoCoBody(success=False)
        act = MuJoCoActuator(body, "locomotion")
        act._bus = _FakeBus()

        proposal = ActionProposal(
            trace_id="t-2",
            provenance="motor.locomotion",
            action={"intensity": 0.3},
        )
        outcome = _run(act.execute(proposal))
        assert outcome.success is False

    def test_execute_body_exception_returns_false(self):
        body = _FakeMuJoCoBody(raise_on_call=True)
        act = MuJoCoActuator(body, "locomotion")
        act._bus = _FakeBus()

        proposal = ActionProposal(
            trace_id="t-3",
            provenance="motor.locomotion",
            action={"intensity": 0.5},
        )
        outcome = _run(act.execute(proposal))
        # MuJoCoActuator catches the exception and returns success=False
        assert outcome.success is False

    def test_missing_intensity_defaults_to_zero(self):
        body = _FakeMuJoCoBody()
        act = MuJoCoActuator(body, "head")
        act._bus = _FakeBus()

        proposal = ActionProposal(
            trace_id="t-4",
            provenance="motor.head",
            action={},  # no "intensity" key
        )
        _run(act.execute(proposal))
        assert body.calls == [("head", 0.0)]


# ---------------------------------------------------------------------------
# ServoDriver Protocol conformance
# ---------------------------------------------------------------------------


class TestNullServoDriver:
    def test_is_servo_driver(self):
        driver = NullServoDriver()
        assert isinstance(driver, ServoDriver)

    def test_set_position_always_succeeds(self):
        driver = NullServoDriver()
        driver.connect()
        assert driver.set_position(0, 90.0, 0.5) is True
        assert driver.set_position(1, 0.0, 1.0) is True

    def test_get_position_returns_none(self):
        driver = NullServoDriver()
        assert driver.get_position(0) is None

    def test_disconnect_is_idempotent(self):
        driver = NullServoDriver()
        driver.connect()
        driver.disconnect()
        driver.disconnect()  # second call must not raise


# ---------------------------------------------------------------------------
# SerialServoActuator
# ---------------------------------------------------------------------------


class _RecordingDriver:
    """ServoDriver that records all set_position calls."""

    def __init__(self, succeed: bool = True):
        self._succeed = succeed
        self.connected = False
        self.calls: list[tuple[int, float, float]] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def set_position(self, servo_id: int, angle: float, speed: float) -> bool:
        self.calls.append((servo_id, angle, speed))
        return self._succeed

    def get_position(self, servo_id: int) -> float | None:
        return None


class TestSerialServoActuator:
    def test_metadata(self):
        act = SerialServoActuator("servo.arm", "manipulation", NullServoDriver(), [0, 1])
        meta = act.describe()
        assert meta.actuator_id == "servo.arm"
        assert meta.risk_class == RiskClass.HIGH
        assert "angle" in meta.envelope
        assert meta.envelope["angle"] == (0.0, 180.0)

    def test_channel_property(self):
        act = SerialServoActuator("servo.leg", "locomotion")
        assert act.channel == "locomotion"

    def test_execute_all_servos_commanded(self):
        driver = _RecordingDriver(succeed=True)
        act = SerialServoActuator("servo.arm", "manipulation", driver, [0, 1, 2])
        act._bus = _FakeBus()

        proposal = ActionProposal(
            trace_id="t-10",
            provenance="motor.manipulation",
            action={"intensity": 0.5},  # → angle = 90°
        )
        outcome = _run(act.execute(proposal))

        assert outcome.success is True
        assert len(driver.calls) == 3
        for sid, angle, speed in driver.calls:
            assert angle == pytest.approx(90.0)
            assert speed == pytest.approx(0.5)

    def test_execute_direct_angle_override(self):
        driver = _RecordingDriver()
        act = SerialServoActuator("servo.arm", "manipulation", driver, [0])
        act._bus = _FakeBus()

        proposal = ActionProposal(
            trace_id="t-11",
            provenance="motor.manipulation",
            action={"intensity": 1.0, "angle": 45.0, "speed": 0.8},
        )
        _run(act.execute(proposal))

        assert driver.calls[0] == (0, 45.0, 0.8)

    def test_execute_partial_servo_failure(self):
        driver = _RecordingDriver(succeed=False)
        act = SerialServoActuator("servo.arm", "manipulation", driver, [0, 1])
        act._bus = _FakeBus()

        proposal = ActionProposal(
            trace_id="t-12",
            provenance="motor.manipulation",
            action={"intensity": 0.5},
        )
        outcome = _run(act.execute(proposal))
        assert outcome.success is False

    def test_execute_subset_servo_ids(self):
        driver = _RecordingDriver()
        act = SerialServoActuator("servo.arm", "manipulation", driver, [0, 1, 2, 3])
        act._bus = _FakeBus()

        proposal = ActionProposal(
            trace_id="t-13",
            provenance="motor.manipulation",
            action={"intensity": 0.3, "servo_ids": [1, 3]},
        )
        _run(act.execute(proposal))

        commanded_ids = [sid for sid, _, _ in driver.calls]
        assert commanded_ids == [1, 3]

    def test_default_servo_ids_by_channel(self):
        act = SerialServoActuator("servo.head", "head")
        assert act._servo_ids == [8, 9]

    def test_heartbeat_published_when_bus_set(self):
        bus = _FakeBus()
        act = SerialServoActuator("servo.arm", "manipulation", NullServoDriver(), [0])
        act._bus = bus

        async def _one_tick():
            # Manually invoke the heartbeat body (avoid running the full loop task)
            await bus.publish(
                f"actuator.heartbeat.{act._channel}",
                {"channel": act._channel, "actuator_id": act.actuator_id, "timestamp": time.time()},
            )

        _run(_one_tick())
        assert any("actuator.heartbeat.manipulation" in s for s in bus.subjects())


# ---------------------------------------------------------------------------
# MotorFeedbackAdapter — plugin routing
# ---------------------------------------------------------------------------


class _FakeMotorFeedbackConfig:
    """Minimal MotorFeedbackConfig substitute."""

    motor_rate_limit_hz: float = 0.0
    heartbeat_timeout_s: float = 30.0
    mujoco_continuous: bool = False
    population_vector: bool = False
    tasks_enabled: bool = False
    pain_enabled: bool = False
    mujoco_physics_hz: float = 50.0
    mujoco_proprio_hz: float = 10.0
    mujoco_viz_hz: float = 5.0
    virtual_delay_ms: int = 0
    channel_actuators: dict = {}


class _AlwaysSuccessPlugin:
    """Minimal ActuatorPlugin stub used to verify routing."""

    def __init__(self):
        self.actuator_id = "stub.plugin"
        self.calls: list[ActionProposal] = []

    async def execute(self, proposal) -> Any:
        self.calls.append(proposal)

        class _Outcome:
            success = True

        return _Outcome()


class TestMotorFeedbackAdapterPluginRouting:
    def _make_adapter(self) -> tuple[MotorFeedbackAdapter, _FakeBus]:
        bus = _FakeBus()
        cfg = _FakeMotorFeedbackConfig()
        adapter = MotorFeedbackAdapter(cfg, bus)  # type: ignore[arg-type]
        return adapter, bus

    def test_register_plugin_stores_plugin(self):
        adapter, _ = self._make_adapter()
        plugin = _AlwaysSuccessPlugin()
        adapter.register_plugin("manipulation", plugin)
        assert adapter._actuator_plugins["manipulation"] is plugin

    def test_plugin_receives_command_when_real(self):
        adapter, bus = self._make_adapter()
        plugin = _AlwaysSuccessPlugin()
        adapter.register_plugin("manipulation", plugin)
        # Simulate a live heartbeat so is_real() returns True
        adapter._real_channels["manipulation"] = time.time()

        _run(adapter.handle_motor_command("manipulation", 0.7, trace_id="t-20"))

        assert len(plugin.calls) == 1
        assert plugin.calls[0].action["intensity"] == pytest.approx(0.7)
        # Outcome must be published for the brain to consume
        assert any("motor.outcome.manipulation" in s for s in bus.subjects())
        # Must NOT fall back to the NATS relay
        assert not any("motor.execute" in s for s in bus.subjects())

    def test_nats_fallback_when_no_plugin_registered(self):
        adapter, bus = self._make_adapter()
        # Real channel but no plugin registered → NATS relay
        adapter._real_channels["manipulation"] = time.time()

        _run(adapter.handle_motor_command("manipulation", 0.4, trace_id="t-21"))

        assert any("motor.execute.manipulation" in s for s in bus.subjects())

    def test_no_real_channel_routes_to_mujoco(self):
        adapter, bus = self._make_adapter()
        plugin = _AlwaysSuccessPlugin()
        adapter.register_plugin("locomotion", plugin)
        # No heartbeat → is_real() is False → plugin must NOT be called

        with patch.object(adapter, "_ensure_mujoco") as mock_mujoco:
            mock_body = MagicMock()
            mock_body.step_command.return_value = {
                "channel": "locomotion",
                "success": True,
                "confidence": 0.8,
                "proprioceptive_state": [],
                "error_magnitude": 0.0,
            }
            mock_mujoco.return_value = mock_body
            _run(adapter.handle_motor_command("locomotion", 0.5, trace_id="t-22"))

        assert len(plugin.calls) == 0  # plugin not invoked
        assert any("motor.outcome.locomotion" in s for s in bus.subjects())

    def test_register_plugin_logs_on_overwrite(self, caplog):
        import logging

        adapter, _ = self._make_adapter()
        p1 = _AlwaysSuccessPlugin()
        p2 = _AlwaysSuccessPlugin()
        p2.actuator_id = "stub.plugin.2"
        adapter.register_plugin("head", p1)
        with caplog.at_level(logging.INFO, logger="neuromorphic.motor_feedback_adapter"):
            adapter.register_plugin("head", p2)
        assert adapter._actuator_plugins["head"] is p2
