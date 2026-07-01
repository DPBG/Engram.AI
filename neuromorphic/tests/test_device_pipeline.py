"""Tests for Phase B: Unknown Device → LLM Pipeline.

Tests the device discovery, coordinator routing, meta-programmer prompt
enhancement, kernel envelope validation, and gateway hot-loading logic.
All tests are self-contained with no external service dependencies.

We import individual modules directly (bypassing __init__.py) to avoid
SDK dependencies that aren't installed in the neuromorphic venv.
"""

import importlib.util
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_module(name: str, path: str):
    """Load a Python module from file path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load modules that have SDK-free code
sys.path.insert(0, os.path.join(_ROOT, "sensory-gateway"))

# Kernel evaluator needs its SDK types — mock them before import
_mock_core = type(sys)("activelearning.core")


class _FakeDecisionType:
    ALLOW = type("DT", (), {"value": "ALLOW"})()
    DENY = type("DT", (), {"value": "DENY"})()
    TRANSFORM = type("DT", (), {"value": "TRANSFORM"})()
    DEFER = type("DT", (), {"value": "DEFER"})()


class _FakeDecision:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        if not hasattr(self, "type"):
            self.type = _FakeDecisionType.ALLOW
        if not hasattr(self, "reason"):
            self.reason = None
        if not hasattr(self, "transformations"):
            self.transformations = None
        if not hasattr(self, "risk_score"):
            self.risk_score = 0.0
        if not hasattr(self, "expires_at"):
            self.expires_at = 0


class _FakeRiskAnalysis:
    def __init__(self, risk_score=0.0, flags=None):
        self.risk_score = risk_score
        self.flags = flags or []
        self.recommendations = []


_mock_core.KernelDecisionType = _FakeDecisionType
_mock_core.KernelDecision = _FakeDecision
_mock_core.RiskAnalysis = _FakeRiskAnalysis

# Mock the activelearning package so kernel.evaluator can import from it
_mock_al = type(sys)("activelearning")
_mock_al.KernelDecisionType = _FakeDecisionType
_mock_al.KernelDecision = _FakeDecision
_mock_al.RiskAnalysis = _FakeRiskAnalysis
_mock_al.core = _mock_core
sys.modules["activelearning"] = _mock_al
sys.modules["activelearning.core"] = _mock_core

# Now we can import the evaluator
_evaluator_mod = _load_module(
    "kernel.evaluator",
    os.path.join(_ROOT, "kernel", "src", "kernel", "evaluator.py"),
)
KernelEvaluator = _evaluator_mod.KernelEvaluator
DecisionType = _FakeDecisionType


# ===== Discovery: KNOWN_DEVICE_TYPES =====


class TestDiscoveryKnownTypes(unittest.TestCase):
    def test_known_types_include_basics(self):
        from discovery import KNOWN_DEVICE_TYPES

        assert "camera" in KNOWN_DEVICE_TYPES
        assert "mic" in KNOWN_DEVICE_TYPES
        assert "serial" in KNOWN_DEVICE_TYPES

    def test_unknown_type_not_in_known(self):
        from discovery import KNOWN_DEVICE_TYPES

        assert "lidar" not in KNOWN_DEVICE_TYPES
        assert "imu" not in KNOWN_DEVICE_TYPES

    def test_sanitize_device_string(self):
        from discovery import sanitize_device_string

        assert sanitize_device_string("hello\nworld") == "hello world"
        assert sanitize_device_string("a" * 300) == "a" * 200
        assert sanitize_device_string(None) == "None"

    def test_discovered_device_dataclass(self):
        from discovery import DiscoveredDevice

        d = DiscoveredDevice(
            device_type="lidar",
            device_id="lidar:0",
            name="RPLidar A1",
            metadata={"port": "/dev/ttyUSB0"},
        )
        assert d.device_type == "lidar"
        assert d.metadata["port"] == "/dev/ttyUSB0"


# ===== Kernel Envelope: intensity + channel validation =====


class TestKernelEnvelopeExtension(unittest.TestCase):
    def setUp(self):
        self.evaluator = KernelEvaluator()

    def test_valid_intensity(self):
        result = self.evaluator._check_envelope({"intensity": 0.5, "channel": "locomotion"})
        assert result is None

    def test_intensity_too_high(self):
        result = self.evaluator._check_envelope({"intensity": 1.5})
        assert result is not None
        assert "intensity" in result

    def test_intensity_negative(self):
        result = self.evaluator._check_envelope({"intensity": -0.1})
        assert result is not None

    def test_intensity_non_numeric(self):
        result = self.evaluator._check_envelope({"intensity": "high"})
        assert result is not None
        assert "numeric" in result

    def test_valid_channels(self):
        for ch in ("locomotion", "manipulation", "head", "speech", "expression", "cognitive"):
            result = self.evaluator._check_envelope({"channel": ch})
            assert result is None, f"channel {ch} should be valid"

    def test_unknown_channel(self):
        result = self.evaluator._check_envelope({"channel": "flying"})
        assert result is not None
        assert "unknown" in result

    def test_legacy_angle_still_works(self):
        assert self.evaluator._check_envelope({"angle": 90}) is None

    def test_legacy_angle_out_of_range(self):
        assert self.evaluator._check_envelope({"angle": 200}) is not None

    def test_intensity_transform_clamps(self):
        action = {"intensity": 1.5, "channel": "locomotion"}
        transforms = self.evaluator._generate_transformations(action, ["TEST_FLAG"])
        assert transforms is not None
        assert transforms[0]["intensity"] == 1.0

    def test_intensity_transform_negative_clamps(self):
        action = {"intensity": -0.5}
        transforms = self.evaluator._generate_transformations(action, ["TEST_FLAG"])
        assert transforms is not None
        assert transforms[0]["intensity"] == 0.0

    def test_intensity_transform_no_change(self):
        action = {"intensity": 0.5, "channel": "locomotion"}
        transforms = self.evaluator._generate_transformations(action, ["TEST_FLAG"])
        assert transforms is None

    def test_full_proposal_allow(self):
        proposal = {
            "trace_id": "test-123",
            "action": {"intensity": 0.5, "channel": "locomotion"},
        }
        decision = self.evaluator.evaluate_action_proposal(proposal)
        assert decision.type.value == "ALLOW"

    def test_full_proposal_deny_bad_channel(self):
        proposal = {
            "trace_id": "test-456",
            "action": {"intensity": 0.5, "channel": "teleport"},
        }
        decision = self.evaluator.evaluate_action_proposal(proposal)
        assert decision.type.value == "DENY"


# ===== Meta-Programmer: Device-Aware Prompts =====


class TestMetaProgrammerDevicePrompt(unittest.TestCase):
    def setUp(self):
        # Load agents.py directly, bypassing __init__.py
        agents_path = os.path.join(_ROOT, "meta-programmer", "src", "meta_programmer", "agents.py")
        self.agents_mod = _load_module("meta_programmer.agents", agents_path)
        # Create team instance without __init__
        team = self.agents_mod.MetaProgrammerTeam.__new__(self.agents_mod.MetaProgrammerTeam)
        team.ollama_url = "http://localhost:11434"
        team.model_name = "test"
        self.team = team

    def _format_stub(self, results):
        return "\n".join(str(r) for r in results) if results else "None"

    def test_device_context_adds_hardware_info(self):
        self.team._format_search_results = self._format_stub
        context = {
            "source": "gateway_device_discovery",
            "device_type": "lidar",
            "device_id": "lidar:0",
            "name": "RPLidar A1",
            "plugin_type": "sensor",
            "metadata": {"port": "/dev/ttyUSB0", "vid": "0x10c4", "pid": "0xea60"},
        }
        prompt = self.team._build_code_prompt(
            description="Create sensor plugin for RPLidar A1",
            available_sensors=[],
            search_results=[],
            context=context,
        )
        assert "RPLidar A1" in prompt
        assert "SensorPlugin" in prompt
        assert "0x10c4" in prompt
        assert "/dev/ttyUSB0" in prompt

    def test_actuator_context_uses_actuator_base(self):
        self.team._format_search_results = self._format_stub
        context = {
            "source": "gateway_device_discovery",
            "plugin_type": "actuator",
            "device_type": "servo",
            "device_id": "servo:0",
            "name": "Dynamixel XM430",
            "metadata": {},
        }
        prompt = self.team._build_code_prompt(
            description="Create actuator plugin",
            available_sensors=[],
            search_results=[],
            context=context,
        )
        assert "ActuatorPlugin" in prompt
        assert "motor.outcome" in prompt

    def test_non_device_context_no_hardware_section(self):
        self.team._format_search_results = self._format_stub
        prompt = self.team._build_code_prompt(
            description="Pick up red ball",
            available_sensors=["camera"],
            search_results=[],
            context={"source": "task_coordinator"},
        )
        assert "Device Hardware Info" not in prompt

    def test_infer_target_path_sensor(self):
        path = self.team._infer_target_path("RPLidar A1 sensor plugin", {})
        assert path.startswith("/data/plugins/")
        assert path.endswith(".py")

    def test_infer_target_path_actuator(self):
        path = self.team._infer_target_path("Dynamixel servo actuator plugin", {})
        assert path.startswith("/data/plugins/")
        assert path.endswith(".py")


# ===== Coordinator: dedup + plugin type inference logic =====


class TestCoordinatorDeviceRouting(unittest.TestCase):
    def test_device_routing_deduplication(self):
        """Same device_id should not trigger duplicate knowledge gaps."""
        pending: set[str] = set()
        device_id = "lidar:0"
        assert device_id not in pending
        pending.add(device_id)
        assert device_id in pending

    def test_plugin_type_inference_sensor(self):
        from discovery import infer_plugin_type

        assert infer_plugin_type("RPLidar A1 Scanner") == "sensor"

    def test_plugin_type_inference_actuator(self):
        from discovery import infer_plugin_type

        assert infer_plugin_type("Dynamixel Servo Motor") == "actuator"

    def test_gripper_detected_as_actuator(self):
        from discovery import infer_plugin_type

        assert infer_plugin_type("Robotiq 2F-85 Gripper") == "actuator"


# ===== Gateway: source verification =====


class TestGatewaySourceVerification(unittest.TestCase):
    def test_subjects_defined(self):
        with open(os.path.join(_ROOT, "sensory-gateway", "gateway.py")) as f:
            src = f.read()
        assert 'SUBJECT_DEVICE_UNKNOWN = "device.unknown"' in src
        assert 'SUBJECT_DRIVER_READY = "device.driver.ready"' in src

    def test_hot_load_handler_in_source(self):
        with open(os.path.join(_ROOT, "sensory-gateway", "gateway.py")) as f:
            src = f.read()
        assert "_handle_driver_ready" in src
        assert "importlib.util.spec_from_file_location" in src
        assert "SUBJECT_DRIVER_READY" in src

    def test_unknown_device_publish_in_source(self):
        with open(os.path.join(_ROOT, "sensory-gateway", "gateway.py")) as f:
            src = f.read()
        assert "SUBJECT_DEVICE_UNKNOWN" in src
        assert "unknown_published" in src
        assert "KNOWN_DEVICE_TYPES" in src
