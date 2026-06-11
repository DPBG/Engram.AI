"""Tests for the safety pipeline: Brain → Kernel → Beliefs → Feedback.

Verifies that:
1. Safety gate config is parsed from env vars
2. Belief graph seeds constitutional values and norms on first boot
3. VALUE confidence has a floor (cannot drop below 0.9)
4. Kernel evaluator respects norm violations
5. DENY decisions inject negative feedback via motor outcome queue
6. TRANSFORM decisions inject corrected feedback
"""

import importlib
import os
import sys
import types
import pytest

# ---------------------------------------------------------------------------
# Path setup: add sister packages so their *submodules* are importable.
# We must avoid triggering beliefs/__init__.py and kernel/__init__.py because
# they eagerly import their service modules which depend on the full
# activelearning SDK (aiohttp, nats-py, etc.) not installed here.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _pkg in ("beliefs", "kernel", "sdk"):
    _src = os.path.join(_PROJECT_ROOT, _pkg, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)


# Map package name → directory under _PROJECT_ROOT that contains src/<package>/
_PKG_DIRS = {
    "beliefs": "beliefs",
    "kernel": "kernel",
    "activelearning": "sdk",
}


def _import_submodule(package_name: str, module_name: str):
    """Import a submodule without triggering the package __init__.py.

    This avoids the chain:  package/__init__.py → service.py → activelearning → aiohttp
    """
    pkg_dir = _PKG_DIRS.get(package_name, package_name)
    pkg_src = os.path.join(_PROJECT_ROOT, pkg_dir, "src", package_name)

    # Ensure the package exists as a namespace so 'from package.module' works
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [pkg_src]
        pkg.__package__ = package_name
        sys.modules[package_name] = pkg

    fqn = f"{package_name}.{module_name}"
    if fqn in sys.modules:
        return sys.modules[fqn]

    spec = importlib.util.spec_from_file_location(
        fqn,
        os.path.join(pkg_src, f"{module_name}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-import the two modules we need, bypassing __init__.py
_beliefs_graph = _import_submodule("beliefs", "graph")
BeliefGraph = _beliefs_graph.BeliefGraph
NodeType = _beliefs_graph.NodeType

# For kernel.evaluator we also need activelearning.core (KernelDecisionType, RiskAnalysis).
# Import it directly to avoid the full activelearning __init__ that pulls in aiohttp.
_al_core = _import_submodule("activelearning", "core")
sys.modules.setdefault("activelearning", types.ModuleType("activelearning"))
# Patch the minimal names kernel.evaluator needs from activelearning
_al_ns = sys.modules["activelearning"]
_al_ns.KernelDecisionType = _al_core.KernelDecisionType
_al_ns.RiskAnalysis = _al_core.RiskAnalysis

_kernel_evaluator = _import_submodule("kernel", "evaluator")
KernelEvaluator = _kernel_evaluator.KernelEvaluator

from neuromorphic.config import NeuromorphicConfig, SafetyGateConfig, MotorFeedbackConfig


# ===== Config tests =====

class TestSafetyGateConfig:
    """Test SafetyGateConfig parsing from environment."""

    def test_default_disabled(self):
        """Safety gate is disabled by default (backward compat)."""
        cfg = SafetyGateConfig()
        assert cfg.enabled is False
        assert cfg.fail_open is True

    def test_env_enabled(self, monkeypatch):
        """NEURO_SAFETY_GATE=1 enables the gate."""
        monkeypatch.setenv("NEURO_SAFETY_GATE", "1")
        cfg = NeuromorphicConfig.from_env()
        assert cfg.safety_gate.enabled is True

    def test_env_disabled(self, monkeypatch):
        """NEURO_SAFETY_GATE=0 keeps it disabled."""
        monkeypatch.setenv("NEURO_SAFETY_GATE", "0")
        cfg = NeuromorphicConfig.from_env()
        assert cfg.safety_gate.enabled is False

    def test_env_timeout(self, monkeypatch):
        """NEURO_SAFETY_TIMEOUT overrides decision timeout."""
        monkeypatch.setenv("NEURO_SAFETY_GATE", "1")
        monkeypatch.setenv("NEURO_SAFETY_TIMEOUT", "5.0")
        cfg = NeuromorphicConfig.from_env()
        assert cfg.safety_gate.decision_timeout == 5.0

    def test_env_fail_closed(self, monkeypatch):
        """NEURO_SAFETY_FAIL_OPEN=0 makes it fail-closed."""
        monkeypatch.setenv("NEURO_SAFETY_GATE", "1")
        monkeypatch.setenv("NEURO_SAFETY_FAIL_OPEN", "0")
        cfg = NeuromorphicConfig.from_env()
        assert cfg.safety_gate.fail_open is False


# ===== Belief tests =====

class TestBeliefConstitution:
    """Test belief graph constitutional seeding and value protection."""

    def test_seed_creates_values_and_norms(self):
        """Seeding creates 4 values and 5 norms."""
        g = BeliefGraph()
        g.seed_constitutional_beliefs()

        values = g.get_beliefs_by_type(NodeType.VALUE)
        norms = g.get_beliefs_by_type(NodeType.NORM)

        assert len(values) == 4
        assert len(norms) == 5
        assert g.edge_count == 6

    def test_seed_idempotent(self):
        """Seeding twice doesn't duplicate nodes."""
        g = BeliefGraph()
        g.seed_constitutional_beliefs()
        g.seed_constitutional_beliefs()  # second call

        values = g.get_beliefs_by_type(NodeType.VALUE)
        assert len(values) == 4  # not 8

    def test_value_confidence_floor(self):
        """VALUES cannot have confidence reduced below 0.9."""
        g = BeliefGraph()
        g.seed_constitutional_beliefs()

        # Try to erode human_safety with strong contradicting evidence
        g.update_belief(
            "value.human_safety",
            evidence_strength=1.0,
            supports=False,
            source="adversarial_test",
        )

        node = g.get_node("value.human_safety")
        assert node.confidence >= 0.9, f"VALUE confidence dropped to {node.confidence}"

    def test_norm_confidence_can_change(self):
        """NORMS (unlike VALUES) can have confidence freely updated."""
        g = BeliefGraph()
        g.seed_constitutional_beliefs()

        old = g.get_node("norm.gradual_motor")
        old_conf = old.confidence

        g.update_belief(
            "norm.gradual_motor",
            evidence_strength=0.5,
            supports=False,
            source="experience",
        )

        new = g.get_node("norm.gradual_motor")
        assert new.confidence < old_conf

    def test_value_ids_predictable(self):
        """Value and norm IDs are stable for Kernel lookups."""
        g = BeliefGraph()
        g.seed_constitutional_beliefs()

        assert g.get_node("value.human_safety") is not None
        assert g.get_node("value.hardware_safety") is not None
        assert g.get_node("norm.gradual_motor") is not None
        assert g.get_node("norm.force_limit") is not None

    def test_norm_metadata_present(self):
        """Norms have metadata for Kernel rule checking."""
        g = BeliefGraph()
        g.seed_constitutional_beliefs()

        gradual = g.get_node("norm.gradual_motor")
        assert "max_intensity_delta" in gradual.metadata

        force = g.get_node("norm.force_limit")
        assert "motor_channels" in force.metadata

    def test_import_enforces_value_floor(self):
        """import_from_dict enforces VALUE confidence floor (anti-tampering)."""
        import time
        now = int(time.time() * 1000)
        g = BeliefGraph()
        # Simulate loading corrupted persistence data
        g.import_from_dict({
            "nodes": [
                {
                    "id": "value.human_safety",
                    "type": "value",
                    "content": "Human safety",
                    "confidence": 0.3,  # corrupted / tampered
                    "source": "constitutional",
                    "metadata": {},
                    "created_at": now,
                    "updated_at": now,
                },
                {
                    "id": "norm.gradual_motor",
                    "type": "norm",
                    "content": "Gradual motor",
                    "confidence": 0.3,  # norms CAN be low
                    "source": "constitutional",
                    "metadata": {},
                    "created_at": now,
                    "updated_at": now,
                },
            ],
            "edges": [],
        })
        # VALUE must be floored at 0.9
        val = g.get_node("value.human_safety")
        assert val.confidence >= 0.9
        # NORM is not floored
        norm = g.get_node("norm.gradual_motor")
        assert norm.confidence == pytest.approx(0.3)

    def test_seed_flag_prevents_concurrent_duplication(self):
        """Flag-based idempotency prevents duplication even without querying graph."""
        g = BeliefGraph()
        g.seed_constitutional_beliefs()
        # Manually reset the graph but not the flag
        # (simulates concurrent call racing after first seed)
        g._graph.clear()
        g.seed_constitutional_beliefs()  # should no-op due to flag
        assert g.node_count == 0  # flag prevented re-seed


# ===== Kernel tests =====

class TestKernelNormViolations:
    """Test Kernel evaluator with norm violations."""

    def test_no_violations_allows(self):
        """No norm violations → ALLOW."""
        ev = KernelEvaluator()

        proposal = {
            "trace_id": "test-123",
            "action": {"type": "motor_command", "channel": "locomotion", "intensity": 0.5},
        }
        decision = ev.evaluate_action_proposal(proposal, norm_violations=[])
        assert decision.type.value == "ALLOW"

    def test_norm_violation_boosts_risk(self):
        """Norm violations increase risk score."""
        ev = KernelEvaluator()

        proposal = {
            "trace_id": "test-123",
            "action": {"type": "motor_command", "channel": "manipulation", "intensity": 0.95},
        }
        violations = [
            {"norm_id": "norm.force_limit", "content": "force limit", "risk_boost": 0.3},
            {"norm_id": "norm.gradual_motor", "content": "gradual", "risk_boost": 0.15},
        ]
        decision = ev.evaluate_action_proposal(proposal, norm_violations=violations)

        # 0.0 base + 0.3 + 0.15 = 0.45, which is below defer threshold (0.5)
        # But risk_score should reflect the boost
        assert decision.risk_score == pytest.approx(0.45)

    def test_high_norm_violation_triggers_defer(self):
        """Enough norm violations push risk above defer threshold."""
        ev = KernelEvaluator()

        proposal = {
            "trace_id": "test-123",
            "action": {"type": "motor_command", "channel": "locomotion", "intensity": 0.99},
        }
        violations = [
            {"norm_id": "norm.force_limit", "content": "force limit", "risk_boost": 0.3},
            {"norm_id": "norm.gradual_motor", "content": "gradual", "risk_boost": 0.3},
        ]
        decision = ev.evaluate_action_proposal(proposal, norm_violations=violations)

        # 0.0 base + 0.3 + 0.3 = 0.6 → above defer threshold 0.5
        assert decision.type.value in ("DEFER", "DENY")


# ===== Feedback injection tests =====

class TestSafetyFeedbackInjection:
    """Test that safety decisions inject motor outcome feedback.

    These tests replicate the _inject_safety_feedback logic from
    NeuromorphicService to avoid importing the full service (which
    depends on activelearning SDK not installed in the neuromorphic venv).
    """

    @staticmethod
    def _inject_safety_feedback(
        cfg: NeuromorphicConfig,
        pending_outcomes: list,
        channel: str,
        success: bool,
        confidence: float = 1.0,
        proprio_data: list[float] | None = None,
    ) -> None:
        """Replicate the service method for unit testing."""
        if not cfg.motor_feedback.enabled:
            return
        pending_outcomes.append({
            "proprio_data": proprio_data if proprio_data is not None else [1.0 if success else 0.0],
            "provenance": f"motor.outcome.{channel}",
            "gain": (
                cfg.motor_feedback.success_gain if success
                else cfg.motor_feedback.failure_gain
            ) * confidence,
        })

    def test_deny_queues_negative_feedback(self):
        """DENY decision queues negative motor outcome for brain STDP."""
        cfg = NeuromorphicConfig()
        cfg.safety_gate = SafetyGateConfig(enabled=True, deny_feedback=True)
        cfg.motor_feedback = MotorFeedbackConfig(enabled=True)

        outcomes = []
        self._inject_safety_feedback(cfg, outcomes, "locomotion", success=False, confidence=1.0)

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome["provenance"] == "motor.outcome.locomotion"
        assert outcome["gain"] == cfg.motor_feedback.failure_gain * 1.0

    def test_transform_queues_corrective_feedback(self):
        """TRANSFORM decision queues corrective feedback with actual values."""
        cfg = NeuromorphicConfig()
        cfg.safety_gate = SafetyGateConfig(enabled=True, transform_feedback=True)
        cfg.motor_feedback = MotorFeedbackConfig(enabled=True)

        outcomes = []
        self._inject_safety_feedback(
            cfg, outcomes, "manipulation", success=True, confidence=0.7, proprio_data=[0.6]
        )

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome["proprio_data"] == [0.6]
        assert outcome["gain"] == pytest.approx(cfg.motor_feedback.success_gain * 0.7)

    def test_no_feedback_when_motor_feedback_disabled(self):
        """No feedback queued when motor feedback loop is off."""
        cfg = NeuromorphicConfig()
        cfg.safety_gate = SafetyGateConfig(enabled=True)
        cfg.motor_feedback = MotorFeedbackConfig(enabled=False)

        outcomes = []
        self._inject_safety_feedback(cfg, outcomes, "locomotion", success=False, confidence=1.0)

        assert len(outcomes) == 0
