"""Tests for the Coordinator's Kernel execution gate (Phase 1.6).

Loads gate.py directly so the test doesn't import the coordinator package
(which pulls in the activelearning SDK and aiohttp). The gate logic is pure
stdlib.
"""

import importlib.util
import os
import sys

_GATE_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "coordinator", "gate.py")
_spec = importlib.util.spec_from_file_location("coord_gate", _GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["coord_gate"] = gate
_spec.loader.exec_module(gate)


# ── proposal construction ──────────────────────────────────────────────────


def test_build_execution_proposal_shape():
    p = gate.build_execution_proposal("trace-1", "task-42", {"speed": 1})
    assert p["trace_id"] == "trace-1"
    assert p["provenance"] == "coordinator"
    assert p["action"]["type"] == "task_execution"
    assert p["action"]["task_id"] == "task-42"
    assert p["action"]["parameters"] == {"speed": 1}


def test_build_execution_proposal_defaults_parameters():
    p = gate.build_execution_proposal("t", "task")
    assert p["action"]["parameters"] == {}


# ── decision gating (fail-closed contract) ─────────────────────────────────


def test_allow_permits_execution():
    assert gate.decision_allows({"type": "ALLOW"}) is True


def test_transform_permits_execution():
    assert gate.decision_allows({"type": "TRANSFORM"}) is True


def test_deny_blocks_execution():
    assert gate.decision_allows({"type": "DENY"}) is False


def test_defer_blocks_execution():
    # A deferred (human-review) decision is not an approval — do not execute.
    assert gate.decision_allows({"type": "DEFER"}) is False


def test_timeout_none_blocks_execution():
    # wait_for_decision timing out → None → fail closed.
    assert gate.decision_allows(None) is False


def test_empty_or_malformed_blocks_execution():
    assert gate.decision_allows({}) is False
    assert gate.decision_allows({"reason": "no type"}) is False
    assert gate.decision_allows("ALLOW") is False  # not a dict
    assert gate.decision_allows({"type": "allow"}) is False  # case-sensitive
