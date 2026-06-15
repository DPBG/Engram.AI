"""Tests for the Kernel code-proposal evaluator (Phase 1.5 + governance gate)."""

from kernel.evaluator import KernelEvaluator
from activelearning import KernelDecisionType as DecisionType


def _ev():
    return KernelEvaluator()


def _proposal(target="/data/plugins/p.py", preview="x = 1"):
    return {"trace_id": "t", "target_path": target, "code_preview": preview}


def test_protected_path_denied():
    d = _ev().evaluate_code_proposal(_proposal(target="/kernel/evaluator.py"))
    assert d.type == DecisionType.DENY


def test_protected_path_safety_supervisor_denied():
    d = _ev().evaluate_code_proposal(_proposal(target="/safety-supervisor/analyzer.py"))
    assert d.type == DecisionType.DENY


def test_self_referential_code_denied_not_deferred():
    # Code touching the safety/meta machinery must be DENIED (fail closed),
    # never merely deferred.
    d = _ev().evaluate_code_proposal(_proposal(preview="import kernel.evaluator as k"))
    assert d.type == DecisionType.DENY


def test_dangerous_pattern_defers():
    d = _ev().evaluate_code_proposal(_proposal(preview="subprocess.run(['ls'])"))
    assert d.type == DecisionType.DEFER


def test_clean_code_allowed():
    d = _ev().evaluate_code_proposal(_proposal(preview="def add(a, b):\n    return a + b\n"))
    assert d.type == DecisionType.ALLOW
    # ALLOW decisions carry a TTL so stale approvals can't be replayed forever.
    assert d.expires_at is not None
