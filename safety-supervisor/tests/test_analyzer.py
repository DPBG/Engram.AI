"""Unit tests for RiskAnalyzer — risk heuristics, envelope checks, and output structure.

Tests cover:
  - RiskAnalysis output structure and trace_id preservation
  - Action-proposal scoring (low / elevated / high categories)
  - Envelope boundary checks (angle, speed)
  - Code-proposal protected-path detection
  - Dangerous-pattern and import detection
  - AST-level introspection and comprehension analysis
  - Syntax-error handling
"""

from __future__ import annotations

import pytest
from activelearning import RiskAnalysis

from safety_supervisor.analyzer import RiskAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _analyzer() -> RiskAnalyzer:
    return RiskAnalyzer()


def _action_proposal(action_type: str = "observe", **action_fields: object) -> dict:
    return {
        "trace_id": "trace-action-001",
        "action": {"type": action_type, **action_fields},
    }


def _code_proposal(
    target_path: str = "/data/plugins/sensor.py",
    code_preview: str = "x = 1",
) -> dict:
    return {
        "trace_id": "trace-code-001",
        "target_path": target_path,
        "code_preview": code_preview,
    }


# ---------------------------------------------------------------------------
# RiskAnalysis output structure
# ---------------------------------------------------------------------------


def test_analyze_action_returns_risk_analysis() -> None:
    result = _analyzer().analyze_action(_action_proposal())
    assert isinstance(result, RiskAnalysis)


def test_analyze_action_preserves_trace_id() -> None:
    result = _analyzer().analyze_action(_action_proposal())
    assert result.trace_id == "trace-action-001"


def test_analyze_action_fields_are_correct_types() -> None:
    result = _analyzer().analyze_action(_action_proposal())
    assert isinstance(result.risk_score, float)
    assert isinstance(result.flags, list)
    assert isinstance(result.recommendations, list)
    assert isinstance(result.details, dict)


def test_analyze_code_returns_risk_analysis() -> None:
    result = _analyzer().analyze_code(_code_proposal())
    assert isinstance(result, RiskAnalysis)


def test_analyze_code_preserves_trace_id() -> None:
    result = _analyzer().analyze_code(_code_proposal())
    assert result.trace_id == "trace-code-001"


# ---------------------------------------------------------------------------
# Action proposal — low-risk (no flags)
# ---------------------------------------------------------------------------


def test_safe_action_type_zero_risk() -> None:
    result = _analyzer().analyze_action(_action_proposal("observe"))
    assert result.risk_score == pytest.approx(0.0)
    assert result.flags == []


def test_empty_action_type_flagged_as_unknown() -> None:
    # Fail-closed: an empty action type cannot be categorized → UNKNOWN_ACTION_TYPE.
    result = _analyzer().analyze_action(_action_proposal(""))
    assert "UNKNOWN_ACTION_TYPE" in result.flags
    assert result.risk_score >= 0.1


# ---------------------------------------------------------------------------
# Action proposal — high-risk action types (score += 0.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action_type", ["shutdown", "delete", "reset", "restart", "format"])
def test_high_risk_action_type_flag_and_score(action_type: str) -> None:
    result = _analyzer().analyze_action(_action_proposal(action_type))
    assert "HIGH_RISK_ACTION" in result.flags
    assert result.risk_score == pytest.approx(0.5)


def test_high_risk_action_type_case_insensitive() -> None:
    result = _analyzer().analyze_action(_action_proposal("SHUTDOWN"))
    assert "HIGH_RISK_ACTION" in result.flags


def test_high_risk_action_adds_recommendation() -> None:
    result = _analyzer().analyze_action(_action_proposal("delete"))
    assert any("delete" in r.lower() for r in result.recommendations)


# ---------------------------------------------------------------------------
# Action proposal — medium-risk action types (score += 0.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action_type", ["move", "execute", "run", "deploy"])
def test_medium_risk_action_type_flag_and_score(action_type: str) -> None:
    result = _analyzer().analyze_action(_action_proposal(action_type))
    assert "MEDIUM_RISK_ACTION" in result.flags
    assert result.risk_score == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Action proposal — envelope checks: angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [181, -1, 360, -90])
def test_out_of_range_angle_sets_flag(angle: float) -> None:
    result = _analyzer().analyze_action(_action_proposal(angle=angle))
    assert "UNSAFE_ANGLE" in result.flags
    assert result.risk_score >= 0.3


@pytest.mark.parametrize("angle", [0, 90, 180])
def test_boundary_angles_are_safe(angle: float) -> None:
    result = _analyzer().analyze_action(_action_proposal(angle=angle))
    assert "UNSAFE_ANGLE" not in result.flags


# ---------------------------------------------------------------------------
# Action proposal — envelope checks: speed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("speed", [101, -1, 200])
def test_out_of_range_speed_sets_flag(speed: float) -> None:
    result = _analyzer().analyze_action(_action_proposal(speed=speed))
    assert "UNSAFE_SPEED" in result.flags
    assert result.risk_score >= 0.3


@pytest.mark.parametrize("speed", [0, 50, 100])
def test_boundary_speeds_are_safe(speed: float) -> None:
    result = _analyzer().analyze_action(_action_proposal(speed=speed))
    assert "UNSAFE_SPEED" not in result.flags


# ---------------------------------------------------------------------------
# Action proposal — risk accumulation
# ---------------------------------------------------------------------------


def test_high_risk_type_and_unsafe_angle_accumulate() -> None:
    result = _analyzer().analyze_action(_action_proposal("shutdown", angle=250))
    assert "HIGH_RISK_ACTION" in result.flags
    assert "UNSAFE_ANGLE" in result.flags
    assert result.risk_score == pytest.approx(0.5 + 0.3)


def test_unsafe_angle_and_speed_both_flagged() -> None:
    result = _analyzer().analyze_action(_action_proposal(angle=200, speed=150))
    assert "UNSAFE_ANGLE" in result.flags
    assert "UNSAFE_SPEED" in result.flags
    assert result.risk_score == pytest.approx(0.3 + 0.3)


# ---------------------------------------------------------------------------
# Code proposal — protected paths (score = 1.0, early return)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "/kernel/evaluator.py",
    "/safety-supervisor/analyzer.py",
    "/meta-programmer/orchestrator/main.py",
    "/meta-programmer/agents/code_gen.py",
])
def test_protected_path_max_risk(path: str) -> None:
    result = _analyzer().analyze_code(_code_proposal(target_path=path))
    assert "PROTECTED_PATH" in result.flags
    assert result.risk_score == pytest.approx(1.0)


def test_protected_path_short_circuits_analysis() -> None:
    # Dangerous code inside a protected path should not add extra flags —
    # the function returns immediately after setting PROTECTED_PATH.
    result = _analyzer().analyze_code(
        _code_proposal(target_path="/kernel/foo.py", code_preview="eval('x')")
    )
    assert result.flags == ["PROTECTED_PATH"]
    assert result.risk_score == pytest.approx(1.0)


def test_unprotected_path_not_blocked() -> None:
    result = _analyzer().analyze_code(_code_proposal(target_path="/data/plugins/sensor.py"))
    assert "PROTECTED_PATH" not in result.flags


# ---------------------------------------------------------------------------
# Code proposal — clean / safe code
# ---------------------------------------------------------------------------


def test_clean_code_zero_risk() -> None:
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    result = _analyzer().analyze_code(_code_proposal(code_preview=code))
    assert result.risk_score == pytest.approx(0.0)
    assert result.flags == []


# ---------------------------------------------------------------------------
# Code proposal — dangerous pattern detection
# ---------------------------------------------------------------------------


def test_eval_triggers_dynamic_execution() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="eval('1+1')"))
    assert "DYNAMIC_EXECUTION" in result.flags
    assert result.risk_score >= 0.5


def test_exec_triggers_dynamic_execution() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="exec(code_str)"))
    assert "DYNAMIC_EXECUTION" in result.flags


def test_compile_triggers_dynamic_execution() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="compile(src, '', 'exec')"))
    assert "DYNAMIC_EXECUTION" in result.flags


def test_os_system_triggers_shell_execution() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="os.system('ls')"))
    assert "SHELL_EXECUTION" in result.flags


def test_subprocess_attribute_triggers_subprocess_flag() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="subprocess.run(['ls'])"))
    assert "SUBPROCESS" in result.flags


def test_file_write_open_flagged() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="open('out.txt', 'w')"))
    assert "FILE_WRITE" in result.flags


def test_file_append_open_flagged() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="open('log.txt', 'a')"))
    assert "FILE_WRITE" in result.flags


def test_multiple_patterns_accumulate_risk() -> None:
    code = "eval('x'); exec('y')"
    result = _analyzer().analyze_code(_code_proposal(code_preview=code))
    # Both eval (0.5) and exec (0.5) fire → total >= 1.0
    assert result.risk_score >= 1.0


# ---------------------------------------------------------------------------
# Code proposal — import analysis
# ---------------------------------------------------------------------------


def test_import_subprocess_flagged() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="import subprocess"))
    assert "DANGEROUS_IMPORT:subprocess" in result.flags
    assert result.risk_score >= 0.5


def test_import_os_flagged() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="import os"))
    assert "DANGEROUS_IMPORT:os" in result.flags


def test_from_os_import_flagged() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="from os import path"))
    assert "DANGEROUS_IMPORT:os" in result.flags


def test_import_socket_flagged() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="import socket"))
    assert "DANGEROUS_IMPORT:socket" in result.flags


def test_safe_stdlib_import_not_flagged() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="import math"))
    assert not any(f.startswith("DANGEROUS_IMPORT") for f in result.flags)


def test_elevated_risk_combined_import_and_pattern() -> None:
    code = "import subprocess\nsubprocess.run(['ls'])"
    result = _analyzer().analyze_code(_code_proposal(code_preview=code))
    assert "DANGEROUS_IMPORT:subprocess" in result.flags
    assert "SUBPROCESS" in result.flags
    assert result.risk_score >= 0.5


# ---------------------------------------------------------------------------
# Code proposal — AST analysis
# ---------------------------------------------------------------------------


def test_dunder_code_attribute_triggers_introspection() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="x = fn.__code__"))
    assert "INTROSPECTION" in result.flags


def test_dunder_globals_attribute_triggers_introspection() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="g = fn.__globals__"))
    assert "INTROSPECTION" in result.flags


def test_dunder_dict_attribute_triggers_introspection() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="d = obj.__dict__"))
    assert "INTROSPECTION" in result.flags


def test_complex_comprehension_three_loops_flagged() -> None:
    # Three nested generators exceed the loop-count threshold (> 2).
    code = "[x for a in A for b in B for c in C]"
    result = _analyzer().analyze_code(_code_proposal(code_preview=code))
    assert "COMPLEX_COMPREHENSION" in result.flags


def test_two_loop_comprehension_not_flagged() -> None:
    # Two nested generators are exactly at the threshold (== 2), not above it.
    code = "[x for a in A for b in B]"
    result = _analyzer().analyze_code(_code_proposal(code_preview=code))
    assert "COMPLEX_COMPREHENSION" not in result.flags


def test_syntax_error_code_flagged() -> None:
    result = _analyzer().analyze_code(_code_proposal(code_preview="def (broken syntax!!!"))
    assert "SYNTAX_ERROR" in result.flags
    assert result.risk_score >= 0.1


# ---------------------------------------------------------------------------
# Malformed proposal handling — fail closed
# ---------------------------------------------------------------------------


def test_action_proposal_missing_action_key_flagged() -> None:
    result = _analyzer().analyze_action({"trace_id": "t"})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_action_proposal_none_action_value_flagged() -> None:
    result = _analyzer().analyze_action({"trace_id": "t", "action": None})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_action_proposal_non_dict_action_flagged() -> None:
    result = _analyzer().analyze_action({"trace_id": "t", "action": "shutdown"})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_action_proposal_empty_dict_flagged() -> None:
    result = _analyzer().analyze_action({})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_code_proposal_missing_target_path_flagged() -> None:
    result = _analyzer().analyze_code({"trace_id": "t", "code_preview": "x = 1"})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_code_proposal_missing_code_preview_flagged() -> None:
    result = _analyzer().analyze_code({"trace_id": "t", "target_path": "/data/plugins/foo.py"})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_code_proposal_none_target_path_flagged() -> None:
    result = _analyzer().analyze_code(
        {"trace_id": "t", "target_path": None, "code_preview": "x = 1"}
    )
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_code_proposal_none_code_preview_flagged() -> None:
    result = _analyzer().analyze_code(
        {"trace_id": "t", "target_path": "/data/plugins/foo.py", "code_preview": None}
    )
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_code_proposal_empty_dict_flagged() -> None:
    result = _analyzer().analyze_code({})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_code_proposal_blank_target_path_flagged() -> None:
    result = _analyzer().analyze_code({"trace_id": "t", "target_path": "   ", "code_preview": "x = 1"})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_code_proposal_whitespace_target_path_not_assessed() -> None:
    # A whitespace-only path must not silently produce zero risk.
    result = _analyzer().analyze_code({"trace_id": "t", "target_path": "\t\n", "code_preview": "x = 1"})
    assert result.risk_score >= 0.5


def test_action_type_non_string_flagged_as_malformed() -> None:
    # An integer type field must not raise AttributeError — fail closed instead.
    result = _analyzer().analyze_action({"trace_id": "t", "action": {"type": 1}})
    assert "MALFORMED_PROPOSAL" in result.flags
    assert result.risk_score >= 0.5


def test_action_type_padded_shutdown_flagged_high_risk() -> None:
    # Whitespace padding must not bypass HIGH_RISK_ACTION matching.
    result = _analyzer().analyze_action(_action_proposal(" shutdown "))
    assert "HIGH_RISK_ACTION" in result.flags
    assert result.risk_score >= 0.5


def test_action_type_padded_deploy_flagged_medium_risk() -> None:
    result = _analyzer().analyze_action(_action_proposal("  deploy  "))
    assert "MEDIUM_RISK_ACTION" in result.flags
