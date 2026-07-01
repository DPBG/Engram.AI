"""Evasion regressions + self-referential detection for RiskAnalyzer (E1.8.3).

The supervisor's risk signal must not be silently evaded. The regex layer can be
fooled by parenthesized names, computed attribute strings, aliased imports, and
string-built calls; these tests pin the AST-based guards that close those gaps,
plus the self-referential / self-modifying-code detection. Each test fails if
its guard is removed. Benign cases assert no false-positive over-escalation.
"""

from __future__ import annotations

import pytest

from safety_supervisor.analyzer import RiskAnalyzer


def _analyzer() -> RiskAnalyzer:
    return RiskAnalyzer()


def _analyze(code: str, target_path: str = "/data/plugins/sensor.py"):
    return _analyzer().analyze_code(
        {"trace_id": "t-evasion", "target_path": target_path, "code_preview": code}
    )


# ── self-referential / self-modifying code ───────────────────────────────────


@pytest.mark.parametrize("code", [
    "import kernel",
    "import kernel as k",
    "import meta_programmer.service",
    "from safety_supervisor.analyzer import RiskAnalyzer",
    "from beliefs.graph import BeliefGraph",
])
def test_self_referential_import_flagged(code: str) -> None:
    result = _analyze(code)
    assert "SELF_REFERENTIAL_CODE" in result.flags
    assert result.risk_score >= 0.6


def test_third_party_import_not_self_referential() -> None:
    # A module that merely starts with a safety-component-like name but isn't one.
    result = _analyze("import kerneldensity")
    assert "SELF_REFERENTIAL_CODE" not in result.flags


# ── evasion: dynamic / aliased attribute access ──────────────────────────────


def test_getattr_with_computed_name_flagged() -> None:
    # Attribute name resolved at runtime — classic regex bypass.
    result = _analyze("name = 'system'\nfn = getattr(os, name)")
    assert "DYNAMIC_ATTRIBUTE_ACCESS" in result.flags


def test_getattr_with_string_built_name_flagged() -> None:
    result = _analyze("fn = getattr(os, 'sys' + 'tem')")
    assert "DYNAMIC_ATTRIBUTE_ACCESS" in result.flags


def test_getattr_with_constant_dangerous_name_flagged() -> None:
    # getattr(__builtins__, "eval")(...) reaches eval without the text "eval(".
    result = _analyze("ev = getattr(__builtins__, 'eval')")
    assert "DYNAMIC_ATTRIBUTE_ACCESS" in result.flags


def test_setattr_with_computed_name_flagged() -> None:
    result = _analyze("setattr(obj, attr_name, value)")
    assert "DYNAMIC_ATTRIBUTE_ACCESS" in result.flags


# ── evasion: obfuscated / parenthesized calls ────────────────────────────────


def test_parenthesized_eval_still_flagged() -> None:
    # "(eval)('x')" defeats the \beval\s*\( regex; AST sees the Name call.
    result = _analyze("(eval)('1+1')")
    assert "DYNAMIC_EXECUTION" in result.flags


def test_whitespace_split_eval_still_flagged() -> None:
    result = _analyze("eval (\n    '1+1'\n)")
    assert "DYNAMIC_EXECUTION" in result.flags


def test_dunder_import_identifies_module() -> None:
    result = _analyze("__import__('subprocess')")
    assert "DYNAMIC_IMPORT" in result.flags
    assert "DANGEROUS_IMPORT:subprocess" in result.flags


def test_aliased_dangerous_import_flagged() -> None:
    result = _analyze("import subprocess as sp\nsp.run(['ls'])")
    assert "DANGEROUS_IMPORT:subprocess" in result.flags


# ── benign code: no false-positive over-escalation ───────────────────────────


def test_getattr_with_benign_constant_not_flagged() -> None:
    result = _analyze("timeout = getattr(config, 'timeout', 30)")
    assert "DYNAMIC_ATTRIBUTE_ACCESS" not in result.flags


def test_plain_getattr_default_not_flagged() -> None:
    result = _analyze("val = getattr(obj, 'value')")
    assert "DYNAMIC_ATTRIBUTE_ACCESS" not in result.flags


def test_benign_stdlib_code_zero_risk() -> None:
    code = "import math\n\ndef area(r):\n    return math.pi * r * r\n"
    result = _analyze(code)
    assert result.flags == []
    assert result.risk_score == pytest.approx(0.0)


def test_benign_import_not_self_referential() -> None:
    result = _analyze("import json\nimport math")
    assert "SELF_REFERENTIAL_CODE" not in result.flags
    assert not any(f.startswith("DANGEROUS_IMPORT") for f in result.flags)
