"""Tests for dashboard SAFE_HALT payload helpers (Phase E1.9.6).

Loads safe_halt.py directly — no FastAPI/NATS imports needed.
"""

import importlib.util
import os
import sys

_HALT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "dashboard", "safe_halt.py"
)
_spec = importlib.util.spec_from_file_location("dash_safe_halt", _HALT_PATH)
sh = importlib.util.module_from_spec(_spec)
sys.modules["dash_safe_halt"] = sh
_spec.loader.exec_module(sh)


# ── get_halt_state / update_halt_state ───────────────────────────────────────

def test_initial_state_not_halted():
    sh._halt_state = {"halted": False}
    assert sh.get_halt_state() == {"halted": False}


def test_update_halt_state_persists():
    sh.update_halt_state({"halted": True, "reason": "test"})
    assert sh.get_halt_state()["halted"] is True
    assert sh.get_halt_state()["reason"] == "test"


def test_get_halt_state_returns_copy():
    sh._halt_state = {"halted": False}
    copy = sh.get_halt_state()
    copy["halted"] = True
    assert sh._halt_state["halted"] is False


def test_update_halt_state_overwrites():
    sh.update_halt_state({"halted": True, "reason": "first"})
    sh.update_halt_state({"halted": False})
    s = sh.get_halt_state()
    assert s["halted"] is False
    assert "reason" not in s


# ── sanitize_halt_payload ─────────────────────────────────────────────────────

def test_halt_valid_payload():
    ok, reason, op = sh.sanitize_halt_payload({"reason": "test", "operator_id": "ops"})
    assert ok is True
    assert reason == "test"
    assert op == "ops"


def test_halt_non_dict_fails_closed():
    ok, reason, op = sh.sanitize_halt_payload("bad")
    assert ok is False
    assert reason == ""
    assert op == ""


def test_halt_none_fails_closed():
    ok, _, _ = sh.sanitize_halt_payload(None)
    assert ok is False


def test_halt_empty_dict_uses_defaults():
    ok, reason, op = sh.sanitize_halt_payload({})
    assert ok is True
    assert reason == "operator SAFE_HALT"
    assert op == "dashboard"


def test_halt_blank_reason_uses_default():
    ok, reason, _ = sh.sanitize_halt_payload({"reason": "   "})
    assert ok is True
    assert reason == "operator SAFE_HALT"


def test_halt_blank_operator_uses_default():
    ok, _, op = sh.sanitize_halt_payload({"operator_id": "  "})
    assert ok is True
    assert op == "dashboard"


def test_halt_reason_truncated_to_max():
    long_reason = "x" * (sh.MAX_REASON_LEN + 50)
    ok, reason, _ = sh.sanitize_halt_payload({"reason": long_reason})
    assert ok is True
    assert len(reason) <= sh.MAX_REASON_LEN


def test_halt_operator_truncated_to_max():
    long_op = "y" * (sh.MAX_OPERATOR_LEN + 20)
    ok, _, op = sh.sanitize_halt_payload({"operator_id": long_op})
    assert ok is True
    assert len(op) <= sh.MAX_OPERATOR_LEN


def test_halt_non_string_fields_coerced():
    ok, reason, op = sh.sanitize_halt_payload({"reason": 42, "operator_id": True})
    assert ok is True
    assert reason == "42"
    assert op == "True"


# ── sanitize_resume_payload ───────────────────────────────────────────────────

def test_resume_valid_payload():
    ok, op = sh.sanitize_resume_payload({"operator_id": "admin"})
    assert ok is True
    assert op == "admin"


def test_resume_non_dict_fails_closed():
    ok, op = sh.sanitize_resume_payload("bad")
    assert ok is False
    assert op == ""


def test_resume_none_fails_closed():
    ok, op = sh.sanitize_resume_payload(None)
    assert ok is False
    assert op == ""


def test_resume_empty_dict_uses_default():
    ok, op = sh.sanitize_resume_payload({})
    assert ok is True
    assert op == "dashboard"


def test_resume_blank_operator_uses_default():
    ok, op = sh.sanitize_resume_payload({"operator_id": "  "})
    assert ok is True
    assert op == "dashboard"


def test_resume_operator_truncated_to_max():
    long_op = "z" * (sh.MAX_OPERATOR_LEN + 10)
    ok, op = sh.sanitize_resume_payload({"operator_id": long_op})
    assert ok is True
    assert len(op) <= sh.MAX_OPERATOR_LEN
