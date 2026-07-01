"""Tests for the M1 prerequisite gate (issue #140).

Loads capability_flags.py directly (pure stdlib) so the test runs without
importing the meta_programmer package __init__, which pulls in the docker SDK.
"""

import importlib.util
import os
import sys

import pytest

_FLAGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "meta_programmer", "capability_flags.py"
)
_spec = importlib.util.spec_from_file_location("mp_flags", _FLAGS_PATH)
_flags = importlib.util.module_from_spec(_spec)
sys.modules["mp_flags"] = _flags
_spec.loader.exec_module(_flags)

m1_complete = _flags.m1_complete
signing_enabled = _flags.signing_enabled
sandbox_failclosed_enabled = _flags.sandbox_failclosed_enabled
check_m1_or_deny = _flags.check_m1_or_deny
SIGNING_ENV = _flags.SIGNING_ENV
SANDBOX_FAILCLOSED_ENV = _flags.SANDBOX_FAILCLOSED_ENV


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove both M1 env vars before every test (fail-closed baseline)."""
    monkeypatch.delenv(SIGNING_ENV, raising=False)
    monkeypatch.delenv(SANDBOX_FAILCLOSED_ENV, raising=False)


# ── m1_complete / individual flag tests ───────────────────────────────────────


def test_both_unset_is_incomplete():
    assert m1_complete() is False


def test_signing_only_is_incomplete(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "1")
    assert m1_complete() is False


def test_sandbox_only_is_incomplete(monkeypatch):
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "1")
    assert m1_complete() is False


def test_both_set_to_1_is_complete(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "1")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "1")
    assert m1_complete() is True


def test_true_string_accepted(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "true")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "true")
    assert m1_complete() is True


def test_yes_string_accepted(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "yes")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "yes")
    assert m1_complete() is True


def test_false_string_rejected(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "false")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "false")
    assert m1_complete() is False


def test_empty_string_rejected(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "")
    assert m1_complete() is False


def test_zero_rejected(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "0")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "0")
    assert m1_complete() is False


def test_case_insensitive(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "TRUE")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "True")
    assert m1_complete() is True


def test_whitespace_trimmed(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "  1  ")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "  true  ")
    assert m1_complete() is True


def test_individual_flags_report_correctly(monkeypatch):
    assert signing_enabled() is False
    assert sandbox_failclosed_enabled() is False

    monkeypatch.setenv(SIGNING_ENV, "1")
    assert signing_enabled() is True
    assert sandbox_failclosed_enabled() is False

    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "1")
    assert signing_enabled() is True
    assert sandbox_failclosed_enabled() is True


# ── check_m1_or_deny ──────────────────────────────────────────────────────────


def test_deny_when_both_unset():
    ok, reason = check_m1_or_deny()
    assert ok is False
    assert "M1 safety prerequisites incomplete" in reason
    assert SIGNING_ENV in reason
    assert SANDBOX_FAILCLOSED_ENV in reason


def test_deny_mentions_only_signing_when_sandbox_present(monkeypatch):
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "1")
    ok, reason = check_m1_or_deny()
    assert ok is False
    assert SIGNING_ENV in reason
    assert SANDBOX_FAILCLOSED_ENV not in reason


def test_deny_mentions_only_sandbox_when_signing_present(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "1")
    ok, reason = check_m1_or_deny()
    assert ok is False
    assert SANDBOX_FAILCLOSED_ENV in reason
    assert SIGNING_ENV not in reason


def test_allow_when_both_set(monkeypatch):
    monkeypatch.setenv(SIGNING_ENV, "1")
    monkeypatch.setenv(SANDBOX_FAILCLOSED_ENV, "1")
    ok, reason = check_m1_or_deny()
    assert ok is True
    assert reason == ""


def test_deny_reason_is_nonempty():
    ok, reason = check_m1_or_deny()
    assert ok is False
    assert len(reason) > 0


def test_deny_contains_fail_closed_language():
    _, reason = check_m1_or_deny()
    assert "fail-closed" in reason
