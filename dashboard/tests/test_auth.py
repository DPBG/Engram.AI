"""Tests for the dashboard control-plane authentication (Phase 1.7).

Loads auth.py directly so the test doesn't import the dashboard package
(which pulls in FastAPI/uvicorn). The auth logic itself is pure stdlib.
"""

import importlib.util
import os
import sys

_AUTH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "dashboard", "auth.py"
)
_spec = importlib.util.spec_from_file_location("dash_auth", _AUTH_PATH)
auth = importlib.util.module_from_spec(_spec)
sys.modules["dash_auth"] = auth
_spec.loader.exec_module(auth)

TOKEN = "unit-test-dashboard-token"


# ── token configuration ───────────────────────────────────────────────────

def test_auth_disabled_without_token(monkeypatch):
    monkeypatch.delenv(auth.DASHBOARD_TOKEN_ENV, raising=False)
    assert auth.auth_enabled() is False
    # Disabled → everything authorized (legacy/dev mode).
    assert auth.authorize(None) is True
    assert auth.request_is_authorized("POST", None) is True


def test_auth_enabled_with_token(monkeypatch):
    monkeypatch.setenv(auth.DASHBOARD_TOKEN_ENV, TOKEN)
    assert auth.auth_enabled() is True


# ── bearer parsing ─────────────────────────────────────────────────────────

def test_parse_bearer():
    assert auth.parse_bearer(f"Bearer {TOKEN}") == TOKEN
    assert auth.parse_bearer(f"bearer {TOKEN}") == TOKEN  # case-insensitive scheme
    assert auth.parse_bearer("Basic abc") is None
    assert auth.parse_bearer("") is None
    assert auth.parse_bearer(None) is None


# ── authorization decisions ────────────────────────────────────────────────

def test_correct_token_authorized(monkeypatch):
    monkeypatch.setenv(auth.DASHBOARD_TOKEN_ENV, TOKEN)
    assert auth.authorize(TOKEN) is True


def test_wrong_token_rejected(monkeypatch):
    monkeypatch.setenv(auth.DASHBOARD_TOKEN_ENV, TOKEN)
    assert auth.authorize("not-the-token") is False


def test_missing_token_rejected_when_enabled(monkeypatch):
    monkeypatch.setenv(auth.DASHBOARD_TOKEN_ENV, TOKEN)
    assert auth.authorize(None) is False


# ── HTTP request gating (the security-relevant contract) ────────────────────

def test_get_never_requires_auth(monkeypatch):
    # Read-only GETs stay open even when auth is enabled (UI must load).
    monkeypatch.setenv(auth.DASHBOARD_TOKEN_ENV, TOKEN)
    assert auth.request_is_authorized("GET", None) is True


def test_mutating_request_requires_valid_token(monkeypatch):
    monkeypatch.setenv(auth.DASHBOARD_TOKEN_ENV, TOKEN)
    assert auth.request_is_authorized("POST", f"Bearer {TOKEN}") is True
    assert auth.request_is_authorized("POST", "Bearer wrong") is False
    assert auth.request_is_authorized("POST", None) is False
    # DELETE (e.g. clearing probe results) is equally gated.
    assert auth.request_is_authorized("DELETE", None) is False


def test_forged_approval_response_path_is_gated(monkeypatch):
    # Regression: the WS approval_response channel forges a human's answer to a
    # Kernel DEFER. With a token set, an unauthenticated client must be refused.
    monkeypatch.setenv(auth.DASHBOARD_TOKEN_ENV, TOKEN)

    class _FakeWS:
        def __init__(self, headers, query):
            self.headers = headers
            self.query_params = query

    forged = _FakeWS(headers={}, query={})
    assert auth.authorize_websocket(forged) is False

    legit = _FakeWS(headers={}, query={"token": TOKEN})
    assert auth.authorize_websocket(legit) is True

    legit_hdr = _FakeWS(headers={"authorization": f"Bearer {TOKEN}"}, query={})
    assert auth.authorize_websocket(legit_hdr) is True
