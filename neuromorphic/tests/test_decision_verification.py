"""Tests for kernel decision verification in the neuromorphic safety gate.

Mirrors ``NeuromorphicService._handle_kernel_decision`` without importing
the full service module (which depends on the activelearning SDK that is not
installed in the neuromorphic-only CI venv).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

import pytest

# Signing lives in the SDK; add it to the path the same way other cross-package
# neuromorphic tests do (see test_safety_pipeline.py).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SDK_SRC = os.path.join(_PROJECT_ROOT, "sdk", "src")
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

from activelearning.signing import (  # noqa: E402
    DECISION_KEY_ENV,
    sign_decision,
    verify_decision,
)


class _DecisionGateStub:
    """Minimal stub with only the state ``_handle_kernel_decision`` touches."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("test-neuromorphic")
        self._pending_decisions: dict[str, asyncio.Future[dict[str, Any]]] = {}


async def _handle_kernel_decision(svc: _DecisionGateStub, data: dict[str, Any]) -> None:
    """Replicate the verification gate from neuromorphic.service (keep in sync)."""
    trace_id = data.get("trace_id", "")
    fut = svc._pending_decisions.get(trace_id)
    if fut is None:
        return
    if fut.done():
        return
    if not verify_decision(data):
        svc.logger.error(
            "Rejected unverified decision for %s — ignoring (possible forgery)",
            trace_id,
        )
        return
    fut.set_result(data)


@pytest.mark.asyncio
async def test_handle_kernel_decision_rejects_unsigned_when_signing_enabled(monkeypatch):
    monkeypatch.setenv(DECISION_KEY_ENV, "neuro-test-key")
    svc = _DecisionGateStub()
    fut = asyncio.get_running_loop().create_future()
    svc._pending_decisions["t-unsigned"] = fut

    await _handle_kernel_decision(
        svc,
        {
            "trace_id": "t-unsigned",
            "type": "ALLOW",
            "reason": "forged",
            "risk_score": 0.0,
        },
    )

    assert not fut.done()


@pytest.mark.asyncio
async def test_handle_kernel_decision_accepts_signed_decision(monkeypatch):
    key = "neuro-test-key-2"
    monkeypatch.setenv(DECISION_KEY_ENV, key)
    svc = _DecisionGateStub()
    fut = asyncio.get_running_loop().create_future()
    svc._pending_decisions["t-signed"] = fut

    signed = sign_decision(
        {
            "trace_id": "t-signed",
            "type": "ALLOW",
            "reason": "ok",
            "risk_score": 0.0,
        },
        key,
    )
    await _handle_kernel_decision(svc, signed)

    assert fut.done()
    assert fut.result()["type"] == "ALLOW"
