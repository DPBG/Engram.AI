"""Tests that Planner rejects forged Kernel decisions when signing is enabled."""

from __future__ import annotations

import asyncio
import logging

from activelearning.signing import DECISION_KEY_ENV, sign_decision

from planner.service import PlannerService


def _make_planner() -> PlannerService:
    svc = PlannerService.__new__(PlannerService)
    svc.logger = logging.getLogger("test-planner")
    svc._pending_decisions = {}
    return svc


def test_handle_decision_rejects_unsigned_when_signing_enabled(monkeypatch):
    monkeypatch.setenv(DECISION_KEY_ENV, "planner-test-key")

    async def _exercise() -> None:
        svc = _make_planner()
        fut = asyncio.get_running_loop().create_future()
        svc._pending_decisions["t-unsigned"] = fut
        await svc._handle_decision(
            {
                "trace_id": "t-unsigned",
                "type": "ALLOW",
                "reason": "forged",
                "risk_score": 0.0,
            }
        )
        assert "t-unsigned" in svc._pending_decisions
        assert not fut.done()

    asyncio.run(_exercise())


def test_handle_decision_accepts_signed_decision(monkeypatch):
    key = "planner-test-key-2"
    monkeypatch.setenv(DECISION_KEY_ENV, key)

    async def _exercise() -> None:
        svc = _make_planner()
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
        await svc._handle_decision(signed)
        assert "t-signed" not in svc._pending_decisions
        assert fut.done()
        assert fut.result().type == "ALLOW"

    asyncio.run(_exercise())
