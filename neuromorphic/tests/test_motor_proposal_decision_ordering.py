"""Regression: motor proposals must register the decision Future before publish.

Imports ``neuromorphic.service`` (requires the ``activelearning`` SDK).
CI runs this file with ``--with-editable ../sdk`` alongside the other
service-importing tests — not in the general ``tests/`` sweep.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

_SMALL_NEURO_ENV = {
    "NEURO_BRAINSTEM_N": "20",
    "NEURO_REFLEX_N": "15",
    "NEURO_SENSORY_N": "60",
    "NEURO_MOTOR_N": "40",
    "NEURO_CEREBELLUM_N": "30",
    "NEURO_ASSOCIATION_N": "60",
    "NEURO_PREDICTIVE_N": "30",
    "NEURO_WORKING_MEM_N": "20",
    "NEURO_FEATURE_N": "0",
    "NEURO_CONCEPT_N": "0",
    "NEURO_DG_N": "0",
    "NEURO_META_N": "0",
    "NEURO_SAFETY_GATE": "1",
    "NEURO_SAFETY_FAIL_OPEN": "0",
    "NEURO_SAFETY_TIMEOUT": "2.0",
}


def _set_small_env(monkeypatch) -> None:
    for key, value in _SMALL_NEURO_ENV.items():
        monkeypatch.setenv(key, value)


class _RaceBus:
    """EventBus stub that delivers a Kernel ALLOW *during* publish.

    Reproduces the local-broker / fast-Kernel race: the decision lands on
    ``decision.>`` before ``_emit_motor_proposal`` returns from publish.
    """

    def __init__(self, service: Any) -> None:
        self._service = service
        self.published: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, subject: str, data: dict[str, Any]) -> None:
        self.published.append((subject, data))
        if subject != "proposal.new":
            return
        # Synchronous delivery mid-publish — the Future must already be
        # registered or this ALLOW is dropped as "stale".
        await self._service._handle_kernel_decision(
            {
                "trace_id": data["trace_id"],
                "type": "ALLOW",
                "risk_score": 0.0,
                "reason": "race-test",
            }
        )


class TestMotorProposalDecisionOrdering:
    @pytest.mark.asyncio
    async def test_registers_future_before_publish_so_fast_allow_is_kept(self, monkeypatch):
        """A Kernel ALLOW that arrives during publish must resolve the Future."""
        _set_small_env(monkeypatch)
        from neuromorphic.service import NeuromorphicService

        svc = NeuromorphicService()
        bus = _RaceBus(svc)
        svc.event_bus = bus

        actuated: list[dict[str, Any]] = []

        class _Adapter:
            async def handle_motor_command(self, **kwargs):
                actuated.append(kwargs)

        svc._motor_adapter = _Adapter()

        await svc._emit_motor_proposal(
            {"channel": "locomotion", "intensity": 0.4},
            step_count=1,
        )

        # Let the background _await_safety_decision task finish.
        if svc._decision_tasks:
            await asyncio.wait(svc._decision_tasks, timeout=2.0)

        assert len(bus.published) == 1
        assert bus.published[0][0] == "proposal.new"
        assert actuated, (
            "Fast ALLOW during publish was dropped — Future was registered "
            "too late (publish-before-register race)"
        )
        assert actuated[0]["channel"] == "locomotion"
        assert actuated[0]["intensity"] == pytest.approx(0.4)
        assert svc._pending_decisions == {}

    @pytest.mark.asyncio
    async def test_publish_failure_unregisters_pending_future(self, monkeypatch):
        """A failed publish must not leave an orphaned pending Future."""
        _set_small_env(monkeypatch)
        from neuromorphic.service import NeuromorphicService

        svc = NeuromorphicService()

        class _FailBus:
            async def publish(self, subject: str, data: dict[str, Any]) -> None:
                raise RuntimeError("nats down")

        svc.event_bus = _FailBus()

        await svc._emit_motor_proposal(
            {"channel": "manipulation", "intensity": 0.2},
            step_count=2,
        )

        assert svc._pending_decisions == {}
        assert svc._pending_proposals == {}
        assert svc._decision_tasks == set()
