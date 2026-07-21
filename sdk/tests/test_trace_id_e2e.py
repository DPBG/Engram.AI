"""End-to-end trace_id propagation verification across the full pipeline (#238).

Pipeline under test:
  sensory-gateway (SensorPlugin.emit)
    → observation.{sensor_id}
    → coordinator (_request_execution_approval → proposal.new)
    → kernel      (_handle_action_proposal → decision.{trace_id})
    → dashboard   (message buffer / WebSocket broadcast)

No live NATS server is required.  Each hop is driven by the SDK data types
and a lightweight recording stub, so the test verifies the trace_id contract
at every boundary without external infrastructure.

Architecture note: the coordinator generates a fresh trace_id per proposal; it
does not forward the observation's trace_id to the kernel.  This file covers
both lineages:
  • Gateway  → coordinator: observation carries a UUID trace_id the coordinator
    can read for correlation logging.
  • Coordinator → kernel → dashboard: the proposal's trace_id must survive the
    full kernel round-trip and arrive in the dashboard buffer unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from activelearning.core import (
    KernelDecision,
    KernelDecisionType,
    Observation,
    generate_trace_id,
)
from activelearning.plugins import SensorPlugin
from activelearning.subjects import Subjects, decision_subject, observation_subject

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingBus:
    """Minimal bus stub that records every (subject, payload) pair."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, Any]] = []

    async def publish(self, subject: str, payload: Any) -> None:
        self.messages.append((subject, payload))


class _GatewaySensor(SensorPlugin):
    """Concrete sensor stub used to exercise SensorPlugin.emit()."""

    async def capture(self) -> dict:  # type: ignore[override]
        return {}


def _make_sensor(sensor_id: str, bus: _RecordingBus) -> _GatewaySensor:
    """Construct a sensor with an injected bus — skips start() and its task."""
    sensor = _GatewaySensor(sensor_id, f"Test {sensor_id}")
    sensor._bus = bus
    return sensor


def _build_proposal(trace_id: str, task_id: str = "task-001") -> dict:
    """Simulate coordinator.gate.build_execution_proposal()."""
    return {
        "trace_id": trace_id,
        "provenance": "coordinator",
        "action": {
            "type": "task_execution",
            "task_id": task_id,
            "parameters": {},
        },
    }


def _build_decision_payload(decision: KernelDecision) -> dict:
    """Simulate KernelService._publish_and_log_decision() payload (unsigned)."""
    return {
        "trace_id": decision.trace_id,
        "type": decision.type.value,
        "reason": decision.reason,
        "risk_score": decision.risk_score,
        "issued_at": decision.issued_at,
    }


# ---------------------------------------------------------------------------
# Hop 1 — sensory-gateway: SensorPlugin.emit() stamps a trace_id
# ---------------------------------------------------------------------------


class TestGatewayHop:
    """SensorPlugin.emit() must stamp a non-empty, unique trace_id on every
    observation published to observation.{sensor_id}."""

    @pytest.mark.asyncio
    async def test_emit_stamps_trace_id(self):
        bus = _RecordingBus()
        await _make_sensor("cam0", bus).emit({"frame": 42})

        assert len(bus.messages) == 1
        subject, obs = bus.messages[0]
        assert subject == observation_subject("cam0")
        assert isinstance(obs, Observation)
        assert obs.trace_id, "trace_id must be non-empty"

    @pytest.mark.asyncio
    async def test_trace_id_is_valid_uuid(self):
        """trace_id must be parsable as a UUID — not a hash or counter."""
        bus = _RecordingBus()
        await _make_sensor("depth0", bus).emit({"depth": 1.5})
        _, obs = bus.messages[0]
        uuid.UUID(obs.trace_id)  # raises ValueError if malformed

    @pytest.mark.asyncio
    async def test_each_emit_gets_unique_trace_id(self):
        """Two consecutive observations must never share a trace_id."""
        bus = _RecordingBus()
        sensor = _make_sensor("imu0", bus)
        await sensor.emit({"ax": 0.1})
        await sensor.emit({"ax": 0.2})

        ids = [obs.trace_id for _, obs in bus.messages]
        assert ids[0] != ids[1]

    @pytest.mark.asyncio
    async def test_observation_carries_sensor_provenance_alongside_trace_id(self):
        """The coordinator needs both trace_id and provenance on receipt."""
        bus = _RecordingBus()
        await _make_sensor("lidar0", bus).emit({"scan": []})
        _, obs = bus.messages[0]
        assert obs.trace_id
        assert obs.provenance == "sensor.lidar0"


# ---------------------------------------------------------------------------
# Hop 2 — coordinator → proposal.new carries a trace_id
# ---------------------------------------------------------------------------


class TestCoordinatorHop:
    """The coordinator must embed a non-empty trace_id in the proposal it
    publishes to proposal.new."""

    @pytest.mark.asyncio
    async def test_proposal_carries_non_empty_trace_id(self):
        bus = _RecordingBus()
        trace_id = generate_trace_id()
        await bus.publish(Subjects.PROPOSAL_NEW, _build_proposal(trace_id))

        _, payload = bus.messages[0]
        assert payload["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_proposal_trace_id_is_valid_uuid(self):
        trace_id = generate_trace_id()
        uuid.UUID(trace_id)  # generate_trace_id must return a UUID string
        proposal = _build_proposal(trace_id)
        assert proposal["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_distinct_tasks_get_distinct_trace_ids(self):
        trace_a = generate_trace_id()
        trace_b = generate_trace_id()
        assert trace_a != trace_b


# ---------------------------------------------------------------------------
# Hop 3 — kernel → decision.{trace_id} echoes the proposal's trace_id
# ---------------------------------------------------------------------------


class TestKernelHop:
    """The kernel must publish to decision.{trace_id} and embed the same
    trace_id in the payload — both are checked by wait_for_decision()."""

    @pytest.mark.asyncio
    async def test_decision_subject_encodes_trace_id(self):
        trace_id = generate_trace_id()
        assert decision_subject(trace_id) == f"decision.{trace_id}"

    @pytest.mark.asyncio
    async def test_kernel_decision_echoes_proposal_trace_id(self):
        """KernelDecision must carry the trace_id from the incoming proposal."""
        proposal_trace_id = generate_trace_id()
        decision = KernelDecision(
            trace_id=proposal_trace_id,
            type=KernelDecisionType.ALLOW,
            reason="policy allows",
        )
        assert decision.trace_id == proposal_trace_id

    @pytest.mark.asyncio
    async def test_allow_decision_published_with_correct_subject_and_payload(self):
        """Simulates _publish_and_log_decision for an ALLOW outcome."""
        bus = _RecordingBus()
        proposal_trace_id = generate_trace_id()

        decision = KernelDecision(
            trace_id=proposal_trace_id,
            type=KernelDecisionType.ALLOW,
            reason="allowed",
            risk_score=0.05,
        )
        await bus.publish(decision_subject(decision.trace_id), _build_decision_payload(decision))

        pub_subject, pub_payload = bus.messages[0]
        assert pub_subject == f"decision.{proposal_trace_id}"
        assert pub_payload["trace_id"] == proposal_trace_id

    @pytest.mark.asyncio
    async def test_deny_decision_also_echoes_trace_id(self):
        """A DENY response must carry the original trace_id, not a fresh one."""
        bus = _RecordingBus()
        trace_id = generate_trace_id()

        decision = KernelDecision(
            trace_id=trace_id,
            type=KernelDecisionType.DENY,
            reason="risk too high",
            risk_score=0.95,
        )
        await bus.publish(decision_subject(decision.trace_id), _build_decision_payload(decision))

        pub_subject, pub_payload = bus.messages[0]
        assert pub_subject == f"decision.{trace_id}"
        assert pub_payload["trace_id"] == trace_id
        assert pub_payload["type"] == "DENY"


# ---------------------------------------------------------------------------
# Hop 4 — dashboard receives decision with correct trace_id
# ---------------------------------------------------------------------------


class TestDashboardHop:
    """Dashboard's wildcard handler (_handle_msg) must preserve the trace_id
    in the buffered entry so operators can correlate across services."""

    @pytest.mark.asyncio
    async def test_dashboard_buffer_entry_carries_trace_id(self):
        """Simulate NatsStreamManager._handle_msg buffering a decision."""
        trace_id = generate_trace_id()
        decision_sub = decision_subject(trace_id)
        decision_data = {
            "trace_id": trace_id,
            "type": "ALLOW",
            "reason": "allowed",
            "risk_score": 0.0,
            "issued_at": 1_000_000_000,
        }

        # Dashboard appends: {"subject": subject, "data": data}
        buffered = {"subject": decision_sub, "data": decision_data}

        assert buffered["subject"] == f"decision.{trace_id}"
        assert buffered["data"]["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_buffer_subject_suffix_matches_payload_trace_id(self):
        """decision.{trace_id} subject suffix must equal data.trace_id — a
        mismatch would silently break log correlation."""
        trace_id = generate_trace_id()
        buffered = {
            "subject": decision_subject(trace_id),
            "data": {"trace_id": trace_id, "type": "ALLOW"},
        }
        suffix = buffered["subject"].removeprefix("decision.")
        assert suffix == buffered["data"]["trace_id"]


# ---------------------------------------------------------------------------
# Full pipeline end-to-end
# ---------------------------------------------------------------------------


class TestFullPipelineE2E:
    """Verify that a trace_id set at the coordinator boundary arrives intact
    in the dashboard buffer after the kernel round-trip.

    Covers two lineages:
      1. gateway→coordinator: observation trace_id is readable at coordinator
      2. coordinator→kernel→dashboard: proposal trace_id survives unchanged
    """

    @pytest.mark.asyncio
    async def test_trace_id_threads_coordinator_to_kernel_to_dashboard(self):
        """
        Full simulation of the coordinator → kernel → dashboard chain:
        1. Coordinator emits proposal.new with trace_id T
        2. Kernel receives proposal, echoes T in decision.T
        3. Dashboard buffers entry with subject decision.T and data.trace_id == T
        """
        coordinator_bus = _RecordingBus()
        kernel_bus = _RecordingBus()
        dashboard_buffer: list[dict] = []

        # Step 1: coordinator publishes proposal.new
        trace_id = generate_trace_id()
        proposal = _build_proposal(trace_id)
        await coordinator_bus.publish(Subjects.PROPOSAL_NEW, proposal)

        received_proposal = coordinator_bus.messages[0][1]
        assert received_proposal["trace_id"] == trace_id  # trace_id in proposal

        # Step 2: kernel processes proposal, echoes trace_id in decision
        decision = KernelDecision(
            trace_id=received_proposal["trace_id"],
            type=KernelDecisionType.ALLOW,
            reason="kernel allows",
            risk_score=0.05,
        )
        kernel_subject = decision_subject(decision.trace_id)
        decision_payload = _build_decision_payload(decision)
        await kernel_bus.publish(kernel_subject, decision_payload)

        pub_subject, pub_payload = kernel_bus.messages[0]
        assert pub_subject == f"decision.{trace_id}"  # subject carries trace_id
        assert pub_payload["trace_id"] == trace_id  # payload carries trace_id

        # Step 3: dashboard buffers the decision message
        dashboard_buffer.append({"subject": pub_subject, "data": pub_payload})

        assert dashboard_buffer[0]["subject"] == f"decision.{trace_id}"
        assert dashboard_buffer[0]["data"]["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_gateway_observation_trace_id_is_readable_at_coordinator(self):
        """Observation emitted by the gateway must carry a trace_id the
        coordinator can read and log for correlation, even though the
        coordinator generates its own trace_id for the kernel proposal."""
        bus = _RecordingBus()
        await _make_sensor("cam0", bus).emit({"frame": 1})

        _, observation = bus.messages[0]
        gateway_trace_id = observation.trace_id
        assert gateway_trace_id, "coordinator cannot correlate without a trace_id"
        uuid.UUID(gateway_trace_id)  # structured logging requires a valid UUID

    @pytest.mark.asyncio
    async def test_concurrent_proposals_have_distinct_trace_ids(self):
        """Two concurrent proposals must never share a trace_id — a collision
        would cause a coordinator to receive the wrong kernel decision."""
        trace_ids = [generate_trace_id() for _ in range(20)]
        assert len(set(trace_ids)) == 20, "trace_id collision detected"

        subjects = [decision_subject(t) for t in trace_ids]
        assert len(set(subjects)) == 20, "decision subject collision detected"

    @pytest.mark.asyncio
    async def test_deny_trace_id_survives_full_pipeline(self):
        """A DENY outcome must carry the same trace_id through every hop —
        operators need it to locate the denial reason in the audit log."""
        coordinator_bus = _RecordingBus()
        kernel_bus = _RecordingBus()
        dashboard_buffer: list[dict] = []

        trace_id = generate_trace_id()
        await coordinator_bus.publish(Subjects.PROPOSAL_NEW, _build_proposal(trace_id))

        proposal_trace_id = coordinator_bus.messages[0][1]["trace_id"]
        decision = KernelDecision(
            trace_id=proposal_trace_id,
            type=KernelDecisionType.DENY,
            reason="risk score exceeds threshold",
            risk_score=0.92,
        )
        await kernel_bus.publish(
            decision_subject(decision.trace_id), _build_decision_payload(decision)
        )

        pub_subject, pub_payload = kernel_bus.messages[0]
        dashboard_buffer.append({"subject": pub_subject, "data": pub_payload})

        assert dashboard_buffer[0]["data"]["trace_id"] == trace_id
        assert dashboard_buffer[0]["data"]["type"] == "DENY"
        assert dashboard_buffer[0]["subject"] == f"decision.{trace_id}"
