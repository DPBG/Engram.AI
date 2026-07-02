"""
Integration tests for SDK functionality.

Tests EventBus, core types, and basic SDK operations.
"""

import asyncio
import uuid

import pytest

# These imports would work when SDK is properly installed
# from activelearning.nats_client import EventBus
# from activelearning.core import Observation, ActionProposal, KernelDecision, KernelDecisionType


@pytest.mark.asyncio
async def test_sdk_import():
    """Test that SDK modules can be imported."""
    try:
        from activelearning import core, nats_client

        assert hasattr(core, "Observation")
        assert hasattr(core, "ActionProposal")
        assert hasattr(core, "KernelDecision")
        assert hasattr(nats_client, "EventBus")
    except ImportError as e:
        pytest.skip(f"SDK not installed: {e}")


@pytest.mark.asyncio
async def test_event_bus_lifecycle(nats_url: str):
    """Test EventBus connection lifecycle."""
    try:
        from activelearning.nats_client import EventBus
    except ImportError:
        pytest.skip("SDK not installed")

    bus = EventBus(nats_url=nats_url, name="test-bus")

    # Connect
    await bus.connect()
    assert bus.is_connected

    # Close
    await bus.close()
    assert not bus.is_connected


@pytest.mark.asyncio
async def test_event_bus_publish_subscribe(nats_url: str):
    """Test EventBus publish and subscribe."""
    try:
        from activelearning.nats_client import EventBus
    except ImportError:
        pytest.skip("SDK not installed")

    bus = EventBus(nats_url=nats_url, name="test-bus")
    await bus.connect()

    try:
        test_subject = f"test.sdk.{uuid.uuid4().hex[:8]}"
        received_data = None

        async def handler(data):
            nonlocal received_data
            received_data = data

        # Subscribe
        await bus.subscribe(test_subject, handler)

        await asyncio.sleep(0.1)

        # Publish
        test_data = {"message": "hello", "value": 42}
        await bus.publish(test_subject, test_data)

        await asyncio.sleep(0.2)

        assert received_data is not None
        assert received_data["message"] == "hello"
        assert received_data["value"] == 42

    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_observation_dataclass():
    """Test Observation dataclass."""
    try:
        from activelearning.core import Observation, generate_trace_id
    except ImportError:
        pytest.skip("SDK not installed")

    trace_id = generate_trace_id()
    obs = Observation(
        trace_id=trace_id,
        provenance="sensor.test",
        data={"test": "data"},
        confidence=0.95,
        tags=["test"],
    )

    assert obs.trace_id == trace_id
    assert obs.provenance == "sensor.test"
    assert obs.data["test"] == "data"
    assert obs.confidence == 0.95
    assert "test" in obs.tags
    assert obs.timestamp > 0


@pytest.mark.asyncio
async def test_action_proposal_dataclass():
    """Test ActionProposal dataclass."""
    try:
        from activelearning.core import ActionProposal, generate_trace_id
    except ImportError:
        pytest.skip("SDK not installed")

    trace_id = generate_trace_id()
    proposal = ActionProposal(
        trace_id=trace_id,
        provenance="planner.test",
        action={"type": "test", "value": 100},
        priority=5,
        requires_approval=False,
        metadata={"source": "test"},
    )

    assert proposal.trace_id == trace_id
    assert proposal.provenance == "planner.test"
    assert proposal.action["type"] == "test"
    assert proposal.priority == 5
    assert not proposal.requires_approval


@pytest.mark.asyncio
async def test_kernel_decision_dataclass():
    """Test KernelDecision dataclass."""
    try:
        from activelearning.core import (
            KernelDecision,
            KernelDecisionType,
            current_timestamp,
            generate_trace_id,
        )
    except ImportError:
        pytest.skip("SDK not installed")

    trace_id = generate_trace_id()
    issued_at = current_timestamp()
    expires_at = issued_at + 60000  # 60 seconds

    decision = KernelDecision(
        trace_id=trace_id,
        type=KernelDecisionType.ALLOW,
        reason="Test approval",
        risk_score=0.15,
        issued_at=issued_at,
        expires_at=expires_at,
    )

    assert decision.trace_id == trace_id
    assert decision.type == KernelDecisionType.ALLOW
    assert decision.reason == "Test approval"
    assert decision.risk_score == 0.15
    assert not decision.is_expired()
    assert decision.is_approved()


@pytest.mark.asyncio
async def test_belief_node_dataclass():
    """Test BeliefNode dataclass."""
    try:
        from activelearning.core import BeliefNode, BeliefNodeType, generate_trace_id
    except ImportError:
        pytest.skip("SDK not installed")

    node = BeliefNode(
        id=generate_trace_id(),
        type=BeliefNodeType.VALUE,
        content="Safety is paramount",
        confidence=0.95,
        source="human_teaching",
        metadata={"priority": "high"},
    )

    assert node.type == BeliefNodeType.VALUE
    assert node.content == "Safety is paramount"
    assert node.confidence == 0.95
    assert node.source == "human_teaching"
    assert node.metadata["priority"] == "high"


@pytest.mark.asyncio
async def test_belief_edge_dataclass():
    """Test BeliefEdge dataclass."""
    try:
        from activelearning.core import BeliefEdge, BeliefEdgeType, generate_trace_id
    except ImportError:
        pytest.skip("SDK not installed")

    source_id = generate_trace_id()
    target_id = generate_trace_id()

    edge = BeliefEdge(
        id=generate_trace_id(),
        type=BeliefEdgeType.SUPPORTS,
        source_id=source_id,
        target_id=target_id,
        strength=0.8,
        evidence="Observed in 50 cases",
    )

    assert edge.type == BeliefEdgeType.SUPPORTS
    assert edge.source_id == source_id
    assert edge.target_id == target_id
    assert edge.strength == 0.8
    assert edge.evidence == "Observed in 50 cases"


@pytest.mark.asyncio
async def test_event_bus_wait_for_decision(nats_url: str):
    """Test EventBus wait_for_decision helper."""
    try:
        from activelearning.core import generate_trace_id
        from activelearning.nats_client import EventBus
    except ImportError:
        pytest.skip("SDK not installed")

    bus = EventBus(nats_url=nats_url, name="test-bus")
    await bus.connect()

    try:
        trace_id = generate_trace_id()

        # Simulate decision being published after delay
        async def publish_decision():
            await asyncio.sleep(0.2)
            decision_data = {
                "trace_id": trace_id,
                "type": "ALLOW",
                "reason": "Test",
                "risk_score": 0.1,
            }
            await bus.publish(f"decision.{trace_id}", decision_data)

        # Start publish task
        publish_task = asyncio.create_task(publish_decision())

        # Wait for decision
        decision = await bus.wait_for_decision(trace_id, timeout=2.0)

        assert decision["trace_id"] == trace_id
        assert decision["type"] == "ALLOW"
        assert decision["reason"] == "Test"

        await publish_task

    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_observation_validation():
    """Test Observation validation."""
    try:
        from activelearning.core import Observation, generate_trace_id
    except ImportError:
        pytest.skip("SDK not installed")

    trace_id = generate_trace_id()

    # Valid confidence
    obs = Observation(trace_id=trace_id, provenance="test", data={}, confidence=0.5)
    assert obs.confidence == 0.5

    # Invalid confidence (too high)
    with pytest.raises(ValueError):
        Observation(trace_id=trace_id, provenance="test", data={}, confidence=1.5)  # > 1.0

    # Invalid confidence (too low)
    with pytest.raises(ValueError):
        Observation(trace_id=trace_id, provenance="test", data={}, confidence=-0.1)  # < 0.0
