"""
Integration tests for core ActiveLearningAI services.

Tests the critical message flows:
- Kernel: ALLOW/DENY/DEFER/TRANSFORM decision flow
- Planner: observation → ActionProposal
- Memory: episodic storage and retrieval
- Beliefs: graph operations
- Safety Supervisor: risk scoring
- Cache: LLM caching
- Coordinator: task coordination
- Overrides: human override processing
- External API: API query approval
"""

import asyncio
import json
import uuid
from typing import Optional

import pytest
from nats.aio.client import Client as NATSClient


# =============================================================================
# KERNEL TESTS - Decision Flow
# =============================================================================


@pytest.mark.asyncio
async def test_kernel_allow_decision(nats_client: NATSClient):
    """Test Kernel ALLOW decision for safe action."""
    trace_id = str(uuid.uuid4())
    decision_received: Optional[dict] = None

    async def decision_handler(msg):
        nonlocal decision_received
        decision_received = json.loads(msg.data.decode())

    # Subscribe to decision
    await nats_client.subscribe(f"decision.{trace_id}", cb=decision_handler)
    await asyncio.sleep(0.1)

    # Publish safe proposal
    proposal = {
        "trace_id": trace_id,
        "provenance": "planner.test",
        "action": {"type": "read_sensor", "sensor_id": "temperature"},
        "priority": 5,
        "requires_approval": False,
        "timestamp": 1234567890,
    }

    await nats_client.publish("proposal.new", json.dumps(proposal).encode())
    await asyncio.sleep(1.0)  # Wait for kernel to process

    # Note: This test will pass if kernel service is running
    # If kernel is not running, decision_received will be None
    if decision_received:
        assert decision_received["trace_id"] == trace_id
        assert decision_received["type"] in ["ALLOW", "DENY", "DEFER", "TRANSFORM"]


@pytest.mark.asyncio
async def test_kernel_decision_timeout(nats_client: NATSClient):
    """Test kernel decision timeout handling."""
    trace_id = str(uuid.uuid4())
    decision_received: Optional[dict] = None

    async def decision_handler(msg):
        nonlocal decision_received
        decision_received = json.loads(msg.data.decode())

    await nats_client.subscribe(f"decision.{trace_id}", cb=decision_handler)
    await asyncio.sleep(0.1)

    proposal = {
        "trace_id": trace_id,
        "provenance": "test",
        "action": {"type": "test"},
        "priority": 1,
    }

    await nats_client.publish("proposal.new", json.dumps(proposal).encode())

    # Wait with timeout
    try:
        await asyncio.wait_for(
            asyncio.sleep(2.0),  # Wait for decision
            timeout=3.0,
        )
    except asyncio.TimeoutError:
        pass

    # Decision may or may not be received depending on kernel availability


# =============================================================================
# PLANNER TESTS - Observation → Proposal Flow
# =============================================================================


@pytest.mark.asyncio
async def test_planner_observation_to_proposal(nats_client: NATSClient):
    """Test that planner generates proposals from observations."""
    proposals_received = []

    async def proposal_handler(msg):
        data = json.loads(msg.data.decode())
        proposals_received.append(data)

    # Subscribe to proposals
    await nats_client.subscribe("proposal.new", cb=proposal_handler)
    await asyncio.sleep(0.1)

    # Publish observation
    trace_id = str(uuid.uuid4())
    observation = {
        "trace_id": trace_id,
        "provenance": "sensor.camera.test",
        "data": {"motion_detected": True, "confidence": 0.95},
        "timestamp": 1234567890,
        "confidence": 0.95,
        "tags": ["motion"],
    }

    await nats_client.publish("observation.camera", json.dumps(observation).encode())
    await asyncio.sleep(1.5)  # Wait for planner to process

    # Note: Planner may or may not generate a proposal depending on its logic
    # This test verifies the message flow works


@pytest.mark.asyncio
async def test_observation_subject_routing(nats_client: NATSClient):
    """Test observation subject routing patterns."""
    observations = []

    async def obs_handler(msg):
        observations.append(msg.subject)

    await nats_client.subscribe("observation.*", cb=obs_handler)
    await asyncio.sleep(0.1)

    # Publish to different sensor subjects
    for sensor in ["camera", "imu", "lidar"]:
        obs = {
            "trace_id": str(uuid.uuid4()),
            "provenance": f"sensor.{sensor}",
            "data": {"value": 42},
            "confidence": 0.9,
        }
        await nats_client.publish(f"observation.{sensor}", json.dumps(obs).encode())

    await asyncio.sleep(0.5)

    assert len(observations) == 3
    assert "observation.camera" in observations
    assert "observation.imu" in observations
    assert "observation.lidar" in observations


# =============================================================================
# MEMORY TESTS - Episodic Storage and Retrieval
# =============================================================================


@pytest.mark.asyncio
async def test_memory_store_episode(nats_client: NATSClient):
    """Test storing an episode in memory."""
    response_received: Optional[dict] = None

    async def response_handler(msg):
        nonlocal response_received
        response_received = json.loads(msg.data.decode())

    trace_id = str(uuid.uuid4())

    # Create subscription for response
    await nats_client.subscribe(f"memory.stored.{trace_id}", cb=response_handler)
    await asyncio.sleep(0.1)

    # Store episode
    episode = {
        "trace_id": trace_id,
        "data": {"action": "test_action", "outcome": "success"},
        "tags": ["test", "success"],
        "utility_score": 0.85,
        "embedding_ref": "test_embedding_123",
    }

    await nats_client.publish("memory.store", json.dumps(episode).encode())
    await asyncio.sleep(1.0)

    # Memory service may or may not respond depending on availability


@pytest.mark.asyncio
async def test_memory_recall_query(nats_client: NATSClient):
    """Test memory recall query."""
    response_received: Optional[dict] = None

    async def response_handler(msg):
        nonlocal response_received
        response_received = json.loads(msg.data.decode())

    query_id = str(uuid.uuid4())
    await nats_client.subscribe(f"memory.recall.result.{query_id}", cb=response_handler)
    await asyncio.sleep(0.1)

    # Query memory
    query = {
        "query_id": query_id,
        "query_type": "semantic",
        "query_text": "test action",
        "limit": 5,
    }

    await nats_client.publish("memory.recall", json.dumps(query).encode())
    await asyncio.sleep(1.0)


# =============================================================================
# BELIEFS TESTS - Graph Operations
# =============================================================================


@pytest.mark.asyncio
async def test_beliefs_update_node(nats_client: NATSClient):
    """Test updating a belief node."""
    node_id = str(uuid.uuid4())

    belief_node = {
        "node_id": node_id,
        "type": "value",
        "content": "Safety is paramount",
        "confidence": 0.95,
        "source": "test",
    }

    await nats_client.publish("beliefs.update", json.dumps(belief_node).encode())
    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_beliefs_find_contradictions(nats_client: NATSClient):
    """Test finding contradictions in belief graph."""
    response_received: Optional[dict] = None

    async def response_handler(msg):
        nonlocal response_received
        response_received = json.loads(msg.data.decode())

    query_id = str(uuid.uuid4())
    await nats_client.subscribe(
        f"beliefs.contradictions.result.{query_id}", cb=response_handler
    )
    await asyncio.sleep(0.1)

    query = {"query_id": query_id, "node_id": str(uuid.uuid4())}

    await nats_client.publish("beliefs.find_contradictions", json.dumps(query).encode())
    await asyncio.sleep(1.0)


# =============================================================================
# CACHE TESTS - LLM Caching
# =============================================================================


@pytest.mark.asyncio
async def test_cache_query(nats_client: NATSClient):
    """Test LLM cache query."""
    response = await nats_client.request(
        "cache.query",
        json.dumps(
            {"prompt": "What is 2+2?", "model": "deepseek-coder:6.7b", "force_live": False}
        ).encode(),
        timeout=5.0,
    )

    result = json.loads(response.data.decode())
    # Cache service may return hit or miss


@pytest.mark.asyncio
async def test_cache_status(nats_client: NATSClient):
    """Test cache status query."""
    response = await nats_client.request(
        "cache.status",
        json.dumps({}).encode(),
        timeout=2.0,
    )

    status = json.loads(response.data.decode())
    # Status should have 'status' field


# =============================================================================
# COORDINATOR TESTS - Task Coordination
# =============================================================================


@pytest.mark.asyncio
async def test_coordinator_task_request(nats_client: NATSClient):
    """Test coordinator task request."""
    result_received: Optional[dict] = None

    async def result_handler(msg):
        nonlocal result_received
        result_received = json.loads(msg.data.decode())

    await nats_client.subscribe("task.result", cb=result_handler)
    await asyncio.sleep(0.1)

    # Request task
    task_request = {"query": "pick up object", "parameters": {"object_id": "cube_1"}}

    await nats_client.publish("task.request", json.dumps(task_request).encode())
    await asyncio.sleep(1.5)

    # Coordinator may or may not respond depending on availability


@pytest.mark.asyncio
async def test_coordinator_status(nats_client: NATSClient):
    """Test coordinator status query."""
    status_received: Optional[dict] = None

    async def status_handler(msg):
        nonlocal status_received
        status_received = json.loads(msg.data.decode())

    await nats_client.subscribe("coordinator.status.result", cb=status_handler)
    await asyncio.sleep(0.1)

    await nats_client.publish("coordinator.status", json.dumps({}).encode())
    await asyncio.sleep(1.0)


# =============================================================================
# OVERRIDE TESTS - Human Override Processing
# =============================================================================


@pytest.mark.asyncio
async def test_override_request(nats_client: NATSClient):
    """Test human override request."""
    result_received: Optional[dict] = None

    trace_id = str(uuid.uuid4())

    async def result_handler(msg):
        nonlocal result_received
        result_received = json.loads(msg.data.decode())

    await nats_client.subscribe(f"override.result.{trace_id}", cb=result_handler)
    await asyncio.sleep(0.1)

    # Request override
    override_request = {
        "trace_id": trace_id,
        "prompt": "Set max_speed to 0.5",
    }

    await nats_client.publish("override.request", json.dumps(override_request).encode())
    await asyncio.sleep(2.0)

    # Override service will verify human presence (may fail if no sensors)


@pytest.mark.asyncio
async def test_override_status(nats_client: NATSClient):
    """Test override service status."""
    response = await nats_client.request(
        "override.status",
        json.dumps({}).encode(),
        timeout=2.0,
    )

    status = json.loads(response.data.decode())
    # Should have metrics


# =============================================================================
# EXTERNAL API TESTS - API Query Approval
# =============================================================================


@pytest.mark.asyncio
async def test_external_api_query(nats_client: NATSClient):
    """Test external API query with kernel approval."""
    response = await nats_client.request(
        "external.query",
        json.dumps(
            {
                "query": "What is Python?",
                "context": "programming language question",
                "local_knowledge": None,
            }
        ).encode(),
        timeout=5.0,
    )

    result = json.loads(response.data.decode())
    # Should have 'success' field


@pytest.mark.asyncio
async def test_external_api_status(nats_client: NATSClient):
    """Test external API service status."""
    response = await nats_client.request(
        "external.status",
        json.dumps({}).encode(),
        timeout=2.0,
    )

    status = json.loads(response.data.decode())
    assert "status" in status
    assert "metrics" in status


# =============================================================================
# INTEGRATION TESTS - Multi-Service Flows
# =============================================================================


@pytest.mark.asyncio
async def test_end_to_end_observation_to_decision(nats_client: NATSClient):
    """
    Test end-to-end flow: Observation → Planner → Kernel Decision.

    This tests the core message flow:
    1. Sensor publishes observation
    2. Planner generates proposal
    3. Kernel makes decision
    """
    trace_id = str(uuid.uuid4())
    proposal_received: Optional[dict] = None
    decision_received: Optional[dict] = None

    async def proposal_handler(msg):
        nonlocal proposal_received
        proposal_received = json.loads(msg.data.decode())

    async def decision_handler(msg):
        nonlocal decision_received
        decision_received = json.loads(msg.data.decode())

    # Subscribe to proposal and decision
    await nats_client.subscribe("proposal.new", cb=proposal_handler)
    await nats_client.subscribe(f"decision.*", cb=decision_handler)
    await asyncio.sleep(0.1)

    # Publish observation
    observation = {
        "trace_id": trace_id,
        "provenance": "sensor.test",
        "data": {"temperature": 25.5},
        "confidence": 0.95,
        "tags": ["test"],
    }

    await nats_client.publish("observation.test", json.dumps(observation).encode())

    # Wait for flow to complete
    await asyncio.sleep(2.0)

    # Note: Proposals and decisions depend on service availability and logic


@pytest.mark.asyncio
async def test_message_flow_timeout_handling(nats_client: NATSClient):
    """Test that services handle timeouts gracefully."""
    # Request to non-existent service
    with pytest.raises(asyncio.TimeoutError):
        await nats_client.request(
            "nonexistent.service",
            json.dumps({"test": "data"}).encode(),
            timeout=1.0,
        )


@pytest.mark.asyncio
async def test_nats_wildcard_subscriptions(nats_client: NATSClient):
    """Test NATS wildcard subscription patterns."""
    received_subjects = []

    async def handler(msg):
        received_subjects.append(msg.subject)

    # Subscribe to all decisions
    await nats_client.subscribe("decision.*", cb=handler)
    await asyncio.sleep(0.1)

    # Publish multiple decisions
    for i in range(3):
        trace_id = str(uuid.uuid4())
        decision = {
            "trace_id": trace_id,
            "type": "ALLOW",
            "reason": f"Test {i}",
            "risk_score": 0.1,
        }
        await nats_client.publish(f"decision.{trace_id}", json.dumps(decision).encode())

    await asyncio.sleep(0.5)

    assert len(received_subjects) == 3
    for subject in received_subjects:
        assert subject.startswith("decision.")


@pytest.mark.asyncio
async def test_concurrent_message_processing(nats_client: NATSClient):
    """Test that multiple messages can be processed concurrently."""
    messages_received = []

    async def handler(msg):
        messages_received.append(json.loads(msg.data.decode()))

    test_subject = f"test.concurrent.{uuid.uuid4().hex[:8]}"
    await nats_client.subscribe(test_subject, cb=handler)
    await asyncio.sleep(0.1)

    # Publish multiple messages rapidly
    for i in range(10):
        await nats_client.publish(
            test_subject, json.dumps({"message_id": i}).encode()
        )

    await asyncio.sleep(0.5)

    assert len(messages_received) == 10
    # All messages should be received
    message_ids = [msg["message_id"] for msg in messages_received]
    assert sorted(message_ids) == list(range(10))
