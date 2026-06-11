"""
Integration tests for NATS message flow.

Tests basic pub/sub patterns and message routing across services.
"""

import asyncio
import json
import uuid
from typing import Any

import pytest
from nats.aio.client import Client as NATSClient


@pytest.mark.asyncio
async def test_nats_connection(nats_client: NATSClient):
    """Test that NATS connection is established."""
    assert nats_client.is_connected


@pytest.mark.asyncio
async def test_basic_pub_sub(nats_client: NATSClient):
    """Test basic publish/subscribe pattern."""
    test_subject = "test.basic.pubsub"
    test_message = {"test": "data", "timestamp": 1234567890}
    received_message = None

    async def message_handler(msg):
        nonlocal received_message
        received_message = json.loads(msg.data.decode())

    # Subscribe
    await nats_client.subscribe(test_subject, cb=message_handler)

    # Give subscription time to establish
    await asyncio.sleep(0.1)

    # Publish
    await nats_client.publish(
        test_subject,
        json.dumps(test_message).encode()
    )

    # Wait for message
    await asyncio.sleep(0.2)

    assert received_message is not None
    assert received_message["test"] == "data"
    assert received_message["timestamp"] == 1234567890


@pytest.mark.asyncio
async def test_wildcard_subscription(nats_client: NATSClient):
    """Test wildcard subscription pattern."""
    base_subject = f"test.wildcard.{uuid.uuid4().hex[:8]}"
    received_subjects = []

    async def message_handler(msg):
        received_subjects.append(msg.subject)

    # Subscribe with wildcard
    await nats_client.subscribe(f"{base_subject}.*", cb=message_handler)

    await asyncio.sleep(0.1)

    # Publish to multiple subjects
    for suffix in ["alpha", "beta", "gamma"]:
        await nats_client.publish(
            f"{base_subject}.{suffix}",
            b"test"
        )

    await asyncio.sleep(0.2)

    assert len(received_subjects) == 3
    assert f"{base_subject}.alpha" in received_subjects
    assert f"{base_subject}.beta" in received_subjects
    assert f"{base_subject}.gamma" in received_subjects


@pytest.mark.asyncio
async def test_request_reply(nats_client: NATSClient):
    """Test request/reply pattern."""
    test_subject = f"test.request.reply.{uuid.uuid4().hex[:8]}"

    async def reply_handler(msg):
        # Echo back the request with "reply_" prefix
        request_data = json.loads(msg.data.decode())
        response = {"reply_to": request_data, "status": "ok"}
        await nats_client.publish(msg.reply, json.dumps(response).encode())

    # Setup reply handler
    await nats_client.subscribe(test_subject, cb=reply_handler)

    await asyncio.sleep(0.1)

    # Send request
    request_data = {"action": "test", "id": str(uuid.uuid4())}
    response = await nats_client.request(
        test_subject,
        json.dumps(request_data).encode(),
        timeout=2.0
    )

    response_data = json.loads(response.data.decode())
    assert response_data["status"] == "ok"
    assert response_data["reply_to"]["action"] == "test"


@pytest.mark.asyncio
async def test_observation_subject_pattern(nats_client: NATSClient):
    """Test observation.* subject pattern."""
    observations_received = []

    async def observation_handler(msg):
        data = json.loads(msg.data.decode())
        observations_received.append({
            "subject": msg.subject,
            "data": data
        })

    # Subscribe to all observations
    await nats_client.subscribe("observation.*", cb=observation_handler)

    await asyncio.sleep(0.1)

    # Publish test observations
    test_trace_id = str(uuid.uuid4())

    observation = {
        "trace_id": test_trace_id,
        "provenance": "sensor.camera.front",
        "data": {"motion_detected": True},
        "timestamp": 1234567890,
        "confidence": 0.95,
        "tags": ["motion"]
    }

    await nats_client.publish(
        "observation.camera",
        json.dumps(observation).encode()
    )

    await asyncio.sleep(0.2)

    assert len(observations_received) == 1
    obs = observations_received[0]
    assert obs["subject"] == "observation.camera"
    assert obs["data"]["trace_id"] == test_trace_id
    assert obs["data"]["data"]["motion_detected"] is True


@pytest.mark.asyncio
async def test_decision_trace_id_pattern(nats_client: NATSClient):
    """Test decision.{trace_id} dynamic subject pattern."""
    test_trace_id = str(uuid.uuid4())
    decision_subject = f"decision.{test_trace_id}"

    decision_received = None

    async def decision_handler(msg):
        nonlocal decision_received
        decision_received = json.loads(msg.data.decode())

    # Subscribe to specific trace_id decision
    await nats_client.subscribe(decision_subject, cb=decision_handler)

    await asyncio.sleep(0.1)

    # Publish decision
    decision = {
        "trace_id": test_trace_id,
        "type": "ALLOW",
        "reason": "Action approved within safety bounds",
        "risk_score": 0.15,
        "issued_at": 1234567890,
        "expires_at": 1234567950
    }

    await nats_client.publish(
        decision_subject,
        json.dumps(decision).encode()
    )

    await asyncio.sleep(0.2)

    assert decision_received is not None
    assert decision_received["trace_id"] == test_trace_id
    assert decision_received["type"] == "ALLOW"
    assert decision_received["risk_score"] == 0.15


@pytest.mark.asyncio
async def test_multiple_subscribers(nats_client: NATSClient):
    """Test that multiple subscribers receive the same message."""
    test_subject = f"test.multi.{uuid.uuid4().hex[:8]}"
    received_count = [0, 0]  # Two subscribers

    async def handler_1(msg):
        received_count[0] += 1

    async def handler_2(msg):
        received_count[1] += 1

    # Multiple subscribers to same subject
    await nats_client.subscribe(test_subject, cb=handler_1)
    await nats_client.subscribe(test_subject, cb=handler_2)

    await asyncio.sleep(0.1)

    # Publish once
    await nats_client.publish(test_subject, b"test message")

    await asyncio.sleep(0.2)

    # Both subscribers should receive
    assert received_count[0] == 1
    assert received_count[1] == 1
