"""Pytest configuration and fixtures for integration tests."""

import asyncio
import os
from collections.abc import AsyncGenerator

import pytest
from nats.aio.client import Client as NATSClient


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def nats_client() -> AsyncGenerator[NATSClient, None]:
    """
    Provide a NATS client for testing.

    Connects to NATS server and yields the client, then closes on cleanup.
    """
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    nc = await NATSClient().connect(nats_url)
    yield nc
    await nc.drain()
    await nc.close()


@pytest.fixture
def nats_url() -> str:
    """Get NATS URL from environment."""
    return os.environ.get("NATS_URL", "nats://localhost:4222")


@pytest.fixture
def qdrant_url() -> str:
    """Get Qdrant URL from environment."""
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


@pytest.fixture
def sqlite_path() -> str:
    """Get SQLite database path from environment."""
    return os.environ.get("SQLITE_PATH", "/data/sqlite/unified.db")
