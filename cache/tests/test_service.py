"""Tests for CacheService request handlers (producer-side tag threading).

Uses asyncio.run (no pytest-asyncio) to run under the bare-pytest governance lane.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cache.service import CacheService


def test_handle_cache_query_forwards_tags():
    """`cache.query` passes the request's tags through to the autopilot."""
    # The handler only touches `_autopilot`, so a stand-in self avoids building
    # the full service (NATS/DB).
    autopilot = MagicMock()
    autopilot.query_llm = AsyncMock(return_value={"response": "x"})
    service = SimpleNamespace(_autopilot=autopilot)

    msg = MagicMock()
    msg.reply = "inbox"
    msg.respond = AsyncMock()

    asyncio.run(
        CacheService._handle_cache_query(service, {"prompt": "p", "tags": ["configuration"]}, msg)
    )

    assert autopilot.query_llm.call_args.kwargs["tags"] == ["configuration"]
    msg.respond.assert_awaited_once()


def test_handle_cache_query_without_reply_skips_response():
    """A fire-and-forget query (no reply subject) produces no response."""
    autopilot = MagicMock()
    autopilot.query_llm = AsyncMock(return_value={"response": "x"})
    service = SimpleNamespace(_autopilot=autopilot)

    msg = MagicMock()
    msg.reply = None
    msg.respond = AsyncMock()

    asyncio.run(CacheService._handle_cache_query(service, {"prompt": "p"}, msg))

    msg.respond.assert_not_called()
