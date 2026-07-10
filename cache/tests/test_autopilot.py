"""Tests for AutopilotController's producer-side tag threading.

Uses asyncio.run (no pytest-asyncio) to run under the bare-pytest governance lane.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from cache.autopilot import AutopilotController


def _controller(cached=None):
    llm_cache = MagicMock()
    llm_cache.get = AsyncMock(return_value=cached)
    llm_cache.set = AsyncMock()
    bus = MagicMock()
    bus.publish = AsyncMock()
    return AutopilotController(event_bus=bus, llm_cache=llm_cache), llm_cache


def test_query_llm_caches_miss_with_tags():
    """A cache miss falls back to live and stores the response under its tags."""
    ctrl, llm_cache = _controller(cached=None)
    ctrl.enable()

    result = asyncio.run(ctrl.query_llm("p", tags=["task_query"]))

    assert result["source"] == "live"
    llm_cache.set.assert_awaited_once()
    assert llm_cache.set.call_args.kwargs["tags"] == ["task_query"]


def test_query_llm_disabled_returns_live_without_caching():
    """While disabled, autopilot neither reads nor writes the cache."""
    ctrl, llm_cache = _controller()

    result = asyncio.run(ctrl.query_llm("p", tags=["task_query"]))

    assert result["source"] == "live"
    llm_cache.get.assert_not_called()
    llm_cache.set.assert_not_called()
