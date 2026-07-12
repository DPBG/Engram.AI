"""Tests for AutopilotController's producer-side tag threading.

Uses asyncio.run (no pytest-asyncio) to run under the bare-pytest governance lane.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from cache.autopilot import AutopilotController


def _controller(cached=None, llm_response="generated answer"):
    llm_cache = MagicMock()
    llm_cache.get = AsyncMock(return_value=cached)
    llm_cache.set = AsyncMock()
    bus = MagicMock()
    bus.publish = AsyncMock()
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(return_value=llm_response)
    return (
        AutopilotController(
            event_bus=bus,
            llm_cache=llm_cache,
            llm_client=llm_client,
        ),
        llm_cache,
        llm_client,
    )


def test_query_llm_caches_miss_with_tags():
    """A cache miss falls back to live and stores the response under its tags."""
    ctrl, llm_cache, llm_client = _controller(cached=None)
    ctrl.enable()

    result = asyncio.run(ctrl.query_llm("p", tags=["task_query"]))

    assert result["source"] == "live"
    assert result["response"] == "generated answer"
    llm_client.generate.assert_awaited_once_with("p", model="deepseek-coder:6.7b")
    llm_cache.set.assert_awaited_once()
    assert llm_cache.set.call_args.kwargs["tags"] == ["task_query"]


def test_query_llm_disabled_returns_live_without_caching():
    """While disabled, autopilot neither reads nor writes the cache."""
    ctrl, llm_cache, llm_client = _controller()

    result = asyncio.run(ctrl.query_llm("p", tags=["task_query"]))

    assert result["source"] == "live"
    assert result["response"] == "generated answer"
    llm_cache.get.assert_not_called()
    llm_cache.set.assert_not_called()
    llm_client.generate.assert_awaited_once()


def test_query_llm_cache_hit_skips_live_llm():
    """A confident cache hit must not call the live LLM client."""
    cached = {
        "response": "cached answer",
        "confidence": 0.95,
        "cached_at": "2026-01-01T00:00:00",
    }
    ctrl, llm_cache, llm_client = _controller(cached=cached)
    ctrl.enable()

    result = asyncio.run(ctrl.query_llm("p"))

    assert result["source"] == "cache"
    assert result["response"] == "cached answer"
    llm_client.generate.assert_not_called()
    llm_cache.set.assert_not_called()
