"""
Cache Service - LLM response caching and autopilot mode.

Manages:
- LLM response caching with semantic similarity
- Autopilot mode configuration
- Cache invalidation rules
"""

import asyncio
from typing import Optional

from activelearning import BaseService
from activelearning.nats_client import serialize_message

from cache.llm_cache import LLMCache
from cache.autopilot import AutopilotController
from cache.invalidator import CacheInvalidator


class CacheService(BaseService):
    """
    LLM Cache service with autopilot mode.
    """

    def __init__(self):
        super().__init__("cache", use_database=True, use_event_bus=True)

        self._llm_cache: Optional[LLMCache] = None
        self._autopilot: Optional[AutopilotController] = None
        self._invalidator: Optional[CacheInvalidator] = None

    async def _setup(self) -> None:
        """Service-specific setup."""
        self._llm_cache = LLMCache(
            qdrant_url=self.config.qdrant_url,
            ollama_url=self.config.ollama_url,
            db=self.database,
        )

        self._autopilot = AutopilotController(
            event_bus=self.event_bus,
            llm_cache=self._llm_cache,
        )

        self._invalidator = CacheInvalidator(
            event_bus=self.event_bus,
            llm_cache=self._llm_cache,
            db=self.database,
        )
        await self._invalidator.start()

        await self.event_bus.subscribe(
            "cache.query",
            self._handle_cache_query,
            is_request_handler=True,
        )
        await self.event_bus.subscribe("cache.setting", self._handle_cache_setting)
        await self.event_bus.subscribe("autopilot.setting", self._handle_autopilot_setting)
        await self.event_bus.subscribe(
            "cache.status",
            self._handle_status,
            is_request_handler=True,
        )

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        if self._invalidator:
            await self._invalidator.stop()

    async def _handle_cache_query(self, data: dict, msg) -> None:
        """Handle cache query request."""
        prompt = data.get("prompt", "")
        model = data.get("model", "deepseek-coder:6.7b")
        force_live = data.get("force_live", False)

        result = await self._autopilot.query_llm(
            prompt=prompt,
            model=model,
            force_live=force_live,
        )

        if msg.reply:
            await msg.respond(serialize_message(result))

    async def _handle_cache_setting(self, data: dict) -> None:
        """Handle cache enable/disable."""
        enabled = data.get("enabled", True)
        self.logger.info(f"Cache setting: {'enabled' if enabled else 'disabled'}")

    async def _handle_autopilot_setting(self, data: dict) -> None:
        """Handle autopilot enable/disable."""
        enabled = data.get("enabled", True)

        if enabled:
            self._autopilot.enable()
        else:
            self._autopilot.disable()

        self.logger.info(f"Autopilot: {'ENABLED' if enabled else 'DISABLED'}")

    async def _handle_status(self, _data: dict, msg) -> None:
        """Handle status request."""
        status = {
            "status": "running",
            "autopilot": self._autopilot.get_status(),
            "cache": {
                "collection": self._llm_cache.collection_name,
                "hit_threshold": self._llm_cache.hit_threshold,
            },
            "invalidator": {
                "max_age_days": self._invalidator.max_age_days,
                "unused_age_days": self._invalidator.unused_age_days,
            },
        }

        if msg.reply:
            await msg.respond(serialize_message(status))


async def main() -> None:
    """Main entry point."""
    service = CacheService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
