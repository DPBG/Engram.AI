"""
Cache Service - LLM response caching and autopilot mode.

Manages:
- LLM response caching with semantic similarity
- Autopilot mode configuration
- Cache invalidation rules
"""

import asyncio

from activelearning import BaseService, get_embedding_service
from activelearning.nats_client import serialize_message

from cache.autopilot import AutopilotController
from cache.invalidator import CacheInvalidator
from cache.llm_cache import LLMCache


class CacheService(BaseService):
    """
    LLM Cache service with autopilot mode.
    """

    def __init__(self):
        super().__init__("cache", use_database=True, use_event_bus=True)

        self._llm_cache: LLMCache | None = None
        self._autopilot: AutopilotController | None = None
        self._invalidator: CacheInvalidator | None = None
        # One shared embedding client per service, injected into the cache and
        # closed on shutdown.
        self._embedding_service = get_embedding_service()

    async def _setup(self) -> None:
        """Service-specific setup."""
        self._llm_cache = LLMCache(
            qdrant_url=self.config.qdrant_url,
            db=self.database,
            embedding_service=self._embedding_service,
        )
        await self._llm_cache.setup()

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
        if self._llm_cache:
            await self._llm_cache.close()
        await self._embedding_service.close()

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
