"""
Cache Service - LLM response caching and autopilot mode.

Manages:
- LLM response caching with semantic similarity
- Autopilot mode configuration
- Cache invalidation rules
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Optional

import aiosqlite
import nats

from cache.llm_cache import LLMCache
from cache.autopilot import AutopilotController
from cache.invalidator import CacheInvalidator

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CacheService:
    """
    LLM Cache service with autopilot mode.
    """

    def __init__(self):
        self.nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
        self.qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
        self.sqlite_path = os.environ.get("SQLITE_PATH", "/data/sqlite/unified.db")

        self._nc: Optional[nats.aio.client.Client] = None
        self._db: Optional[aiosqlite.Connection] = None
        self._shutdown_event = asyncio.Event()

        self._llm_cache: Optional[LLMCache] = None
        self._autopilot: Optional[AutopilotController] = None
        self._invalidator: Optional[CacheInvalidator] = None

    async def start(self) -> None:
        """Start the cache service."""
        logger.info("Starting LLM Cache Service...")

        # Connect to NATS
        logger.info(f"Connecting to NATS at {self.nats_url}")
        self._nc = await nats.connect(self.nats_url)

        # Connect to SQLite
        logger.info(f"Connecting to SQLite at {self.sqlite_path}")
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self._db = await aiosqlite.connect(self.sqlite_path)

        # Initialize LLM cache
        self._llm_cache = LLMCache(
            qdrant_url=self.qdrant_url,
            ollama_url=self.ollama_url,
            db=self._db,
        )

        # Initialize autopilot controller
        self._autopilot = AutopilotController(
            nats_client=self._nc,
            llm_cache=self._llm_cache,
        )

        # Initialize cache invalidator
        self._invalidator = CacheInvalidator(
            nats_client=self._nc,
            llm_cache=self._llm_cache,
            db=self._db,
        )
        await self._invalidator.start()

        # Subscribe to NATS subjects
        await self._nc.subscribe("cache.query", cb=self._handle_cache_query)
        await self._nc.subscribe("cache.setting", cb=self._handle_cache_setting)
        await self._nc.subscribe("autopilot.setting", cb=self._handle_autopilot_setting)
        await self._nc.subscribe("cache.status", cb=self._handle_status)

        logger.info("LLM Cache Service started successfully")

    async def stop(self) -> None:
        """Stop the cache service."""
        logger.info("Stopping LLM Cache Service...")

        if self._invalidator:
            await self._invalidator.stop()

        if self._nc:
            await self._nc.drain()
            await self._nc.close()

        if self._db:
            await self._db.close()

        logger.info("LLM Cache Service stopped")

    async def run(self) -> None:
        """Run the service until shutdown."""
        await self.start()
        await self._shutdown_event.wait()
        await self.stop()

    def shutdown(self) -> None:
        """Signal the service to shut down."""
        self._shutdown_event.set()

    async def _handle_cache_query(self, msg) -> None:
        """Handle cache query request."""
        try:
            request = json.loads(msg.data.decode())
            prompt = request.get("prompt", "")
            model = request.get("model", "deepseek-coder:6.7b")
            force_live = request.get("force_live", False)

            # Query via autopilot controller
            result = await self._autopilot.query_llm(
                prompt=prompt,
                model=model,
                force_live=force_live,
            )

            if msg.reply:
                await self._nc.publish(msg.reply, json.dumps(result).encode())

        except Exception as e:
            logger.error(f"Error handling cache query: {e}", exc_info=True)

    async def _handle_cache_setting(self, msg) -> None:
        """Handle cache enable/disable."""
        try:
            request = json.loads(msg.data.decode())
            enabled = request.get("enabled", True)

            # Note: Cache is always available, this setting is for autopilot
            logger.info(f"Cache setting: {'enabled' if enabled else 'disabled'}")

        except Exception as e:
            logger.error(f"Error handling cache setting: {e}")

    async def _handle_autopilot_setting(self, msg) -> None:
        """Handle autopilot enable/disable."""
        try:
            request = json.loads(msg.data.decode())
            enabled = request.get("enabled", True)

            if enabled:
                self._autopilot.enable()
            else:
                self._autopilot.disable()

            logger.info(f"Autopilot: {'ENABLED' if enabled else 'DISABLED'}")

        except Exception as e:
            logger.error(f"Error handling autopilot setting: {e}")

    async def _handle_status(self, msg) -> None:
        """Handle status request."""
        try:
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
                await self._nc.publish(msg.reply, json.dumps(status).encode())

        except Exception as e:
            logger.error(f"Error getting status: {e}")


async def main() -> None:
    """Main entry point."""
    service = CacheService()

    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        service.shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows: add_signal_handler unsupported; Ctrl+C still works

    try:
        await service.run()
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
