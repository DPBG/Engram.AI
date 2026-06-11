"""
Cache Invalidator - Manages cache invalidation rules.

Invalidates cache entries when:
- Code is deployed
- Overrides are applied
- New tasks are saved
- Entries exceed age limit
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class CacheInvalidator:
    """
    Manages cache invalidation based on system events and age.
    """

    def __init__(self, nats_client: Any, llm_cache: Any, db: Any):
        self.nats_client = nats_client
        self.llm_cache = llm_cache
        self.db = db

        # Configuration
        self.max_age_days = 7  # Maximum cache entry age
        self.unused_age_days = 3  # Delete if not used in this many days

        self._running = False

    async def start(self) -> None:
        """Start the invalidator."""
        logger.info("Starting cache invalidator...")

        self._running = True

        # Subscribe to invalidation events
        await self.nats_client.subscribe("code.deployed", cb=self._on_code_deployed)
        await self.nats_client.subscribe("override.applied.*", cb=self._on_override_applied)
        await self.nats_client.subscribe("task.saved", cb=self._on_task_saved)

        # Start periodic cleanup task
        asyncio.create_task(self._periodic_cleanup())

        logger.info("Cache invalidator started")

    async def stop(self) -> None:
        """Stop the invalidator."""
        self._running = False
        logger.info("Cache invalidator stopped")

    async def _on_code_deployed(self, msg) -> None:
        """Invalidate cache when code is deployed."""
        try:
            import json
            data = json.loads(msg.data.decode())
            logger.info(f"Code deployed: invalidating related cache entries")

            # Invalidate all code generation related entries
            # TODO: More targeted invalidation based on what was deployed
            await self._invalidate_by_tag("code_generation")

        except Exception as e:
            logger.error(f"Error handling code deployment: {e}")

    async def _on_override_applied(self, msg) -> None:
        """Invalidate cache when override is applied."""
        try:
            import json
            data = json.loads(msg.data.decode())
            parameter = data.get("parameter", "")

            logger.info(f"Override applied to {parameter}: invalidating cache")

            # Invalidate entries that might be affected by the override
            await self._invalidate_by_tag("configuration")

        except Exception as e:
            logger.error(f"Error handling override: {e}")

    async def _on_task_saved(self, msg) -> None:
        """Invalidate cache when new task is saved."""
        try:
            import json
            data = json.loads(msg.data.decode())
            task_id = data.get("task_id", "")

            logger.info(f"Task saved: {task_id}")

            # Invalidate task-related queries
            await self._invalidate_by_tag("task_query")

        except Exception as e:
            logger.error(f"Error handling task save: {e}")

    async def _invalidate_by_tag(self, tag: str) -> None:
        """
        Invalidate cache entries by tag.

        TODO: Implement tag-based indexing in cache
        """
        logger.info(f"Invalidating cache entries tagged: {tag}")
        # For now, this is a no-op
        # In a full implementation, we'd query Qdrant for entries with this tag

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old cache entries."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                logger.info("Running periodic cache cleanup...")

                # Get current time
                now = int(time.time() * 1000)
                max_age_ms = self.max_age_days * 24 * 60 * 60 * 1000
                unused_age_ms = self.unused_age_days * 24 * 60 * 60 * 1000

                # Find old entries
                cursor = await self.db.execute(
                    """
                    SELECT prompt_hash, cached_at, last_hit_at
                    FROM llm_cache
                    WHERE
                        (cached_at < ?) OR
                        (last_hit_at IS NOT NULL AND last_hit_at < ?)
                    """,
                    (now - max_age_ms, now - unused_age_ms),
                )

                rows = await cursor.fetchall()

                deleted_count = 0
                for row in rows:
                    prompt_hash = row[0]
                    success = await self.llm_cache.invalidate(prompt_hash)
                    if success:
                        deleted_count += 1

                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} old cache entries")

            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}", exc_info=True)

    async def invalidate_all(self) -> int:
        """
        Invalidate all cache entries.

        Returns:
            Number of entries deleted
        """
        try:
            logger.warning("Invalidating ALL cache entries")

            # Get all prompt hashes
            cursor = await self.db.execute("SELECT prompt_hash FROM llm_cache")
            rows = await cursor.fetchall()

            deleted_count = 0
            for row in rows:
                prompt_hash = row[0]
                success = await self.llm_cache.invalidate(prompt_hash)
                if success:
                    deleted_count += 1

            logger.info(f"Deleted {deleted_count} cache entries")
            return deleted_count

        except Exception as e:
            logger.error(f"Error invalidating all: {e}", exc_info=True)
            return 0
