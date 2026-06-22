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

from activelearning.nats_client import EventBus

logger = logging.getLogger(__name__)


class CacheInvalidator:
    """
    Manages cache invalidation based on system events and age.
    """

    _SUBJECTS = ("code.deployed", "override.applied.*", "task.saved")

    def __init__(self, event_bus: EventBus, llm_cache: Any, db: Any):
        self.event_bus = event_bus
        self.llm_cache = llm_cache
        self.db = db

        # Configuration
        self.max_age_days = 7  # Maximum cache entry age
        self.unused_age_days = 3  # Delete if not used in this many days

        self._running = False
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the invalidator."""
        logger.info("Starting cache invalidator...")

        self._running = True

        await self.event_bus.subscribe("code.deployed", self._on_code_deployed)
        await self.event_bus.subscribe("override.applied.*", self._on_override_applied)
        await self.event_bus.subscribe("task.saved", self._on_task_saved)

        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

        logger.info("Cache invalidator started")

    async def stop(self) -> None:
        """Stop the invalidator."""
        self._running = False

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        for subject in self._SUBJECTS:
            await self.event_bus.unsubscribe(subject)

        logger.info("Cache invalidator stopped")

    async def _on_code_deployed(self, data: dict) -> None:
        """Invalidate cache when code is deployed."""
        logger.info(f"Code deployed: invalidating related cache entries ({data})")
        await self._invalidate_by_tag("code_generation")

    async def _on_override_applied(self, data: dict) -> None:
        """Invalidate cache when override is applied."""
        parameter = data.get("parameter", "")
        logger.info(f"Override applied to {parameter}: invalidating cache")
        await self._invalidate_by_tag("configuration")

    async def _on_task_saved(self, data: dict) -> None:
        """Invalidate cache when new task is saved."""
        task_id = data.get("task_id", "")
        logger.info(f"Task saved: {task_id}")
        await self._invalidate_by_tag("task_query")

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

            except asyncio.CancelledError:
                break
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
