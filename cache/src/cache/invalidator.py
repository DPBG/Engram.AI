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
from collections.abc import Iterable
from typing import Any

from activelearning import current_timestamp
from activelearning.nats_client import EventBus

from cache.llm_cache import CacheTag

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
        await self._invalidate_by_tag(CacheTag.CODE_GENERATION)

    async def _on_override_applied(self, data: dict) -> None:
        """Invalidate cache when override is applied."""
        parameter = data.get("parameter", "")
        logger.info(f"Override applied to {parameter}: invalidating cache")
        await self._invalidate_by_tag(CacheTag.CONFIGURATION)

    async def _on_task_saved(self, data: dict) -> None:
        """Invalidate cache when new task is saved."""
        task_id = data.get("task_id", "")
        logger.info(f"Task saved: {task_id}")
        await self._invalidate_by_tag(CacheTag.TASK_QUERY)

    async def _invalidate_by_tag(self, tag: str) -> int:
        """Evict every entry carrying ``tag``; returns how many were removed.

        ``json_each`` gives exact membership (so ``task_query`` won't match
        ``task_query_extended``) and skips NULL (untagged) rows.
        """
        try:
            cursor = await self.db.execute(
                """
                SELECT DISTINCT c.prompt_hash
                FROM llm_cache c, json_each(c.tags) t
                WHERE t.value = ?
                """,
                (tag,),
            )
            rows = await cursor.fetchall()
            deleted = await self._delete_hashes(row[0] for row in rows)
        except Exception as e:
            logger.error(f"Error invalidating cache by tag '{tag}': {e}", exc_info=True)
            return 0

        logger.info(f"Invalidated {deleted} cache entries tagged '{tag}'")
        return deleted

    async def _delete_hashes(self, prompt_hashes: Iterable[str]) -> int:
        """Evict each hash via :meth:`LLMCache.invalidate`; returns the count.
        Shared by every invalidation path so they evict identically."""
        deleted = 0
        for prompt_hash in prompt_hashes:
            if await self.llm_cache.invalidate(prompt_hash):
                deleted += 1
        return deleted

    async def _periodic_cleanup(self) -> None:
        """Run age-based eviction once an hour for the life of the service."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._evict_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}", exc_info=True)

    async def _evict_expired(self) -> int:
        """Evict entries past ``max_age_days`` or unused for ``unused_age_days``.

        Returns the number of entries evicted.
        """
        logger.info("Running periodic cache cleanup...")

        now = current_timestamp()
        max_age_ms = self.max_age_days * 24 * 60 * 60 * 1000
        unused_age_ms = self.unused_age_days * 24 * 60 * 60 * 1000

        cursor = await self.db.execute(
            """
            SELECT prompt_hash
            FROM llm_cache
            WHERE
                (cached_at < ?) OR
                (COALESCE(last_hit_at, cached_at) < ?)
            """,
            (now - max_age_ms, now - unused_age_ms),
        )
        rows = await cursor.fetchall()

        deleted_count = await self._delete_hashes(row[0] for row in rows)
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} old cache entries")
        return deleted_count
