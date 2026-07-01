"""
Autopilot Controller - Manages autopilot mode using cached knowledge.

Autopilot mode:
- Prefers cached tasks and LLM responses
- Falls back to live LLM for novel situations
- Configurable enable/disable
"""

import logging
from typing import Any, Dict, List, Optional

from activelearning.nats_client import EventBus

logger = logging.getLogger(__name__)


class AutopilotController:
    """
    Controls autopilot mode for the system.

    In autopilot mode, the system prefers cached knowledge and
    minimizes live LLM calls.
    """

    def __init__(self, event_bus: EventBus, llm_cache: Any):
        self.event_bus = event_bus
        self.llm_cache = llm_cache

        self._enabled = False
        self._confidence_threshold = 0.9  # Only use cache if very confident

    def enable(self) -> None:
        """Enable autopilot mode."""
        logger.info("Autopilot mode: ENABLED")
        self._enabled = True

    def disable(self) -> None:
        """Disable autopilot mode."""
        logger.info("Autopilot mode: DISABLED")
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if autopilot is enabled."""
        return self._enabled

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for autopilot decisions."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self._confidence_threshold = threshold
        logger.info(f"Autopilot confidence threshold: {threshold}")

    async def query_llm(
        self,
        prompt: str,
        model: str = "deepseek-coder:6.7b",
        force_live: bool = False,
        tags: Optional[List[str]] = None,
    ) -> Dict:
        """
        Query LLM with autopilot caching.

        Args:
            prompt: LLM prompt
            model: Model name
            force_live: Force live LLM call, bypass cache
            tags: Invalidation tags stored with a freshly cached response so the
                CacheInvalidator can evict it on relevant system events.

        Returns:
            dict with response and metadata
        """
        # Check cache first (unless force_live)
        if self._enabled and not force_live:
            cached = await self.llm_cache.get(prompt, model)
            if cached and cached["confidence"] >= self._confidence_threshold:
                logger.info(f"Autopilot: using cached response (confidence: {cached['confidence']:.3f})")

                # Publish cache hit event
                await self._publish_cache_event("hit", prompt, cached["confidence"])

                return {
                    "response": cached["response"],
                    "source": "cache",
                    "confidence": cached["confidence"],
                    "cached_at": cached["cached_at"],
                }

        # Cache miss or forced live - call LLM
        logger.info("Autopilot: calling live LLM")

        # Publish cache miss event
        await self._publish_cache_event("miss", prompt, 0.0)

        # TODO: Make actual LLM call (integrate with Meta-Programmer's Ollama client)
        # For now, return placeholder
        response = "Live LLM response (not implemented)"

        # Cache the response
        if self._enabled:
            await self.llm_cache.set(prompt, response, model, tags=tags)

        return {
            "response": response,
            "source": "live",
            "confidence": 1.0,
            "cached_at": None,
        }

    async def _publish_cache_event(
        self,
        event_type: str,
        prompt: str,
        confidence: float,
    ) -> None:
        """Publish cache hit/miss event."""
        try:
            await self.event_bus.publish(
                f"llm.cache.{event_type}",
                {
                    "prompt_preview": prompt[:100],
                    "confidence": confidence,
                },
            )
        except Exception as e:
            logger.error(f"Error publishing cache event: {e}")

    def get_status(self) -> Dict:
        """Get autopilot status."""
        return {
            "enabled": self._enabled,
            "confidence_threshold": self._confidence_threshold,
            "cache_metrics": self.llm_cache.get_metrics(),
        }
