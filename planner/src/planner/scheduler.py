"""
Scheduler - Manages operational modes and action prioritization.

Supports modes: EXECUTION, LEARNING, EXPLORATION, SAFE_HALT
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time

logger = logging.getLogger(__name__)


class SchedulerMode(Enum):
    """Operating modes for the scheduler."""
    EXECUTION = "EXECUTION"     # Normal operation
    LEARNING = "LEARNING"       # Prioritize learning/training
    EXPLORATION = "EXPLORATION" # Curiosity-driven exploration
    SAFE_HALT = "SAFE_HALT"     # Emergency stop, cancel all pending


@dataclass
class PendingAction:
    """A pending action in the queue."""
    trace_id: str
    priority: int
    proposal: dict[str, Any]
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    expires_at: Optional[int] = None


class Scheduler:
    """
    Priority scheduler with mode-based behavior.

    Manages action queues and handles mode transitions,
    including emergency SAFE_HALT.
    """

    def __init__(self):
        self._mode = SchedulerMode.EXECUTION
        self._pending: list[PendingAction] = []
        self._mode_lock = asyncio.Lock()
        self._mode_change_callbacks: list[callable] = []

    @property
    def mode(self) -> SchedulerMode:
        """Get current scheduler mode."""
        return self._mode

    async def set_mode(self, mode: SchedulerMode) -> None:
        """
        Set the scheduler mode.

        If switching to SAFE_HALT, all pending actions are cancelled.
        """
        async with self._mode_lock:
            old_mode = self._mode
            self._mode = mode

            logger.info(f"Scheduler mode changed: {old_mode.value} -> {mode.value}")

            if mode == SchedulerMode.SAFE_HALT:
                cancelled_count = len(self._pending)
                self._pending.clear()
                logger.warning(f"SAFE_HALT activated: cancelled {cancelled_count} pending actions")

            # Notify callbacks
            for callback in self._mode_change_callbacks:
                try:
                    await callback(old_mode, mode)
                except Exception as e:
                    logger.error(f"Mode change callback error: {e}")

    def on_mode_change(self, callback: callable) -> None:
        """Register a callback for mode changes."""
        self._mode_change_callbacks.append(callback)

    async def enqueue(self, action: PendingAction) -> bool:
        """
        Enqueue an action for execution.

        Returns False if in SAFE_HALT mode.
        """
        if self._mode == SchedulerMode.SAFE_HALT:
            logger.warning(f"Rejecting action {action.trace_id}: SAFE_HALT active")
            return False

        # Adjust priority based on mode
        adjusted_priority = self._adjust_priority(action)
        action.priority = adjusted_priority

        # Insert in priority order (higher priority first)
        inserted = False
        for i, pending in enumerate(self._pending):
            if action.priority > pending.priority:
                self._pending.insert(i, action)
                inserted = True
                break

        if not inserted:
            self._pending.append(action)

        logger.debug(f"Enqueued action {action.trace_id} with priority {action.priority}")
        return True

    async def dequeue(self) -> Optional[PendingAction]:
        """
        Dequeue the highest priority action.

        Returns None if queue is empty or in SAFE_HALT mode.
        """
        if self._mode == SchedulerMode.SAFE_HALT:
            return None

        if not self._pending:
            return None

        # Remove expired actions
        now = int(time.time() * 1000)
        self._pending = [
            a for a in self._pending
            if a.expires_at is None or a.expires_at > now
        ]

        if not self._pending:
            return None

        return self._pending.pop(0)

    def _adjust_priority(self, action: PendingAction) -> int:
        """Adjust priority based on current mode."""
        base_priority = action.priority

        if self._mode == SchedulerMode.LEARNING:
            # Boost learning-related actions
            if "learning" in action.proposal.get("tags", []):
                return base_priority + 100

        elif self._mode == SchedulerMode.EXPLORATION:
            # Boost exploration actions
            if "exploration" in action.proposal.get("tags", []):
                return base_priority + 100

        return base_priority

    @property
    def pending_count(self) -> int:
        """Get number of pending actions."""
        return len(self._pending)

    async def clear_pending(self) -> int:
        """Clear all pending actions. Returns count cleared."""
        count = len(self._pending)
        self._pending.clear()
        return count

    def get_queue_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "mode": self._mode.value,
            "pending_count": len(self._pending),
            "oldest_action": self._pending[0].trace_id if self._pending else None,
        }
