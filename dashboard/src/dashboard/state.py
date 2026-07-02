"""
Shared runtime state for the dashboard.

``DashboardState`` is the single source of truth the focused components (chat,
NATS, metrics, routers) share — rolling buffers, detected system info, cached
metrics, and the brain/service state streamed in over NATS. It replaces the
former module-level globals. Pure stdlib, so it stays unit-testable.
"""

import time
from collections import deque
from typing import Any

from dashboard.knowledge import KnowledgeBase
from dashboard.skills import SkillRegistry
from dashboard.util import now_iso

# ── capacity limits ────────────────────────────────────────────────────────
MAX_MESSAGES = 500
MAX_CHAT_HISTORY = 200
MAX_VIDEO_SESSIONS = 100
MAX_INSIGHTS = 100
MAX_DENY_ESCALATIONS = 50
MAX_PROBE_RESULTS = 200


class ConnectionManager:
    """Tracks active WebSocket clients and fans broadcasts out to all of them.

    Dead sockets (those that raise on send) are pruned automatically. The
    manager is truthy/sized by its connection count so callers can guard work
    with ``if connections:`` exactly as the old ``active_connections`` list did.
    """

    def __init__(self):
        self._connections: list = []

    def add(self, ws) -> None:
        self._connections.append(ws)

    def discard(self, ws) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    def __len__(self) -> int:
        return len(self._connections)

    def __bool__(self) -> bool:
        return bool(self._connections)

    async def broadcast(self, message: dict) -> None:
        """Send ``message`` (JSON) to every connected client; drop dead ones."""
        if not self._connections:
            return
        dead = []
        for c in list(self._connections):
            try:
                await c.send_json(message)
            except Exception:
                dead.append(c)
        for c in dead:
            self.discard(c)


class DashboardState:
    """All mutable runtime state for the dashboard, in one place."""

    def __init__(self):
        self.started_at: float = time.time()

        # WebSocket fan-out
        self.connections = ConnectionManager()

        # Core components
        self.skills = SkillRegistry()
        self.knowledge = KnowledgeBase()

        # Rolling buffers
        self.message_buffer: deque = deque(maxlen=MAX_MESSAGES)
        self.chat_history: list[dict] = []
        self.insights_log: deque = deque(maxlen=MAX_INSIGHTS)
        self.deny_escalations: deque = deque(maxlen=MAX_DENY_ESCALATIONS)

        # Detected info / cached metrics
        self.system_info: dict[str, Any] = {}
        self.system_metrics_cache: dict[str, Any] = {}
        self.last_metrics_update: float = 0.0

        # Brain / service state (updated via NATS)
        self.neuro_metrics: dict[str, Any] = {}
        self.gateway_status: dict[str, Any] = {}
        self.video_sessions: dict[str, dict] = {}  # session_id -> status dict
        self.watchdog_status: dict[str, Any] = {}
        self.service_status: dict[str, dict] = {}
        self.concept_probe_results: list[dict] = []

    # ── chat history ──────────────────────────────────────────────────────
    def append_chat(self, role: str, content: str) -> None:
        """Append a chat turn and trim to the most recent ``MAX_CHAT_HISTORY``."""
        self.chat_history.append(
            {
                "role": role,
                "content": content,
                "timestamp": now_iso(),
            }
        )
        if len(self.chat_history) > MAX_CHAT_HISTORY:
            self.chat_history[:] = self.chat_history[-MAX_CHAT_HISTORY:]

    # ── concept probe results ─────────────────────────────────────────────
    def add_probe_result(self, data: dict) -> None:
        """Store a concept-probe result, capped at ``MAX_PROBE_RESULTS``."""
        if len(self.concept_probe_results) >= MAX_PROBE_RESULTS:
            self.concept_probe_results.pop(0)
        self.concept_probe_results.append(data)

    # ── video training sessions ───────────────────────────────────────────
    def record_video_session(self, data: dict) -> None:
        """Store a real training session, pruning oldest finished ones if full.

        Transient/error markers (no real session id) are not stored.
        """
        sid = data.get("session_id")
        if not sid or sid in ("error", "download_error", "pending"):
            return
        self.video_sessions[sid] = data
        if len(self.video_sessions) > MAX_VIDEO_SESSIONS:
            removable = [
                (k, v)
                for k, v in self.video_sessions.items()
                if v.get("status") in ("stopped", "completed", "error")
            ]
            removable.sort(key=lambda x: x[1].get("created_at", 0))
            for k, _ in removable[: len(self.video_sessions) - MAX_VIDEO_SESSIONS]:
                self.video_sessions.pop(k, None)
