"""
Knowledge base — the data flywheel.

Tracks what Engram learns and from which of its four data sources. Pure
stdlib, no web-stack dependency, so it can be unit-tested without FastAPI.
"""

import time
import uuid
from collections import deque
from datetime import datetime

from dashboard.util import now_iso


class KnowledgeBase:
    """
    In-memory knowledge base tracking the data flywheel.

    Engram learns from 4 sources:
    1. Simulation (synthetic experiences)
    2. Internet/external data (observation)
    3. Teleoperation (human guidance — chat)
    4. Real-world deployments (self-generated data)
    """

    def __init__(self):
        self._entries: deque = deque(maxlen=1000)
        self._source_counts = {
            "teleoperation": 0,  # Chat interactions
            "observation": 0,  # System observations, NATS messages
            "deployment": 0,  # Self-generated from monitoring
            "simulation": 0,  # Synthetic / test data
        }
        self._total_interactions = 0

    def __len__(self) -> int:
        return len(self._entries)

    def learn(self, source: str, category: str, content: str, metadata: dict = None):
        """Record a learning event."""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "source": source,
            "category": category,
            "content": content,
            "metadata": metadata or {},
            "timestamp": now_iso(),
        }
        self._entries.append(entry)
        if source in self._source_counts:
            self._source_counts[source] += 1
        self._total_interactions += 1

    def get_flywheel_stats(self) -> dict:
        """Get data flywheel statistics."""
        return {
            "total_knowledge_entries": len(self._entries),
            "total_interactions": self._total_interactions,
            "sources": dict(self._source_counts),
            "recent_entries": list(self._entries)[-10:],
            "growth_rate": self._calculate_growth_rate(),
        }

    def _calculate_growth_rate(self) -> float:
        """Entries per hour over last hour."""
        if not self._entries:
            return 0.0
        now = time.time()
        one_hour_ago = now - 3600
        recent = sum(
            1
            for e in self._entries
            if datetime.fromisoformat(e["timestamp"]).timestamp() > one_hour_ago
        )
        return round(recent, 1)

    def get_entries(self, limit: int = 50, source: str = None) -> list[dict]:
        entries = list(self._entries)
        if source:
            entries = [e for e in entries if e["source"] == source]
        return entries[-limit:]
