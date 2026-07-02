"""
Skill registry — abstracted capabilities the dashboard can invoke.

Each skill is an abstracted capability the system can invoke. Skills track
their own execution history and health. Pure stdlib, no web-stack dependency,
so it can be unit-tested without FastAPI.
"""

from collections import deque

from dashboard.util import now_iso


class SkillRegistry:
    """
    Modular skill registry.

    Each skill is an abstracted capability the system can invoke.
    Skills track their own execution history and health.
    "These skills are abstracted away using an API call."
    """

    def __init__(self):
        self._skills: dict[str, dict] = {}
        self._execution_log: deque = deque(maxlen=500)
        self._init_core_skills()

    def _init_core_skills(self):
        """Register the core built-in skills."""
        core = [
            {
                "id": "env.detect",
                "name": "Environment Detection",
                "category": "perception",
                "icon": "🔍",
                "description": "Server environment detection — OS, hardware, services, APIs",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "env.monitor",
                "name": "Resource Monitor",
                "category": "perception",
                "icon": "📊",
                "description": "Real-time CPU, memory, disk, network monitoring",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "env.docker",
                "name": "Container Orchestration",
                "category": "perception",
                "icon": "🐳",
                "description": "Docker container metrics, status, and lifecycle awareness",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "brain.chat",
                "name": "Brain Communication",
                "category": "cognition",
                "icon": "💬",
                "description": "LLM interface to the Engram neuromorphic brain (teleoperation channel)",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "brain.self_monitor",
                "name": "Self-Improvement Loop",
                "category": "cognition",
                "icon": "🔄",
                "description": "Periodic health checks, anomaly detection, optimization suggestions",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "brain.knowledge",
                "name": "Knowledge Base",
                "category": "memory",
                "icon": "🧠",
                "description": "Stores learnings from interactions, observations, and deployments",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "bus.nats",
                "name": "NATS Message Bus",
                "category": "communication",
                "icon": "📡",
                "description": "Inter-service messaging and event monitoring",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "bus.websocket",
                "name": "WebSocket Stream",
                "category": "communication",
                "icon": "⚡",
                "description": "Real-time bidirectional client communication",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
        ]
        for skill in core:
            self._skills[skill["id"]] = skill

    def record_call(self, skill_id: str, duration_ms: float, success: bool = True):
        """Record a skill invocation."""
        skill = self._skills.get(skill_id)
        if not skill:
            return
        skill["calls"] += 1
        if not success:
            skill["errors"] += 1
        now = now_iso()
        skill["last_called"] = now
        # Running average
        old_avg = skill["avg_ms"]
        n = skill["calls"]
        skill["avg_ms"] = round(old_avg + (duration_ms - old_avg) / n, 1)

        self._execution_log.append(
            {
                "skill_id": skill_id,
                "timestamp": now,
                "duration_ms": round(duration_ms, 1),
                "success": success,
            }
        )

    def get_all(self) -> list[dict]:
        return list(self._skills.values())

    def get_by_category(self) -> dict[str, list[dict]]:
        cats: dict[str, list[dict]] = {}
        for s in self._skills.values():
            cats.setdefault(s["category"], []).append(s)
        return cats

    def get_execution_log(self, limit: int = 50) -> list[dict]:
        return list(self._execution_log)[-limit:]

    def total_calls(self) -> int:
        return sum(s["calls"] for s in self._skills.values())

    def total_errors(self) -> int:
        return sum(s["errors"] for s in self._skills.values())
