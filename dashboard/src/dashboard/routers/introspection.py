"""Introspection routes: skills, knowledge / data flywheel, messages, insights."""

from fastapi import APIRouter

from dashboard.context import DashboardContext


def build_introspection_router(ctx: DashboardContext) -> APIRouter:
    router = APIRouter()
    state = ctx.state

    # ── Skills ────────────────────────────────────────────────────────────
    @router.get("/api/skills")
    async def get_skills():
        return {
            "skills": state.skills.get_all(),
            "by_category": state.skills.get_by_category(),
            "total_calls": state.skills.total_calls(),
            "total_errors": state.skills.total_errors(),
        }

    @router.get("/api/skills/log")
    async def get_skill_log(limit: int = 50):
        return {"log": state.skills.get_execution_log(limit)}

    # ── Knowledge / Flywheel ──────────────────────────────────────────────
    @router.get("/api/knowledge")
    async def get_knowledge(limit: int = 50, source: str = None):
        return {
            "entries": state.knowledge.get_entries(limit, source),
            "flywheel": state.knowledge.get_flywheel_stats(),
        }

    @router.get("/api/flywheel")
    async def get_flywheel():
        return state.knowledge.get_flywheel_stats()

    # ── NATS messages ─────────────────────────────────────────────────────
    @router.get("/api/messages")
    async def get_messages(limit: int = 100):
        return {"messages": list(state.message_buffer)[-limit:], "total": len(state.message_buffer)}

    # ── Insights ──────────────────────────────────────────────────────────
    @router.get("/api/insights")
    async def get_insights(limit: int = 20):
        return {"insights": list(state.insights_log)[-limit:]}

    return router
