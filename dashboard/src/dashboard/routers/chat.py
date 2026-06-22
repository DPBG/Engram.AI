"""Chat (teleoperation) routes: chat, history, observation injection, concept probe."""

import logging

from fastapi import APIRouter

from dashboard.context import DashboardContext
from dashboard.models import ChatMessage, ObservationPayload
from dashboard.util import now_iso

logger = logging.getLogger("dashboard.chat")


def build_chat_router(ctx: DashboardContext) -> APIRouter:
    router = APIRouter()
    state = ctx.state

    # ── Chat (teleoperation channel) ──────────────────────────────────────
    @router.post("/api/chat")
    async def chat(msg: ChatMessage):
        reply = await ctx.chat.converse(msg.message)
        return {
            "reply": reply["content"],
            "timestamp": now_iso(),
            "model": reply.get("model", ctx.chat.llm_model),
        }

    @router.get("/api/chat/history")
    async def get_chat_history(limit: int = 50):
        return {"history": state.chat_history[-limit:]}

    # ── Observation injection (for demos) ─────────────────────────────────
    @router.post("/api/observation")
    async def inject_observation(obs: ObservationPayload):
        """Inject a sensory observation into the brain via NATS.

        Used by the demo reaction probe to send stimuli and measure real brain responses.
        """
        if not ctx.nats.can_publish:
            return {"error": "NATS not connected", "ok": False}
        try:
            await ctx.nats.publish(obs.provenance, {
                "provenance": obs.provenance,
                "data": obs.data,
            })
            logger.info(f"Injected observation via {obs.provenance}")
            return {"ok": True, "provenance": obs.provenance}
        except Exception as e:
            logger.warning(f"Failed to inject observation: {e}")
            return {"error": str(e), "ok": False}

    # ── Concept probe ─────────────────────────────────────────────────────
    @router.post("/api/concept-probe")
    async def concept_probe(body: dict):
        """Inject a stimulus and probe concept layer response."""
        if not ctx.nats.can_publish:
            return {"error": "NATS not connected", "ok": False}
        err = await ctx.nats.try_publish("neuromorphic.concept.probe", body)
        if err:
            return {**err, "ok": False}
        return {"ok": True, "label": body.get("label", "probe")}

    @router.get("/api/concept-probe/results")
    async def concept_probe_results():
        """Return all stored concept probe results."""
        return {"results": state.concept_probe_results}

    @router.delete("/api/concept-probe/results")
    async def clear_concept_probe_results():
        """Clear stored probe results."""
        state.concept_probe_results.clear()
        return {"ok": True}

    return router
