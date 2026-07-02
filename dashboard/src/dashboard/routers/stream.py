"""Real-time brain stream: the ``/ws`` WebSocket endpoint."""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from dashboard import commands
from dashboard.auth import authorize_websocket
from dashboard.context import DashboardContext
from dashboard.safe_halt import (
    get_halt_state,
    sanitize_halt_payload,
    sanitize_resume_payload,
)
from dashboard.system import get_live_metrics
from dashboard.util import now_iso

logger = logging.getLogger("dashboard.stream")


def build_stream_router(ctx: DashboardContext) -> APIRouter:
    router = APIRouter()
    state = ctx.state
    nats = ctx.nats

    async def forward_gateway(cmd: dict) -> None:
        """Forward a gateway command to NATS when connected."""
        if nats.can_publish:
            await nats.publish(commands.GATEWAY_COMMAND, cmd)

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        # The /ws channel carries state-changing commands (approval responses
        # to the Kernel, motor guidance, training control). When auth is
        # enabled, refuse connections that don't present the token — otherwise a
        # forged approval_response could defeat the Kernel gate.
        if not authorize_websocket(websocket):
            await websocket.close(code=1008)  # policy violation
            logger.warning("WS rejected: missing/invalid dashboard token")
            return
        await websocket.accept()
        state.connections.add(websocket)
        state.skills.record_call("bus.websocket", 0)
        logger.info(f"WS connected ({len(state.connections)})")

        try:
            await websocket.send_json(
                {
                    "type": "init",
                    "data": {
                        "system": state.system_info,
                        "skills": state.skills.get_all(),
                        "skills_by_category": state.skills.get_by_category(),
                        "flywheel": state.knowledge.get_flywheel_stats(),
                        "services": list(state.service_status.values()),
                        "messages": list(state.message_buffer)[-30:],
                        "insights": list(state.insights_log)[-10:],
                        "live_metrics": get_live_metrics(),
                        "neuromorphic": state.neuro_metrics,
                        "gateway": state.gateway_status,
                        "video_sessions": list(state.video_sessions.values()),
                        "halt_state": get_halt_state(),
                    },
                }
            )

            while True:
                data = await websocket.receive_text()
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                msg_type = payload.get("type")

                if msg_type == "chat":
                    reply = await ctx.chat.converse(payload.get("message", ""))
                    await websocket.send_json(
                        {
                            "type": "chat_response",
                            "data": {
                                "reply": reply["content"],
                                "model": reply.get("model", ctx.chat.llm_model),
                                "timestamp": now_iso(),
                            },
                        }
                    )
                elif msg_type == "gateway_command":
                    await forward_gateway(payload.get("command", {}))
                elif msg_type == "video_submit":
                    await forward_gateway(commands.add_video(payload))
                elif msg_type == "video_stop":
                    await forward_gateway(commands.stop_video(payload))
                elif msg_type == "video_queue":
                    await forward_gateway(commands.queue_video(payload))
                elif msg_type == "video_skip":
                    await forward_gateway(commands.skip_video())
                elif msg_type == "video_clear_queue":
                    await forward_gateway(commands.clear_queue())
                elif msg_type == "video_remove_queued":
                    await forward_gateway(commands.remove_queued(payload))
                elif msg_type == "video_blacklist":
                    await forward_gateway(commands.blacklist_video(payload))
                elif msg_type == "approval_response":
                    await _handle_approval_response(websocket, nats, payload)
                elif msg_type == "safe_halt":
                    await _handle_safe_halt(websocket, nats, payload)
                elif msg_type == "safe_resume":
                    await _handle_safe_resume(websocket, nats, payload)
                elif msg_type == "mujoco_guide":
                    # Forward guidance command to NATS.
                    guide_data = payload.get("data", {})
                    if nats.can_publish and guide_data.get("action"):
                        try:
                            await nats.publish(commands.MOTOR_GUIDANCE, guide_data)
                        except Exception as e:
                            logger.error(f"Guidance publish failed: {e}")
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WS error: {e}")
        finally:
            state.connections.discard(websocket)

    return router


async def _handle_approval_response(websocket, nats, payload) -> None:
    """Relay a human's answer to a Kernel DEFER back onto the bus."""
    resp = payload.get("data", {})
    trace_id = str(resp.get("trace_id", ""))[:64]
    if not trace_id or not nats.can_publish:
        return  # silently ignore invalid or no-NATS
    try:
        # Sanitize trace_id for NATS subject safety.
        safe_trace = trace_id.replace(" ", "").replace(">", "").replace("*", "")
        await nats.publish(f"approval.response.{safe_trace}", resp)
    except Exception as e:
        logger.error(f"Failed to publish approval response: {e}")
        await websocket.send_json(
            {"type": "error", "data": {"message": "Approval delivery failed"}}
        )


async def _handle_safe_halt(websocket, nats, payload) -> None:
    """Publish an operator SAFE_HALT to the Kernel (fails closed on bad payload)."""
    ok, reason, operator_id = sanitize_halt_payload(payload.get("data", {}))
    if not ok:
        await websocket.send_json(
            {"type": "error", "data": {"message": "Invalid SAFE_HALT payload"}}
        )
    elif not nats.can_publish:
        await websocket.send_json(
            {"type": "error", "data": {"message": "NATS not connected — SAFE_HALT not delivered"}}
        )
    else:
        try:
            await nats.publish("safety.halt", {"reason": reason, "operator_id": operator_id})
            logger.critical("SAFE_HALT published by %s: %s", operator_id, reason)
        except Exception as e:
            logger.error(f"Failed to publish safety.halt: {e}")
            await websocket.send_json(
                {"type": "error", "data": {"message": "SAFE_HALT delivery failed"}}
            )


async def _handle_safe_resume(websocket, nats, payload) -> None:
    """Release a SAFE_HALT on the Kernel (operator SAFE_RESUME)."""
    ok, operator_id = sanitize_resume_payload(payload.get("data", {}))
    if not ok:
        await websocket.send_json(
            {"type": "error", "data": {"message": "Invalid SAFE_RESUME payload"}}
        )
    elif not nats.can_publish:
        await websocket.send_json(
            {"type": "error", "data": {"message": "NATS not connected — SAFE_RESUME not delivered"}}
        )
    else:
        try:
            await nats.publish("safety.resume", {"operator_id": operator_id})
            logger.warning("SAFE_HALT released by %s", operator_id)
        except Exception as e:
            logger.error(f"Failed to publish safety.resume: {e}")
            await websocket.send_json(
                {"type": "error", "data": {"message": "SAFE_RESUME delivery failed"}}
            )
