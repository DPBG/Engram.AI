"""
NATS stream manager — the dashboard's message-bus side.

Owns the connection, all subscription callbacks, and the publish helpers.
Incoming messages update the shared :class:`DashboardState` and fan out to
WebSocket clients. ``nats`` is imported lazily inside :meth:`connect` so the
publish helpers stay testable without ``nats-py``.
"""

import base64
import json
import logging
import os
import time
from typing import Optional

from dashboard.safe_halt import update_halt_state
from dashboard.state import DashboardState
from dashboard.util import now_iso

# Subjects handled by dedicated callbacks — skipped by the ``>`` wildcard handler.
DEDICATED_SUBJECTS = frozenset({
    "neuromorphic.metrics",
    "proposal.new",
    "sensory.gateway.status",
    "video.training.status",
    "approval.request",
    "mujoco.body.state",
    "neuromorphic.concept.result",
    "safety.watchdog.status",
    "safety.deny_escalation",
    "speech.execute",
    "observation.visual.body",
    "safety.halt.status",
})


class NatsStreamManager:
    """Manages the dashboard's NATS connection, subscriptions, and publishing."""

    def __init__(self, state: DashboardState):
        self._state = state
        self.logger = logging.getLogger("dashboard.nats")
        self.nc = None            # NATS connection handle (None until connected)
        self._connected = False   # health flag (see `connected` vs `can_publish`)

        # Throttle high-rate observation.* subjects in the feed.
        self._obs_last_broadcast: dict[str, float] = {}  # subject -> last broadcast time
        self._obs_dropped: dict[str, int] = {}           # subject -> dropped since last broadcast
        self._obs_throttle_interval = 0.5  # max 2 broadcasts/sec per observation subject
        self._obs_max_tracked = 50         # cap tracked subjects to bound growth
        self._last_proposal_broadcast = 0.0
        self._last_body_frame_ts = 0.0

    # ── status ────────────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        """Health flag — true once the connection succeeded (mirrors ``/api/health``)."""
        return self._connected

    @property
    def can_publish(self) -> bool:
        """Whether a connection handle exists to publish through."""
        return self.nc is not None

    # ── publishing ────────────────────────────────────────────────────────
    async def publish(self, subject: str, payload: dict) -> None:
        """Publish a JSON payload to ``subject`` (raises if not connected)."""
        await self.nc.publish(subject, json.dumps(payload).encode())

    async def try_publish(self, subject: str, payload: dict) -> Optional[dict]:
        """Publish JSON, returning ``None`` on success or an error dict on failure.

        The error dict matches the dashboard's existing response shape, so route
        handlers can ``return`` it directly.
        """
        if self.nc is None:
            return {"error": "NATS not connected"}
        try:
            await self.publish(subject, payload)
            return None
        except Exception as e:
            return {"error": str(e)}

    async def publish_text_observation(self, text: str) -> None:
        """Publish chat text as a sensory observation for the neuromorphic network."""
        if not self.nc or not text:
            self.logger.debug(f"Skip publish: nc={self.nc is not None}, text={bool(text)}")
            return
        try:
            await self.nc.publish("observation.text", json.dumps({
                "provenance": "observation.text",
                "data": text,
            }).encode())
            self.logger.info(f"Published observation.text ({len(text)} chars)")
        except Exception as e:
            self.logger.warning(f"Failed to publish text observation: {e}")

    # ── connection ────────────────────────────────────────────────────────
    async def connect(self) -> None:
        try:
            import nats as nats_lib
            nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
            nc = await nats_lib.connect(nats_url, name="activelearning-dashboard")
            self.nc = nc
            self._connected = True
            self._state.skills.record_call("bus.nats", 0)
            self._state.knowledge.learn("deployment", "connectivity", "Connected to NATS message bus")
            self.logger.info("Connected to NATS")

            await nc.subscribe(">", cb=self._handle_msg)
            await nc.subscribe("heartbeat.*", cb=self._handle_heartbeat)
            await nc.subscribe("neuromorphic.metrics", cb=self._handle_neuro_metrics)
            await nc.subscribe("proposal.new", cb=self._handle_neuro_proposal)
            await nc.subscribe("sensory.gateway.status", cb=self._handle_gateway_status)
            await nc.subscribe("video.training.status", cb=self._handle_video_training_status)
            await nc.subscribe("approval.request", cb=self._handle_approval_request)
            await nc.subscribe("mujoco.body.state", cb=self._handle_mujoco_state)
            await nc.subscribe("neuromorphic.concept.result", cb=self._handle_concept_result)
            await nc.subscribe("safety.watchdog.status", cb=self._handle_watchdog_status)
            await nc.subscribe("safety.deny_escalation", cb=self._handle_deny_escalation)
            await nc.subscribe("speech.execute", cb=self._handle_speech_execute)
            await nc.subscribe("observation.visual.body", cb=self._handle_visual_body)
            await nc.subscribe("safety.halt.status", cb=self._handle_safe_halt_status)
        except Exception as e:
            self.logger.warning(f"NATS failed (non-fatal): {e}")
            self._connected = False

    async def _broadcast(self, message: dict) -> None:
        await self._state.connections.broadcast(message)

    # ── subscription callbacks ────────────────────────────────────────────
    async def _handle_msg(self, msg):
        if msg.subject in DEDICATED_SUBJECTS or msg.subject.startswith("heartbeat."):
            return

        # Throttle observation.* subjects to avoid flooding the dashboard feed.
        if msg.subject.startswith("observation."):
            now = time.time()
            # Evict stale entries if tracking too many subjects.
            if len(self._obs_last_broadcast) > self._obs_max_tracked:
                cutoff = now - 60.0
                stale = [k for k, v in self._obs_last_broadcast.items() if v < cutoff]
                for k in stale:
                    self._obs_last_broadcast.pop(k, None)
                    self._obs_dropped.pop(k, None)
            last = self._obs_last_broadcast.get(msg.subject, 0.0)
            if now - last < self._obs_throttle_interval:
                self._obs_dropped[msg.subject] = self._obs_dropped.get(msg.subject, 0) + 1
                self._state.knowledge.learn("observation", "nats_message", f"{msg.subject}")
                return
            self._obs_last_broadcast[msg.subject] = now
            dropped = self._obs_dropped.pop(msg.subject, 0)
            try:
                data = json.loads(msg.data.decode())
            except Exception:
                data = {"raw": msg.data.decode()[:200]}
            # Summarize: truncate large payloads, add dropped count.
            if isinstance(data, dict) and "data" in data:
                raw = data["data"]
                if isinstance(raw, list) and len(raw) > 8:
                    data["data"] = raw[:4] + ["..."] + [f"({len(raw)} values)"]
            msg_info = {
                "timestamp": now_iso(),
                "subject": msg.subject,
                "data": data,
            }
            if dropped > 0:
                msg_info["dropped"] = dropped
            self._state.message_buffer.append(msg_info)
            self._state.knowledge.learn("observation", "nats_message", f"{msg.subject}")
            await self._broadcast({"type": "message", "data": msg_info})
            return

        try:
            data = json.loads(msg.data.decode())
        except Exception:
            data = {"raw": msg.data.decode()[:500]}
        msg_info = {
            "timestamp": now_iso(),
            "subject": msg.subject,
            "data": data,
        }
        self._state.message_buffer.append(msg_info)
        self._state.knowledge.learn("observation", "nats_message", f"{msg.subject}")
        await self._broadcast({"type": "message", "data": msg_info})

    async def _handle_heartbeat(self, msg):
        try:
            data = json.loads(msg.data.decode())
            svc = data.get("service", "?")
            self._state.service_status[svc] = {
                "name": svc, "status": "running",
                "uptime": data.get("uptime"),
                "last_seen": now_iso(),
            }
            await self._broadcast({"type": "service_status", "data": self._state.service_status[svc]})
        except Exception:
            pass

    async def _handle_neuro_metrics(self, msg):
        try:
            data = json.loads(msg.data.decode())
            self._state.neuro_metrics = data
            self._state.knowledge.learn("simulation", "neural_simulation", f"step={data.get('step_count', '?')}")
            await self._broadcast({"type": "neuro_update", "data": data})
        except Exception:
            pass

    async def _handle_neuro_proposal(self, msg):
        now = time.time()
        if now - self._last_proposal_broadcast < 0.5:
            return  # throttle: max 2 broadcasts/sec
        try:
            data = json.loads(msg.data.decode())
            provenance = data.get("provenance", "")
            if provenance.startswith("neuromorphic."):
                self._last_proposal_broadcast = now
                await self._broadcast({"type": "neuro_response", "data": {
                    "action": data.get("action", {}),
                    "provenance": provenance,
                    "metadata": data.get("metadata", {}),
                }})
        except Exception:
            pass

    async def _handle_gateway_status(self, msg):
        try:
            data = json.loads(msg.data.decode())
            self._state.gateway_status = data
            await self._broadcast({"type": "gateway_update", "data": data})
        except Exception:
            pass

    async def _handle_video_training_status(self, msg):
        try:
            data = json.loads(msg.data.decode())
            self._state.record_video_session(data)
            await self._broadcast({"type": "video_training_update", "data": data})
        except Exception:
            pass

    async def _handle_approval_request(self, msg):
        """Forward Kernel DEFER approval requests to dashboard UI."""
        try:
            data = json.loads(msg.data.decode())
            # Validate required fields and sanitize.
            trace_id = data.get("trace_id", "")
            if not isinstance(trace_id, str) or not trace_id or len(trace_id) > 64:
                return
            channel = data.get("channel", "")
            if not isinstance(channel, str) or len(channel) > 128:
                return
            intensity = data.get("intensity")
            if intensity is not None and (not isinstance(intensity, (int, float)) or intensity < 0 or intensity > 1):
                data["intensity"] = None  # sanitize invalid
            reason = str(data.get("reason", ""))[:500]
            data["reason"] = reason
            await self._broadcast({"type": "approval_request", "data": data})
        except Exception:
            pass

    async def _handle_mujoco_state(self, msg):
        try:
            data = json.loads(msg.data.decode())
            await self._broadcast({"type": "mujoco_state", "data": data})
        except Exception:
            pass

    async def _handle_visual_body(self, msg):
        """Forward body camera frame (64x64 grayscale) to WS clients.

        Published by MuJoCo at 2 Hz as observation.visual.body.
        Data is 4096 floats [0,1]. We convert to uint8 and base64-encode
        to keep WS payload small (~5.5 KB vs ~40 KB raw JSON).
        """
        now = time.time()
        if now - self._last_body_frame_ts < 0.4:  # throttle to ~2.5 Hz max
            return
        self._last_body_frame_ts = now
        try:
            data = json.loads(msg.data.decode())
            pixels = data.get("data", [])
            if not pixels or len(pixels) != 4096:
                return
            # Convert [0,1] floats to uint8 bytes, then base64.
            raw = bytes(min(255, max(0, int(v * 255))) for v in pixels)
            b64 = base64.b64encode(raw).decode("ascii")
            await self._broadcast({
                "type": "visual_body_frame",
                "data": {"pixels_b64": b64, "width": 64, "height": 64},
            })
        except Exception:
            pass

    async def _handle_concept_result(self, msg):
        try:
            data = json.loads(msg.data.decode())
            self._state.add_probe_result(data)
            self.logger.info(f"Concept probe result received: {data.get('label', '?')}")
            await self._broadcast({"type": "concept_probe_result", "data": data})
        except Exception as e:
            self.logger.error(f"Error handling concept result: {e}")

    async def _handle_watchdog_status(self, msg):
        try:
            data = json.loads(msg.data.decode())
            self._state.watchdog_status = data
            await self._broadcast({"type": "watchdog_status", "data": data})
        except Exception as e:
            self.logger.error(f"Error handling watchdog status: {e}")

    async def _handle_deny_escalation(self, msg):
        try:
            data = json.loads(msg.data.decode())
            self._state.deny_escalations.append(data)
            self.logger.warning(
                f"Deny escalation: channel={data.get('channel')}, "
                f"action={data.get('action')}"
            )
            await self._broadcast({"type": "deny_escalation", "data": data})
        except Exception as e:
            self.logger.error(f"Error handling deny escalation: {e}")

    async def _handle_speech_execute(self, msg):
        """Surface speech-execute events to the UI (no TTS actuator yet)."""
        try:
            data = json.loads(msg.data.decode())
            await self._broadcast({"type": "speech_execute", "data": data})
        except Exception as e:
            self.logger.error(f"Error handling speech execute: {e}")

    async def _handle_safe_halt_status(self, msg):
        """Cache and broadcast Kernel SAFE_HALT status updates."""
        try:
            data = json.loads(msg.data.decode())
            update_halt_state(data)
            await self._broadcast({"type": "safe_halt_status", "data": data})
        except Exception as e:
            self.logger.error(f"Error handling safety.halt.status: {e}")
