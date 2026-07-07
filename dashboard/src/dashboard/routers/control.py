"""Control routes: sensory gateway, video training pipeline, MuJoCo body."""

from fastapi import APIRouter

from dashboard import commands
from dashboard.context import DashboardContext
from dashboard.util import now_iso


def build_control_router(ctx: DashboardContext) -> APIRouter:
    router = APIRouter()
    state = ctx.state
    nats = ctx.nats

    # ── Sensory gateway ───────────────────────────────────────────────────
    @router.get("/api/gateway")
    async def get_gateway():
        return {"gateway": state.gateway_status, "timestamp": now_iso()}

    @router.post("/api/gateway/command")
    async def gateway_command(cmd: dict):
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        err = await nats.try_publish(commands.GATEWAY_COMMAND, cmd)
        return err or {"status": "sent", "command": cmd}

    # ── MuJoCo body visualization ─────────────────────────────────────────
    @router.get("/api/mujoco/model")
    async def get_mujoco_model():
        """Static geometry definitions for the MuJoCo humanoid."""
        return {
            "geoms": [
                {
                    "body": "world",
                    "type": "plane",
                    "size": [10, 10, 0.1],
                    "rgba": [0.8, 0.8, 0.8, 1],
                },
                {
                    "body": "torso",
                    "type": "capsule",
                    "size": [0.1, 0.2, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
                {"body": "head", "type": "sphere", "size": [0.1, 0, 0], "rgba": [0.9, 0.8, 0.7, 1]},
                {
                    "body": "r_upper_arm",
                    "type": "capsule",
                    "size": [0.04, 0.15, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
                {
                    "body": "r_forearm",
                    "type": "capsule",
                    "size": [0.03, 0.12, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
                {
                    "body": "l_upper_arm",
                    "type": "capsule",
                    "size": [0.04, 0.15, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
                {
                    "body": "l_forearm",
                    "type": "capsule",
                    "size": [0.03, 0.12, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
                {
                    "body": "r_thigh",
                    "type": "capsule",
                    "size": [0.05, 0.2, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
                {
                    "body": "r_shin",
                    "type": "capsule",
                    "size": [0.04, 0.18, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
                {
                    "body": "l_thigh",
                    "type": "capsule",
                    "size": [0.05, 0.2, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
                {
                    "body": "l_shin",
                    "type": "capsule",
                    "size": [0.04, 0.18, 0],
                    "rgba": [0.3, 0.6, 0.9, 1],
                },
            ]
        }

    @router.post("/api/mujoco/guide")
    async def mujoco_guide(body: dict):
        """Send a guidance command to the MuJoCo body via NATS.

        Body formats:
            {"action": "pose", "joints": {"r_hip": 30, "l_hip": -10}}
            {"action": "push", "body": "torso", "force": [0, 5, 0]}
            {"action": "reward", "channel": "locomotion", "success": true}
            {"action": "reset"}
        """
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        action = body.get("action", "")
        if action not in ("pose", "push", "reward", "reset", "teach"):
            return {"error": f"Unknown action: {action}"}
        await nats.publish(commands.MOTOR_GUIDANCE, body)
        return {"ok": True, "action": action}

    @router.get("/api/mujoco/joints")
    async def mujoco_joints():
        """Joint info (names, ranges, channels) for UI sliders — 29-DOF humanoid."""
        return {
            "joints": [
                # Waist (locomotion)
                {"name": "waist_yaw", "channel": "locomotion", "range_deg": [-75, 75]},
                {"name": "waist_roll", "channel": "locomotion", "range_deg": [-30, 30]},
                {"name": "waist_pitch", "channel": "locomotion", "range_deg": [-30, 45]},
                # Right leg (locomotion)
                {"name": "r_hip_yaw", "channel": "locomotion", "range_deg": [-40, 40]},
                {"name": "r_hip_roll", "channel": "locomotion", "range_deg": [-25, 45]},
                {"name": "r_hip_pitch", "channel": "locomotion", "range_deg": [-30, 120]},
                {"name": "r_knee", "channel": "locomotion", "range_deg": [-145, 0]},
                {"name": "r_ankle_pitch", "channel": "locomotion", "range_deg": [-40, 50]},
                {"name": "r_ankle_roll", "channel": "locomotion", "range_deg": [-25, 25]},
                # Left leg (locomotion)
                {"name": "l_hip_yaw", "channel": "locomotion", "range_deg": [-40, 40]},
                {"name": "l_hip_roll", "channel": "locomotion", "range_deg": [-45, 25]},
                {"name": "l_hip_pitch", "channel": "locomotion", "range_deg": [-30, 120]},
                {"name": "l_knee", "channel": "locomotion", "range_deg": [-145, 0]},
                {"name": "l_ankle_pitch", "channel": "locomotion", "range_deg": [-40, 50]},
                {"name": "l_ankle_roll", "channel": "locomotion", "range_deg": [-25, 25]},
                # Right arm (manipulation)
                {"name": "r_shoulder_pitch", "channel": "manipulation", "range_deg": [-90, 180]},
                {"name": "r_shoulder_roll", "channel": "manipulation", "range_deg": [-30, 150]},
                {"name": "r_shoulder_yaw", "channel": "manipulation", "range_deg": [-90, 70]},
                {"name": "r_elbow", "channel": "manipulation", "range_deg": [0, 145]},
                {"name": "r_wrist_pitch", "channel": "manipulation", "range_deg": [-70, 70]},
                {"name": "r_wrist_yaw", "channel": "manipulation", "range_deg": [-45, 45]},
                # Left arm (manipulation)
                {"name": "l_shoulder_pitch", "channel": "manipulation", "range_deg": [-90, 180]},
                {"name": "l_shoulder_roll", "channel": "manipulation", "range_deg": [-150, 30]},
                {"name": "l_shoulder_yaw", "channel": "manipulation", "range_deg": [-70, 90]},
                {"name": "l_elbow", "channel": "manipulation", "range_deg": [0, 145]},
                {"name": "l_wrist_pitch", "channel": "manipulation", "range_deg": [-70, 70]},
                {"name": "l_wrist_yaw", "channel": "manipulation", "range_deg": [-45, 45]},
                # Neck (head)
                {"name": "neck_pitch", "channel": "head", "range_deg": [-40, 40]},
                {"name": "neck_yaw", "channel": "head", "range_deg": [-55, 55]},
            ]
        }

    # ── Video training ────────────────────────────────────────────────────
    @router.get("/api/video/sessions")
    async def get_video_sessions():
        return {"sessions": list(state.video_sessions.values()), "timestamp": now_iso()}

    @router.post("/api/video/submit")
    async def submit_video(body: dict):
        """Submit a video URL or path for training via the gateway."""
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        url = body.get("url", "").strip()
        if not url:
            return {"error": "url is required"}
        cmd = commands.add_video({**body, "url": url})
        err = await nats.try_publish(commands.GATEWAY_COMMAND, cmd)
        return err or {"status": "submitted", "command": cmd}

    @router.post("/api/video/stop")
    async def stop_video(body: dict):
        """Stop a video training session."""
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        if not body.get("session_id", ""):
            return {"error": "session_id is required"}
        cmd = commands.stop_video(body)
        err = await nats.try_publish(commands.GATEWAY_COMMAND, cmd)
        return err or {"status": "sent", "command": cmd}

    @router.post("/api/video/queue")
    async def queue_video(body: dict):
        """Add a video to the training queue."""
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        url = body.get("url", "").strip()
        if not url:
            return {"error": "url is required"}
        cmd = commands.queue_video({**body, "url": url})
        err = await nats.try_publish(commands.GATEWAY_COMMAND, cmd)
        return err or {"status": "queued", "command": cmd}

    @router.post("/api/video/skip")
    async def skip_video():
        """Skip currently playing video, advance queue."""
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        err = await nats.try_publish(commands.GATEWAY_COMMAND, commands.skip_video())
        return err or {"status": "sent"}

    @router.post("/api/video/clear-queue")
    async def clear_queue():
        """Clear all queued (non-active) videos."""
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        err = await nats.try_publish(commands.GATEWAY_COMMAND, commands.clear_queue())
        return err or {"status": "sent"}

    @router.post("/api/video/remove-queued")
    async def remove_queued(body: dict):
        """Remove a specific video from the queue."""
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        if not body.get("session_id", ""):
            return {"error": "session_id is required"}
        err = await nats.try_publish(commands.GATEWAY_COMMAND, commands.remove_queued(body))
        return err or {"status": "sent"}

    @router.post("/api/video/blacklist")
    async def blacklist_video(body: dict):
        """Blacklist a video — stop if active, remove from queue."""
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        if not body.get("session_id", ""):
            return {"error": "session_id is required"}
        cmd = commands.blacklist_video(body)
        err = await nats.try_publish(commands.GATEWAY_COMMAND, cmd)
        return err or {"status": "blacklisted", "command": cmd}

    @router.get("/api/video/blacklist")
    async def get_blacklist():
        """Request current blacklist from gateway."""
        if not nats.can_publish:
            return {"error": "NATS not connected"}
        err = await nats.try_publish(commands.GATEWAY_COMMAND, commands.get_blacklist())
        return err or {"status": "requested"}

    return router
