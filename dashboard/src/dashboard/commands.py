"""
Outbound control commands the dashboard publishes to the bus.

Subject constants + builders for the sensory-gateway and motor-guidance
commands, shared by the HTTP control routes and the WebSocket channel so each
command shape is defined exactly once. Builders read from a mapping (a request
body or a WS payload) and never mutate it.
"""

GATEWAY_COMMAND = "sensory.gateway.command"
MOTOR_GUIDANCE = "motor.guidance"


def add_video(src: dict) -> dict:
    return {
        "action": "add_video",
        "url": src.get("url", ""),
        "fps": src.get("fps", 2.0),
        "loop": src.get("loop", True),
        "transcript": src.get("transcript", False),
    }


def queue_video(src: dict) -> dict:
    return {
        "action": "queue_video",
        "url": src.get("url", ""),
        "fps": src.get("fps", 2.0),
        "transcript": src.get("transcript", False),
        "target_loops": src.get("target_loops", 5),
        "category": src.get("category", ""),
    }


def stop_video(src: dict) -> dict:
    return {"action": "stop_video", "session_id": src.get("session_id", "")}


def remove_queued(src: dict) -> dict:
    return {"action": "remove_queued", "session_id": src.get("session_id", "")}


def blacklist_video(src: dict) -> dict:
    return {
        "action": "blacklist_video",
        "session_id": src.get("session_id", ""),
        "reason": src.get("reason", "Blacklisted by user"),
    }


def skip_video() -> dict:
    return {"action": "skip_video"}


def clear_queue() -> dict:
    return {"action": "clear_queue"}


def get_blacklist() -> dict:
    return {"action": "get_blacklist"}
