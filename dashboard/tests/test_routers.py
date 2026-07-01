"""Smoke tests for the FastAPI routers.

These need the web stack, so they self-skip where FastAPI/its TestClient (and
its httpx backend) are unavailable — e.g. the minimal governance CI env. Each
router is mounted on an isolated app wired to a fake context, so the tests
exercise routing + handler logic with no NATS, no LLM, and no lifespan.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from fastapi import FastAPI  # noqa: E402, I001
from fastapi.testclient import TestClient  # noqa: E402

from dashboard.context import DashboardContext  # noqa: E402
from dashboard.routers.chat import build_chat_router  # noqa: E402
from dashboard.routers.control import build_control_router  # noqa: E402
from dashboard.routers.introspection import build_introspection_router  # noqa: E402
from dashboard.routers.stream import build_stream_router  # noqa: E402
from dashboard.routers.system import build_system_router  # noqa: E402
from dashboard.state import DashboardState  # noqa: E402


class _FakeNats:
    def __init__(self, connected=False, fail=False):
        self._connected = connected
        self._fail = fail
        self.published: list = []

    @property
    def connected(self):
        return self._connected

    @property
    def can_publish(self):
        return self._connected

    async def publish(self, subject, payload):
        if self._fail:
            raise RuntimeError("boom")
        self.published.append((subject, payload))

    async def try_publish(self, subject, payload):
        if not self._connected:
            return {"error": "NATS not connected"}
        if self._fail:
            return {"error": "boom"}
        self.published.append((subject, payload))
        return None


class _FakeChat:
    llm_model = "fake-model"

    def __init__(self):
        self.seen: list = []

    async def converse(self, message):
        self.seen.append(message)
        return {"content": f"echo:{message}", "model": "fake-model"}


class _FakeMetrics:
    async def fetch_docker(self):
        return [{"service": "x", "cpu_percent": 1}]


def _ctx(nats_connected=False):
    nats = _FakeNats(connected=nats_connected)
    chat = _FakeChat()
    metrics = _FakeMetrics()
    ctx = DashboardContext(state=DashboardState(), nats=nats, chat=chat, metrics=metrics)
    return ctx, nats, chat, metrics


def _client(router):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ── system router ──────────────────────────────────────────────────────────

def test_system_router_health_and_endpoints():
    ctx, _, _, _ = _ctx()
    client = _client(build_system_router(ctx, static_dir="/nonexistent"))

    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy" and body["nats"] is False

    assert client.get("/api/system").status_code == 200
    assert client.get("/api/services").json() == {"services": []}
    assert client.get("/api/metrics").json()["metrics"] == [{"service": "x", "cpu_percent": 1}]
    # No static index present -> graceful fallback.
    assert client.get("/dashboard").json() == {"status": "ok"}


# ── introspection router ───────────────────────────────────────────────────

def test_introspection_router():
    ctx, _, _, _ = _ctx()
    ctx.state.knowledge.learn("observation", "nats_message", "subj")
    client = _client(build_introspection_router(ctx))

    skills = client.get("/api/skills").json()
    assert "skills" in skills and skills["total_calls"] == 0
    assert client.get("/api/flywheel").json()["total_interactions"] == 1
    assert client.get("/api/messages").json() == {"messages": [], "total": 0}
    assert client.get("/api/insights").json() == {"insights": []}


# ── chat router ────────────────────────────────────────────────────────────

def test_chat_router_post_delegates_to_engine():
    ctx, _, chat, _ = _ctx()
    client = _client(build_chat_router(ctx))

    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "echo:hi" and body["model"] == "fake-model"
    assert chat.seen == ["hi"]


def test_chat_router_concept_probe_lifecycle():
    ctx, nats, _, _ = _ctx(nats_connected=True)
    client = _client(build_chat_router(ctx))

    r = client.post("/api/concept-probe", json={"label": "red"})
    assert r.json() == {"ok": True, "label": "red"}
    assert nats.published[0][0] == "neuromorphic.concept.probe"

    assert client.get("/api/concept-probe/results").json() == {"results": []}
    assert client.delete("/api/concept-probe/results").json() == {"ok": True}


def test_chat_router_observation_requires_nats():
    ctx, _, _, _ = _ctx(nats_connected=False)
    client = _client(build_chat_router(ctx))
    r = client.post("/api/observation", json={"provenance": "observation.text", "data": "x"})
    assert r.json() == {"error": "NATS not connected", "ok": False}


# ── control router ─────────────────────────────────────────────────────────

def test_control_router_video_submit_publishes_command():
    ctx, nats, _, _ = _ctx(nats_connected=True)
    client = _client(build_control_router(ctx))

    r = client.post("/api/video/submit", json={"url": "http://v"})
    assert r.json()["status"] == "submitted"
    subject, cmd = nats.published[0]
    assert subject == "sensory.gateway.command"
    assert cmd["action"] == "add_video" and cmd["url"] == "http://v"


def test_control_router_validates_before_publish():
    ctx, _, _, _ = _ctx(nats_connected=True)
    client = _client(build_control_router(ctx))
    # Empty url is rejected with a validation error (not a NATS error).
    assert client.post("/api/video/submit", json={"url": "   "}).json() == {"error": "url is required"}


def test_control_router_gateway_command_requires_nats():
    ctx, _, _, _ = _ctx(nats_connected=False)
    client = _client(build_control_router(ctx))
    assert client.post("/api/gateway/command", json={"action": "x"}).json() == {
        "error": "NATS not connected"
    }


def test_control_router_static_geometry_endpoints():
    ctx, _, _, _ = _ctx()
    client = _client(build_control_router(ctx))
    assert len(client.get("/api/mujoco/model").json()["geoms"]) > 0
    assert len(client.get("/api/mujoco/joints").json()["joints"]) == 29


# ── stream router ──────────────────────────────────────────────────────────

def test_stream_router_websocket_init_and_ping():
    ctx, nats, _, _ = _ctx(nats_connected=True)
    client = _client(build_stream_router(ctx))

    with client.websocket_connect("/ws") as ws:
        init = ws.receive_json()
        assert init["type"] == "init"
        assert "skills" in init["data"]

        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}

        ws.send_json({"type": "gateway_command", "command": {"action": "go"}})
        ws.send_json({"type": "ping"})  # round-trip to flush the publish above
        assert ws.receive_json() == {"type": "pong"}

    assert ("sensory.gateway.command", {"action": "go"}) in nats.published


def test_stream_router_safe_halt_and_resume_publish_to_kernel():
    from dashboard import safe_halt
    safe_halt.update_halt_state({"halted": False})
    ctx, nats, _, _ = _ctx(nats_connected=True)
    client = _client(build_stream_router(ctx))

    with client.websocket_connect("/ws") as ws:
        assert "halt_state" in ws.receive_json()["data"]  # init snapshot
        ws.send_json({"type": "safe_halt", "data": {"reason": "r", "operator_id": "op"}})
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}
        ws.send_json({"type": "safe_resume", "data": {"operator_id": "op"}})
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}

    assert ("safety.halt", {"reason": "r", "operator_id": "op"}) in nats.published
    assert ("safety.resume", {"operator_id": "op"}) in nats.published


def test_stream_router_safe_halt_invalid_payload_fails_closed():
    ctx, nats, _, _ = _ctx(nats_connected=True)
    client = _client(build_stream_router(ctx))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # init
        ws.send_json({"type": "safe_halt", "data": "not-a-dict"})
        assert ws.receive_json() == {
            "type": "error", "data": {"message": "Invalid SAFE_HALT payload"},
        }
    assert nats.published == []


def test_stream_router_chat_over_websocket():
    ctx, _, chat, _ = _ctx()
    client = _client(build_stream_router(ctx))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # init
        ws.send_json({"type": "chat", "message": "yo"})
        resp = ws.receive_json()
        assert resp["type"] == "chat_response"
        assert resp["data"]["reply"] == "echo:yo"
    assert chat.seen == ["yo"]


# ── auth integration against the real app ──────────────────────────────────

def test_real_app_auth_gate_blocks_unauthenticated_mutations(monkeypatch):
    monkeypatch.setenv("ENGRAM_DASHBOARD_TOKEN", "secret-token")
    from dashboard.api import app  # imported lazily so other tests stay web-stack-light

    client = TestClient(app)
    # Mutating request without the token is rejected by the middleware.
    r = client.delete("/api/concept-probe/results")
    assert r.status_code == 401

    # With the token it passes the gate.
    r = client.delete("/api/concept-probe/results", headers={"Authorization": "Bearer secret-token"})
    assert r.status_code == 200

    # Read-only GETs stay open.
    assert client.get("/api/health").status_code == 200
