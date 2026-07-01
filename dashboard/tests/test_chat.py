"""Tests for the ChatEngine context builders + teleoperation orchestration.

Pure: aiohttp is imported lazily inside the network backends, so importing and
exercising the context/interpretation logic needs no web stack. The network
calls themselves are never reached — ``_complete`` is stubbed.
"""

import asyncio

from dashboard.chat import ChatEngine
from dashboard.state import DashboardState


class _FakeNats:
    """Captures the sensory observation a chat turn injects into the brain."""

    def __init__(self):
        self.observations: list = []

    async def publish_text_observation(self, text):
        self.observations.append(text)


def _engine(state=None, nats=None):
    return ChatEngine(state or DashboardState(), nats or _FakeNats())


# ── brain-state interpretation ─────────────────────────────────────────────

def test_interpret_brain_state_empty_returns_blank():
    assert _engine()._interpret_brain_state() == ""


def test_interpret_brain_state_reports_phase_and_activity():
    state = DashboardState()
    state.neuro_metrics = {
        "step_count": 1000,
        "phase": "juvenile",
        "firing_rates": {"association_cortex": 0.06, "motor_cortex": 0.06},
        "drives": {"energy": 0.9, "fatigue": 0.8, "prediction_error": 0.9},
    }
    text = _engine(state)._interpret_brain_state()
    assert "Phase: juvenile" in text
    assert "association cortex highly active" in text
    assert "motor cortex active" in text
    assert "alert and energetic" in text
    assert "fatigued" in text
    assert "Surprise level: high" in text


def test_interpret_brain_state_infers_phase_from_step_count():
    state = DashboardState()
    state.neuro_metrics = {"step_count": 10}  # < 60k -> infant
    assert "infant" in _engine(state)._interpret_brain_state()


# ── system context ─────────────────────────────────────────────────────────

def test_build_system_context_includes_brain_and_server_sections():
    state = DashboardState()
    state.neuro_metrics = {"step_count": 5, "phase": "infant", "firing_rates": {}, "drives": {}}
    state.system_info = {"os": {"system": "Linux", "release": "6.1"},
                         "cpu": {"cores": 8, "architecture": "x86_64"},
                         "memory": {"total_gb": 32}}
    ctx = _engine(state)._build_system_context()
    assert "You are the voice of Engram" in ctx
    assert "CURRENT BRAIN STATE" in ctx
    assert "Server: Linux 6.1" in ctx and "8 cores" in ctx


def test_build_system_context_without_state_has_no_brain_or_server():
    ctx = _engine()._build_system_context()
    assert "CURRENT BRAIN STATE" not in ctx
    assert "Server:" not in ctx


# ── brain-only fallback ────────────────────────────────────────────────────

def test_generate_brain_only_response_initializing_when_no_metrics():
    out = _engine()._generate_brain_only_response()
    assert out["model"] == "neural-only"
    assert "initializing" in out["content"]


def test_generate_brain_only_response_lists_active_regions():
    state = DashboardState()
    state.neuro_metrics = {"step_count": 7, "phase": "toddler",
                           "firing_rates": {"motor_cortex": 0.2, "sensory_cortex": 0.05}}
    out = _engine(state)._generate_brain_only_response()
    assert "Active regions" in out["content"]
    assert out["model"] == "neural-only"


# ── teleoperation turn orchestration ───────────────────────────────────────

def test_converse_runs_full_turn():
    state = DashboardState()
    nats = _FakeNats()
    engine = ChatEngine(state, nats)

    async def fake_complete(message):
        return {"content": "pong", "model": "test-model"}

    engine._complete = fake_complete  # avoid any network backend

    reply = asyncio.run(engine.converse("teach me about balls"))

    assert reply == {"content": "pong", "model": "test-model"}
    # Text was injected into the brain as sensory input.
    assert nats.observations == ["teach me about balls"]
    # Skill timing recorded as a successful brain.chat call.
    chat_skill = next(s for s in state.skills.get_all() if s["id"] == "brain.chat")
    assert chat_skill["calls"] == 1 and chat_skill["errors"] == 0
    # Flywheel learned a teleoperation interaction.
    assert state.knowledge.get_flywheel_stats()["sources"]["teleoperation"] == 1
    # History holds the user + assistant turns.
    assert [m["role"] for m in state.chat_history] == ["user", "assistant"]
    assert state.chat_history[1]["content"] == "pong"


def test_converse_marks_error_model_as_failed_skill_call():
    state = DashboardState()
    engine = ChatEngine(state, _FakeNats())

    async def fake_complete(message):
        return {"content": "boom", "model": "error"}

    engine._complete = fake_complete
    asyncio.run(engine.converse("hi"))

    chat_skill = next(s for s in state.skills.get_all() if s["id"] == "brain.chat")
    assert chat_skill["calls"] == 1 and chat_skill["errors"] == 1
