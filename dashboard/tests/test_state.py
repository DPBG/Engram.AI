"""Tests for DashboardState + ConnectionManager (pure stdlib — no web stack)."""

import asyncio

from dashboard.state import (
    MAX_CHAT_HISTORY,
    MAX_PROBE_RESULTS,
    MAX_VIDEO_SESSIONS,
    ConnectionManager,
    DashboardState,
)


class _FakeWS:
    """Records broadcast payloads; optionally raises to simulate a dead client."""

    def __init__(self, dead: bool = False):
        self.sent: list = []
        self.dead = dead

    async def send_json(self, message):
        if self.dead:
            raise RuntimeError("connection closed")
        self.sent.append(message)


# ── ConnectionManager ──────────────────────────────────────────────────────

def test_broadcast_reaches_all_live_clients():
    cm = ConnectionManager()
    a, b = _FakeWS(), _FakeWS()
    cm.add(a)
    cm.add(b)
    assert len(cm) == 2 and bool(cm) is True

    asyncio.run(cm.broadcast({"type": "x"}))
    assert a.sent == [{"type": "x"}]
    assert b.sent == [{"type": "x"}]


def test_broadcast_prunes_dead_clients():
    cm = ConnectionManager()
    live, dead = _FakeWS(), _FakeWS(dead=True)
    cm.add(live)
    cm.add(dead)

    asyncio.run(cm.broadcast({"type": "ping"}))
    # The dead client is dropped; the live one stays and received the message.
    assert len(cm) == 1
    assert live.sent == [{"type": "ping"}]


def test_broadcast_noop_when_empty():
    cm = ConnectionManager()
    assert bool(cm) is False
    asyncio.run(cm.broadcast({"type": "x"}))  # must not raise


# ── chat history ───────────────────────────────────────────────────────────

def test_append_chat_trims_to_max():
    state = DashboardState()
    for i in range(MAX_CHAT_HISTORY + 25):
        state.append_chat("user", f"m{i}")
    assert len(state.chat_history) == MAX_CHAT_HISTORY
    # Oldest dropped, newest kept.
    assert state.chat_history[-1]["content"] == f"m{MAX_CHAT_HISTORY + 24}"
    assert all("timestamp" in e for e in state.chat_history)


# ── concept-probe results ──────────────────────────────────────────────────

def test_add_probe_result_caps():
    state = DashboardState()
    for i in range(MAX_PROBE_RESULTS + 10):
        state.add_probe_result({"i": i})
    assert len(state.concept_probe_results) == MAX_PROBE_RESULTS
    assert state.concept_probe_results[-1] == {"i": MAX_PROBE_RESULTS + 9}


# ── video sessions ─────────────────────────────────────────────────────────

def test_record_video_session_filters_transient_markers():
    state = DashboardState()
    for sid in ("error", "download_error", "pending", ""):
        state.record_video_session({"session_id": sid})
    state.record_video_session({})  # missing id
    assert state.video_sessions == {}

    state.record_video_session({"session_id": "abc", "status": "running"})
    assert "abc" in state.video_sessions


def test_record_video_session_prunes_oldest_finished_when_full():
    state = DashboardState()
    # Fill to capacity with finished sessions (eligible for pruning).
    for i in range(MAX_VIDEO_SESSIONS):
        state.record_video_session(
            {"session_id": f"done{i}", "status": "completed", "created_at": i}
        )
    assert len(state.video_sessions) == MAX_VIDEO_SESSIONS

    # One more pushes over the limit -> the oldest finished one is evicted.
    state.record_video_session({"session_id": "new", "status": "running", "created_at": 999})
    assert len(state.video_sessions) == MAX_VIDEO_SESSIONS
    assert "new" in state.video_sessions
    assert "done0" not in state.video_sessions  # oldest finished pruned
