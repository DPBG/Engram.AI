"""Tests for NatsStreamManager publish helpers + subscription callbacks.

``nats`` is imported lazily inside ``connect()``, so the publish helpers and
handlers can be exercised with fakes and no ``nats-py`` install.
"""

import asyncio
import json

from dashboard.nats_stream import DEDICATED_SUBJECTS, NatsStreamManager
from dashboard.state import DashboardState


class _FakeNc:
    """Minimal NATS connection double recording published frames."""

    def __init__(self, fail: bool = False):
        self.published: list = []
        self.fail = fail

    async def publish(self, subject, data):
        if self.fail:
            raise RuntimeError("publish failed")
        self.published.append((subject, data))


class _FakeMsg:
    def __init__(self, subject, payload):
        self.subject = subject
        self.data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()


class _RecordingWS:
    def __init__(self):
        self.sent: list = []

    async def send_json(self, message):
        self.sent.append(message)


def _manager():
    state = DashboardState()
    ws = _RecordingWS()
    state.connections.add(ws)
    return NatsStreamManager(state), state, ws


# ── status flags ───────────────────────────────────────────────────────────


def test_status_defaults_disconnected():
    mgr, _, _ = _manager()
    assert mgr.connected is False
    assert mgr.can_publish is False


# ── publishing ─────────────────────────────────────────────────────────────


def test_try_publish_not_connected():
    mgr, _, _ = _manager()
    err = asyncio.run(mgr.try_publish("subj", {"a": 1}))
    assert err == {"error": "NATS not connected"}


def test_try_publish_success_returns_none_and_sends_json():
    mgr, _, _ = _manager()
    mgr.nc = _FakeNc()
    err = asyncio.run(mgr.try_publish("subj", {"a": 1}))
    assert err is None
    subject, data = mgr.nc.published[0]
    assert subject == "subj"
    assert json.loads(data.decode()) == {"a": 1}


def test_try_publish_wraps_exceptions():
    mgr, _, _ = _manager()
    mgr.nc = _FakeNc(fail=True)
    err = asyncio.run(mgr.try_publish("subj", {"a": 1}))
    assert "error" in err and err["error"] != "NATS not connected"


def test_publish_text_observation_noop_without_connection_or_text():
    mgr, _, _ = _manager()
    asyncio.run(mgr.publish_text_observation("hi"))  # nc is None -> no-op, no raise
    mgr.nc = _FakeNc()
    asyncio.run(mgr.publish_text_observation(""))  # empty text -> no-op
    assert mgr.nc.published == []


def test_publish_text_observation_sends_provenance_frame():
    mgr, _, _ = _manager()
    mgr.nc = _FakeNc()
    asyncio.run(mgr.publish_text_observation("a ball is round"))
    subject, data = mgr.nc.published[0]
    assert subject == "observation.text"
    assert json.loads(data.decode()) == {
        "provenance": "observation.text",
        "data": "a ball is round",
    }


# ── subscription callbacks ─────────────────────────────────────────────────


def test_handle_neuro_metrics_updates_state_and_broadcasts():
    mgr, state, ws = _manager()
    asyncio.run(mgr._handle_neuro_metrics(_FakeMsg("neuromorphic.metrics", {"step_count": 42})))
    assert state.neuro_metrics == {"step_count": 42}
    assert ws.sent[-1] == {"type": "neuro_update", "data": {"step_count": 42}}
    # Logged as a simulation learning event for the flywheel.
    assert state.knowledge.get_flywheel_stats()["sources"]["simulation"] == 1


def test_handle_heartbeat_records_service_status():
    mgr, state, ws = _manager()
    asyncio.run(
        mgr._handle_heartbeat(_FakeMsg("heartbeat.kernel", {"service": "kernel", "uptime": 5}))
    )
    assert state.service_status["kernel"]["status"] == "running"
    assert ws.sent[-1]["type"] == "service_status"


def test_handle_video_training_status_stores_real_session():
    mgr, state, ws = _manager()
    asyncio.run(
        mgr._handle_video_training_status(
            _FakeMsg("video.training.status", {"session_id": "vid1", "status": "running"})
        )
    )
    assert "vid1" in state.video_sessions
    assert ws.sent[-1]["type"] == "video_training_update"


def test_handle_safe_halt_status_caches_and_broadcasts():
    from dashboard import safe_halt

    mgr, state, ws = _manager()
    asyncio.run(
        mgr._handle_safe_halt_status(
            _FakeMsg("safety.halt.status", {"halted": True, "reason": "kernel"})
        )
    )
    assert safe_halt.get_halt_state() == {"halted": True, "reason": "kernel"}
    assert ws.sent[-1] == {"type": "safe_halt_status", "data": {"halted": True, "reason": "kernel"}}
    assert "safety.halt.status" in DEDICATED_SUBJECTS


def test_dedicated_subjects_skipped_by_wildcard_handler():
    mgr, state, ws = _manager()
    # A subject owned by a dedicated callback must be ignored by the > handler.
    asyncio.run(mgr._handle_msg(_FakeMsg("neuromorphic.metrics", {"x": 1})))
    assert len(state.message_buffer) == 0
    assert ws.sent == []
    assert "neuromorphic.metrics" in DEDICATED_SUBJECTS


def test_wildcard_handler_buffers_and_broadcasts_generic_message():
    mgr, state, ws = _manager()
    asyncio.run(mgr._handle_msg(_FakeMsg("some.event", {"hello": "world"})))
    assert len(state.message_buffer) == 1
    assert state.message_buffer[0]["subject"] == "some.event"
    assert ws.sent[-1]["type"] == "message"


def test_observation_throttle_drops_rapid_duplicates():
    mgr, state, ws = _manager()
    mgr._obs_throttle_interval = 10_000  # force the 2nd message to be throttled

    asyncio.run(mgr._handle_msg(_FakeMsg("observation.text", {"data": "first"})))
    assert len(state.message_buffer) == 1  # first one broadcast + buffered
    assert ws.sent[-1]["type"] == "message"

    asyncio.run(mgr._handle_msg(_FakeMsg("observation.text", {"data": "second"})))
    # Throttled: not buffered, not broadcast, but still counted as a drop + learned.
    assert len(state.message_buffer) == 1
    assert mgr._obs_dropped["observation.text"] == 1
    # observation learnings: once per message (broadcast + throttled both learn).
    assert state.knowledge.get_flywheel_stats()["sources"]["observation"] == 2


def test_observation_dropped_count_surfaces_on_next_broadcast():
    mgr, state, ws = _manager()
    mgr._obs_throttle_interval = 10_000
    asyncio.run(mgr._handle_msg(_FakeMsg("observation.text", {"data": "a"})))
    asyncio.run(mgr._handle_msg(_FakeMsg("observation.text", {"data": "b"})))  # dropped
    # Now allow the next one through and assert the dropped count is reported.
    mgr._obs_throttle_interval = 0
    asyncio.run(mgr._handle_msg(_FakeMsg("observation.text", {"data": "c"})))
    assert state.message_buffer[-1].get("dropped") == 1


def test_large_observation_payload_is_summarized():
    mgr, state, ws = _manager()
    mgr._obs_throttle_interval = 0
    big = list(range(20))
    asyncio.run(mgr._handle_msg(_FakeMsg("observation.image", {"data": big})))
    stored = state.message_buffer[-1]["data"]["data"]
    # First 4 kept, then an ellipsis + a count marker (not the full 20 values).
    assert stored[:4] == [0, 1, 2, 3]
    assert "..." in stored
    assert any("20 values" in str(x) for x in stored)


def test_handle_bus_metrics_stores_service_and_broadcasts():
    mgr, state, ws = _manager()
    payload = {
        "service": "kernel",
        "publish": {"count": 3, "avg_ms": 1.0, "max_ms": 2.0, "jetstream_count": 1},
        "subscribe": {"count": 10, "avg_ms": 0.5, "max_ms": 1.0},
        "request": {"count": 2, "avg_ms": 5.0, "max_ms": 8.0},
    }
    asyncio.run(mgr._handle_bus_metrics(_FakeMsg("eventbus.metrics.kernel", payload)))
    assert state.bus_metrics_by_service["kernel"] == payload
    assert ws.sent[-1]["type"] == "bus_metrics_update"
    services = {row["service"] for row in ws.sent[-1]["data"]}
    assert "kernel" in services
    assert "dashboard" in services


def test_eventbus_metrics_subject_skipped_by_wildcard_handler():
    mgr, state, ws = _manager()
    asyncio.run(mgr._handle_msg(_FakeMsg("eventbus.metrics.kernel", {"service": "kernel"})))
    assert len(state.message_buffer) == 0
    assert ws.sent == []


def test_handle_dlq_message_stores_and_broadcasts():
    mgr, state, ws = _manager()
    payload = {
        "original_subject": "proposal.new",
        "reason": "max_deliver_exhausted(5): always fails",
        "num_delivered": 5,
        "payload": "{}",
    }
    asyncio.run(mgr._handle_dlq_message(_FakeMsg("dlq.proposal.new", payload)))

    summary = state.dlq_summary()
    assert summary["total"] == 1
    assert summary["by_subject"] == {"proposal.new": 1}
    assert ws.sent[-1]["type"] == "dlq_update"
    assert ws.sent[-1]["data"]["total"] == 1


def test_dlq_subject_skipped_by_wildcard_handler():
    mgr, state, ws = _manager()
    asyncio.run(mgr._handle_msg(_FakeMsg("dlq.proposal.new", {"original_subject": "proposal.new"})))
    assert len(state.message_buffer) == 0
    assert ws.sent == []
