"""Tests for the SkillRegistry (pure stdlib — no web stack)."""

from dashboard.skills import SkillRegistry


def test_core_skills_registered():
    reg = SkillRegistry()
    ids = {s["id"] for s in reg.get_all()}
    assert {"env.detect", "brain.chat", "bus.nats", "bus.websocket"} <= ids


def test_record_call_updates_counts_and_running_average():
    reg = SkillRegistry()
    reg.record_call("brain.chat", 100.0)
    reg.record_call("brain.chat", 200.0)
    skill = next(s for s in reg.get_all() if s["id"] == "brain.chat")
    assert skill["calls"] == 2
    assert skill["errors"] == 0
    assert skill["avg_ms"] == 150.0  # running mean of 100 and 200
    assert skill["last_called"] is not None


def test_record_call_tracks_errors():
    reg = SkillRegistry()
    reg.record_call("brain.chat", 10.0, success=False)
    skill = next(s for s in reg.get_all() if s["id"] == "brain.chat")
    assert skill["errors"] == 1
    assert reg.total_errors() == 1


def test_record_call_unknown_skill_is_noop():
    reg = SkillRegistry()
    before = reg.total_calls()
    reg.record_call("does.not.exist", 5.0)
    assert reg.total_calls() == before


def test_totals_and_categories():
    reg = SkillRegistry()
    reg.record_call("brain.chat", 1.0)
    reg.record_call("env.detect", 1.0)
    assert reg.total_calls() == 2
    cats = reg.get_by_category()
    assert "cognition" in cats and "perception" in cats


def test_execution_log_limit():
    reg = SkillRegistry()
    for _ in range(10):
        reg.record_call("brain.chat", 1.0)
    assert len(reg.get_execution_log(limit=3)) == 3
    assert len(reg.get_execution_log(limit=50)) == 10
