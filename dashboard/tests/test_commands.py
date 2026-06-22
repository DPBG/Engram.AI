"""Tests for the shared gateway command builders (pure stdlib — no web stack)."""

from dashboard import commands


def test_add_video_defaults_and_overrides():
    assert commands.add_video({}) == {
        "action": "add_video", "url": "", "fps": 2.0, "loop": True, "transcript": False,
    }
    cmd = commands.add_video({"url": "u", "fps": 5, "loop": False, "transcript": True})
    assert cmd == {
        "action": "add_video", "url": "u", "fps": 5, "loop": False, "transcript": True,
    }


def test_queue_video_shape():
    assert commands.queue_video({"url": "u", "target_loops": 3, "category": "c"}) == {
        "action": "queue_video", "url": "u", "fps": 2.0, "transcript": False,
        "target_loops": 3, "category": "c",
    }


def test_session_scoped_commands():
    assert commands.stop_video({"session_id": "s"}) == {"action": "stop_video", "session_id": "s"}
    assert commands.remove_queued({"session_id": "s"}) == {
        "action": "remove_queued", "session_id": "s",
    }
    assert commands.blacklist_video({"session_id": "s"}) == {
        "action": "blacklist_video", "session_id": "s", "reason": "Blacklisted by user",
    }
    assert commands.blacklist_video({"session_id": "s", "reason": "spam"})["reason"] == "spam"


def test_parameterless_commands():
    assert commands.skip_video() == {"action": "skip_video"}
    assert commands.clear_queue() == {"action": "clear_queue"}
    assert commands.get_blacklist() == {"action": "get_blacklist"}


def test_builders_do_not_mutate_source():
    src = {"url": "u"}
    commands.add_video(src)
    assert src == {"url": "u"}  # builder read only, no in-place keys added
