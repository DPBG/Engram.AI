"""Tests for the KnowledgeBase data flywheel (pure stdlib — no web stack)."""

from dashboard.knowledge import KnowledgeBase


def test_learn_increments_source_and_total_counts():
    kb = KnowledgeBase()
    kb.learn("teleoperation", "conversation", "hello")
    kb.learn("teleoperation", "conversation", "again")
    kb.learn("observation", "nats_message", "subject.x")

    stats = kb.get_flywheel_stats()
    assert stats["sources"]["teleoperation"] == 2
    assert stats["sources"]["observation"] == 1
    assert stats["total_interactions"] == 3
    assert stats["total_knowledge_entries"] == 3


def test_unknown_source_counts_interaction_but_not_bucket():
    kb = KnowledgeBase()
    kb.learn("mystery", "category", "content")
    stats = kb.get_flywheel_stats()
    # Interaction + entry still recorded, but no matching source bucket.
    assert stats["total_interactions"] == 1
    assert "mystery" not in stats["sources"]


def test_get_entries_filters_by_source_and_limit():
    kb = KnowledgeBase()
    for i in range(5):
        kb.learn("observation", "nats_message", f"o{i}")
    kb.learn("deployment", "system", "deployed")

    obs = kb.get_entries(source="observation")
    assert len(obs) == 5
    assert all(e["source"] == "observation" for e in obs)

    assert len(kb.get_entries(limit=2)) == 2


def test_recent_entries_are_capped_at_ten():
    kb = KnowledgeBase()
    for i in range(20):
        kb.learn("observation", "nats_message", f"o{i}")
    assert len(kb.get_flywheel_stats()["recent_entries"]) == 10
