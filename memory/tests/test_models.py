"""Tests for memory data models."""

import uuid

from activelearning import current_timestamp

from memory import models
from memory.models import Episode


def test_models_reuse_sdk_helpers():
    # The local duplicate helpers are gone; defaults come from the SDK.
    assert not hasattr(models, "generate_id")
    assert models.current_timestamp is current_timestamp


def test_episode_defaults():
    episode = Episode(trace_id="t-1", summary="hello")
    assert uuid.UUID(episode.id).version == 4  # fresh UUID4 id
    assert isinstance(episode.timestamp, int)  # Unix epoch millis
    assert episode.timestamp > 1_700_000_000_000


def test_episode_ids_are_unique():
    # default_factory must mint a fresh id per instance.
    ids = {Episode(trace_id="t", summary="s").id for _ in range(100)}
    assert len(ids) == 100
