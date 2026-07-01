"""Data models for Memory Service."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def generate_id() -> str:
    return str(uuid.uuid4())


def current_timestamp() -> int:
    return int(time.time() * 1000)


@dataclass
class Episode:
    """An episodic memory entry."""

    trace_id: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    utility_score: float = 1.0
    id: str = field(default_factory=generate_id)
    timestamp: int = field(default_factory=current_timestamp)


@dataclass
class MemoryQuery:
    """A memory query request."""

    query: str
    limit: int = 10
    min_score: float = 0.5


@dataclass
class MemoryResult:
    """A memory search result."""

    episode_id: str
    score: float
    trace_id: str
    timestamp: int
    tags: list[str]
    summary: str
