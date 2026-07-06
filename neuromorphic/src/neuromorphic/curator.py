"""Curator -- accept / reject incoming cross-brain reports on a main brain.

Phase 1 of docs/CROSS-BRAIN-TRANSFER-DESIGN.md. Observability-only:
subscribes to the three report subjects from any specialist (via
wildcard prefix), applies conservative thresholds, and logs accept /
reject decisions. No replay yet -- the accepted reports flow back into
the brain's learning pipeline in Phase 2.

Iron rule (CLAUDE.md): brain code never interprets actuator tag strings.
The curator does not look at tags either; tag-aware routing belongs to
the motor-pattern stream in Phase 3, and even there it is pure string
matching.

Defaults are CONSERVATIVE on purpose. Phase 1 will produce too many
rejects, not too few. Loosen via env once we see real specialist data.

Env knobs (all optional; defaults below):
  NEURO_CURATOR_ENABLED        -- "1" to run, "0" off (default 0).
                                  Set to 1 on main brains only.
  NEURO_CURATOR_MIN_CONFIDENCE -- accept floor (default 0.7).
  NEURO_CURATOR_NOVELTY_MAX_SIM -- reject if cosine sim vs recent > this
                                   (default 0.85). Concept events only.
  NEURO_CURATOR_RECENT_WINDOW  -- how many accepted concept vectors to
                                  remember for the novelty check
                                  (default 100).
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_STREAMS = ("observation", "concept_event", "cognitive_event")


@dataclass
class CuratorConfig:
    enabled: bool = False
    min_confidence: float = 0.7
    novelty_max_cosine_sim: float = 0.85
    recent_window: int = 100

    @classmethod
    def from_env(cls) -> "CuratorConfig":
        def _f(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        def _i(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=os.environ.get("NEURO_CURATOR_ENABLED", "0") != "0",
            min_confidence=_f("NEURO_CURATOR_MIN_CONFIDENCE", 0.7),
            novelty_max_cosine_sim=_f("NEURO_CURATOR_NOVELTY_MAX_SIM", 0.85),
            recent_window=_i("NEURO_CURATOR_RECENT_WINDOW", 100),
        )


@dataclass
class _CuratorStats:
    accepts: dict[str, int] = field(default_factory=lambda: {s: 0 for s in _STREAMS})
    rejects: dict[str, int] = field(default_factory=lambda: {s: 0 for s in _STREAMS})


class Curator:
    """Decides accept or reject for incoming cross-brain reports.

    Use:
        c = Curator(CuratorConfig.from_env())
        accept, reason = c.decide("concept_event", payload)
        if accept:
            # Phase 2+: replay into brain's learning pipeline.
            ...

    All decisions are logged for observability. Stats are queryable via
    :meth:`stats` for dashboard / metrics surfacing.
    """

    def __init__(self, config: CuratorConfig | None = None) -> None:
        self._config = config or CuratorConfig.from_env()
        self._recent_concepts: deque[np.ndarray] = deque(
            maxlen=self._config.recent_window
        )
        self._stats = _CuratorStats()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def config(self) -> CuratorConfig:
        return self._config

    def decide(self, stream: str, payload: dict[str, Any]) -> tuple[bool, str]:
        """Return ``(accept, reason)`` for an incoming report payload.

        ``stream`` must be one of "observation", "concept_event",
        "cognitive_event". Unknown streams are always rejected.
        """
        if stream not in _STREAMS:
            return False, f"unknown stream {stream!r}"
        if not self._config.enabled:
            return False, "curator disabled"

        # Confidence floor applies to every stream.
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            self._stats.rejects[stream] += 1
            return False, "confidence not a float"

        # NaN comparisons always return False, so a NaN confidence would
        # silently pass the floor check. Reject NaN and inf explicitly.
        if not np.isfinite(confidence):
            self._stats.rejects[stream] += 1
            return False, f"confidence {confidence!r} not finite"

        if confidence < self._config.min_confidence:
            self._stats.rejects[stream] += 1
            return False, (
                f"confidence {confidence:.3f} < min "
                f"{self._config.min_confidence:.3f}"
            )

        # Novelty check applies only to concept events (the only stream
        # that ships an activation vector). Observation + cognitive events
        # skip novelty for now; can be added if duplicates become a problem.
        if stream == "concept_event":
            activation = payload.get("activation")
            if activation is not None:
                vec = np.asarray(activation, dtype=np.float32)
                if vec.size == 0:
                    self._stats.rejects[stream] += 1
                    return False, "empty activation vector"
                if not self._is_novel(vec):
                    self._stats.rejects[stream] += 1
                    return False, "near-duplicate of recent concept"
                self._recent_concepts.append(vec)

        self._stats.accepts[stream] += 1
        return True, "accepted"

    def _is_novel(self, vec: np.ndarray) -> bool:
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return False
        unit = vec / norm
        max_sim = 0.0
        for prior in self._recent_concepts:
            pn = float(np.linalg.norm(prior))
            if pn == 0.0:
                continue
            sim = float(np.dot(unit, prior / pn))
            if sim > max_sim:
                max_sim = sim
            if sim > self._config.novelty_max_cosine_sim:
                return False
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "accepts": dict(self._stats.accepts),
            "rejects": dict(self._stats.rejects),
            "recent_concepts": len(self._recent_concepts),
            "enabled": self._config.enabled,
        }


__all__ = ["Curator", "CuratorConfig"]
