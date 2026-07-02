"""Hardwired reflex arc definitions — innate from birth."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from neuromorphic.regions import ReflexArc

logger = logging.getLogger(__name__)


@dataclass
class ReflexResponse:
    """A triggered reflex response."""

    reflex_name: str
    intensity: float
    action: dict[str, Any]
    priority: int = 100  # reflexes are high-priority


_REFLEX_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "pain_withdrawal",
        "sub_range": "pain_withdrawal",
        "threshold": 0.15,
        "action_type": "withdraw",
        "priority": 100,
    },
    {
        "name": "grip",
        "sub_range": "grip",
        "threshold": 0.2,
        "action_type": "grip",
        "priority": 80,
    },
    {
        "name": "startle",
        "sub_range": "startle",
        "threshold": 0.25,
        "action_type": "startle",
        "priority": 90,
    },
    {
        "name": "flinch",
        "sub_range": "flinch",
        "threshold": 0.2,
        "action_type": "flinch",
        "priority": 95,
    },
]


class ReflexManager:
    """
    Monitors reflex arc activity and triggers innate responses.

    These are hardwired — they don't learn. They exist from "birth"
    and provide baseline survival behavior.
    """

    def __init__(self) -> None:
        self._definitions = _REFLEX_DEFINITIONS

    def check(self, reflex_arc: ReflexArc) -> list[ReflexResponse]:
        """
        Check reflex arc activity and return any triggered reflexes.

        Args:
            reflex_arc: The reflex arc brain region.

        Returns:
            List of triggered ReflexResponse objects.
        """
        responses: list[ReflexResponse] = []

        for defn in self._definitions:
            sr = reflex_arc.get_subrange(defn["sub_range"])
            if sr is None:
                continue

            # Compute firing rate in the sub-range
            rate = float(reflex_arc.spikes[sr.slice()].mean())

            if rate >= defn["threshold"]:
                responses.append(
                    ReflexResponse(
                        reflex_name=defn["name"],
                        intensity=rate,
                        action={
                            "type": defn["action_type"],
                            "source": "reflex_arc",
                            "intensity": rate,
                        },
                        priority=defn["priority"],
                    )
                )
                logger.debug(f"Reflex triggered: {defn['name']} (rate={rate:.3f})")

        return responses
