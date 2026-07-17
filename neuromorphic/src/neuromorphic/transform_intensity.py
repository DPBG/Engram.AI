"""Helpers for Kernel TRANSFORM payloads.

The Kernel evaluator emits bare action dicts in ``transformations``
(e.g. ``{"channel": "manipulation", "intensity": 0.4}``). The SDK type hint
documents nested ActionProposal shapes (``{"action": {...}}``). Consumers must
accept both so body-profile clamps teach the safe intensity, not the original.
"""

from __future__ import annotations

from typing import Any


def intensity_from_transformations(
    original_intensity: float,
    transformations: list[Any] | None,
) -> float:
    """Return the corrected intensity from a Kernel TRANSFORM payload.

    Prefers top-level ``intensity`` (Kernel wire shape). Falls back to nested
    ``action.intensity`` (ActionProposal shape). Leaves ``original_intensity``
    unchanged when no usable intensity is present.
    """
    corrected = original_intensity
    if not transformations:
        return corrected

    for t in transformations:
        if not isinstance(t, dict):
            continue
        intensity = t.get("intensity")
        if isinstance(intensity, (int, float)):
            corrected = float(intensity)
            continue
        action = t.get("action")
        if isinstance(action, dict):
            nested = action.get("intensity")
            if isinstance(nested, (int, float)):
                corrected = float(nested)
    return corrected


def proposal_from_transformation(
    original_proposal: dict[str, Any],
    transformation: Any,
) -> dict[str, Any]:
    """Normalize a Kernel transformation into an ActionProposal-shaped dict.

    Bare action dicts (Kernel) are wrapped as ``action`` on a copy of the
    original proposal. Nested ActionProposal dicts are returned as-is.
    """
    if not isinstance(transformation, dict):
        return original_proposal
    if "action" in transformation and isinstance(transformation["action"], dict):
        return transformation
    return {**original_proposal, "action": transformation}
