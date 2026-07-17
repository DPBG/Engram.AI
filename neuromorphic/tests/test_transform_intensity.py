"""Tests for Kernel TRANSFORM intensity / proposal shape helpers."""

from __future__ import annotations

from neuromorphic.transform_intensity import (
    intensity_from_transformations,
    proposal_from_transformation,
)


def test_intensity_from_kernel_bare_action_shape():
    """Kernel body-profile clamps emit top-level intensity on the action dict."""
    corrected = intensity_from_transformations(
        0.95,
        [{"channel": "manipulation", "intensity": 0.4, "type": "motor_command"}],
    )
    assert corrected == 0.4


def test_intensity_from_nested_action_proposal_shape():
    """SDK ActionProposal nesting still works."""
    corrected = intensity_from_transformations(
        0.95,
        [{"trace_id": "t1", "action": {"channel": "manipulation", "intensity": 0.5}}],
    )
    assert corrected == 0.5


def test_intensity_unchanged_when_transformations_empty():
    assert intensity_from_transformations(0.8, None) == 0.8
    assert intensity_from_transformations(0.8, []) == 0.8


def test_intensity_unchanged_when_only_nested_without_intensity():
    # Old broken consumer path: looking only at action would also miss bare intensity.
    # Nested without intensity must not invent a value.
    assert intensity_from_transformations(0.9, [{"action": {"channel": "head"}}]) == 0.9


def test_proposal_from_bare_action_wraps_into_action_key():
    original = {
        "trace_id": "t1",
        "provenance": "planner",
        "action": {"channel": "manipulation", "intensity": 0.95},
    }
    bare = {"channel": "manipulation", "intensity": 0.4}
    wrapped = proposal_from_transformation(original, bare)
    assert wrapped["trace_id"] == "t1"
    assert wrapped["action"]["intensity"] == 0.4
    assert wrapped["action"]["channel"] == "manipulation"


def test_proposal_from_nested_action_proposal_passthrough():
    original = {"trace_id": "t1", "action": {"intensity": 0.9}}
    nested = {"trace_id": "t1", "action": {"intensity": 0.3}}
    assert proposal_from_transformation(original, nested) is nested
