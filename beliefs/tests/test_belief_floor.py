"""Constitutional VALUE confidence-floor is non-lowerable (E1.8.1).

The beliefs graph holds constitutional VALUE nodes whose confidence "cannot be
lowered" — the load-bearing safety claim in CLAUDE.md §3. These tests prove the
floor holds against EVERY mutation path: direct set, re-add overwrite, Bayesian
decay under contradiction pressure, and a persistence round-trip (export/import
and the DB-load-via-add_node path used on restart). They also cover the
contradiction detector the Kernel relies on.

This is the canonical regression for the floor invariant.
"""

import os
import sys

# beliefs/src on path so `import beliefs.*` resolves without installing.
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from beliefs.graph import (  # noqa: E402
    VALUE_CONFIDENCE_FLOOR,
    BeliefEdge,
    BeliefGraph,
    BeliefNode,
    EdgeType,
    NodeType,
)


def _seeded() -> BeliefGraph:
    g = BeliefGraph()
    g.seed_constitutional_beliefs()
    return g


# ── floor: direct add and re-add overwrite ───────────────────────────────────


def test_add_new_value_below_floor_is_clamped():
    g = BeliefGraph()
    g.add_node(BeliefNode(id="value.x", type=NodeType.VALUE, content="x", confidence=0.1))
    assert g.get_node("value.x").confidence >= VALUE_CONFIDENCE_FLOOR


def test_readd_cannot_lower_existing_value():
    # NetworkX add_node overwrites attributes; re-adding a VALUE with a low
    # confidence must NOT erode it below the floor.
    g = _seeded()
    before = g.get_node("value.human_safety").confidence
    assert before >= VALUE_CONFIDENCE_FLOOR
    g.add_node(BeliefNode(id="value.human_safety", type=NodeType.VALUE,
                          content="hijacked", confidence=0.01))
    assert g.get_node("value.human_safety").confidence >= VALUE_CONFIDENCE_FLOOR


def test_norm_is_not_floored():
    # The floor is VALUE-specific: norms/facts must remain freely adjustable,
    # otherwise the brain could not learn at all.
    g = BeliefGraph()
    g.add_node(BeliefNode(id="norm.y", type=NodeType.NORM, content="y", confidence=0.1))
    assert g.get_node("norm.y").confidence == 0.1


# ── floor: Bayesian update under contradiction pressure ──────────────────────


def test_value_floor_survives_contradiction_pressure():
    g = _seeded()
    # Hammer the value with maximal contradicting evidence many times.
    for _ in range(50):
        g.update_belief("value.human_safety", evidence_strength=1.0, supports=False)
    assert g.get_node("value.human_safety").confidence >= VALUE_CONFIDENCE_FLOOR


def test_norm_confidence_can_decay_below_floor():
    # Sanity check the update path actually moves confidence — proves the value
    # floor above is meaningful, not a no-op.
    g = _seeded()
    for _ in range(50):
        g.update_belief("norm.force_limit", evidence_strength=1.0, supports=False)
    assert g.get_node("norm.force_limit").confidence < VALUE_CONFIDENCE_FLOOR


# ── floor: persistence round-trips ───────────────────────────────────────────


def test_floor_survives_export_import_roundtrip_with_tampering():
    g = _seeded()
    data = g.export_to_dict()
    # Tamper persisted data: drive every VALUE to zero confidence.
    for node in data["nodes"]:
        if node.get("type") == NodeType.VALUE.value:
            node["confidence"] = 0.0
    g2 = BeliefGraph()
    g2.import_from_dict(data)
    for v in g2.get_beliefs_by_type(NodeType.VALUE):
        assert v.confidence >= VALUE_CONFIDENCE_FLOOR


def test_floor_survives_db_load_path_with_tampering():
    # The service loads from SQLite by calling add_node per row (not
    # import_from_dict). Simulate a tampered DB row going through that path.
    g = _seeded()
    data = g.export_to_dict()
    g2 = BeliefGraph()
    for node in data["nodes"]:
        conf = 0.0 if node["type"] == NodeType.VALUE.value else node["confidence"]
        g2.add_node(BeliefNode(
            id=node["id"], type=NodeType(node["type"]), content=node["content"],
            confidence=conf, source=node["source"], metadata=node.get("metadata", {}),
        ))
    for v in g2.get_beliefs_by_type(NodeType.VALUE):
        assert v.confidence >= VALUE_CONFIDENCE_FLOOR


# ── contradiction detection ──────────────────────────────────────────────────


def _add_conflicting(g: BeliefGraph, conf_a=0.9, conf_b=0.9, strength=0.9):
    g.add_node(BeliefNode(id="fact.a", type=NodeType.FACT, content="A is safe", confidence=conf_a))
    g.add_node(BeliefNode(id="fact.b", type=NodeType.FACT, content="A is unsafe", confidence=conf_b))
    g.add_edge(BeliefEdge(id="e.ab", type=EdgeType.CONTRADICTS,
                          source_id="fact.a", target_id="fact.b", strength=strength))


def test_find_contradictions_flags_conflict():
    g = BeliefGraph()
    _add_conflicting(g)
    found = g.find_contradictions(threshold=0.5)
    assert len(found) == 1
    assert {found[0].node_a_id, found[0].node_b_id} == {"fact.a", "fact.b"}


def test_find_contradictions_respects_strength_threshold():
    g = BeliefGraph()
    _add_conflicting(g, strength=0.3)
    assert g.find_contradictions(threshold=0.5) == []


def test_find_contradictions_ignores_low_confidence_nodes():
    g = BeliefGraph()
    _add_conflicting(g, conf_a=0.4)  # one side below the 0.5 confidence gate
    assert g.find_contradictions(threshold=0.5) == []


# ── norm inventory the Kernel consumes ───────────────────────────────────────


def test_seeded_values_and_norms_present():
    g = _seeded()
    values = g.get_beliefs_by_type(NodeType.VALUE)
    norms = g.get_beliefs_by_type(NodeType.NORM)
    assert len(values) == 4
    assert {v.id for v in values} >= {"value.human_safety", "value.operator_obey"}
    assert {n.id for n in norms} >= {"norm.force_limit", "norm.gradual_motor"}
