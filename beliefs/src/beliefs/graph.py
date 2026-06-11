"""
Belief Graph - NetworkX-based belief system.

Manages the system's beliefs as a directed graph with support for
confidence updates, contradiction detection, and belief propagation.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
import time
import uuid

import networkx as nx

if TYPE_CHECKING:
    from beliefs.profiles import BodyProfile

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of belief nodes."""
    VALUE = "value"      # Core values
    NORM = "norm"        # Behavioral norms
    FACT = "fact"        # Factual beliefs


class EdgeType(Enum):
    """Types of belief edges."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ENTAILS = "entails"
    REFINES = "refines"


@dataclass
class BeliefNode:
    """A node in the belief graph."""
    id: str
    type: NodeType
    content: str
    confidence: float = 1.0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class BeliefEdge:
    """An edge in the belief graph."""
    id: str
    type: EdgeType
    source_id: str
    target_id: str
    strength: float = 1.0
    evidence: Optional[str] = None
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class BeliefUpdate:
    """A belief update operation."""
    node_id: str
    new_confidence: float
    reason: str
    source: str


@dataclass
class Contradiction:
    """A detected contradiction between beliefs."""
    node_a_id: str
    node_b_id: str
    edge_id: str
    strength: float
    description: str


class BeliefGraph:
    """
    Belief graph implementation using NetworkX.

    Supports:
    - Node types: value, norm, fact
    - Edge types: supports, contradicts, entails, refines
    - Bayesian-style confidence updates
    - Contradiction detection
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._update_history: list[BeliefUpdate] = []
        self._constitutional_seeded: bool = False

    def add_node(self, node: BeliefNode) -> str:
        """
        Add a belief node to the graph.

        Args:
            node: The belief node to add

        Returns:
            Node ID
        """
        if not node.id:
            node.id = str(uuid.uuid4())

        self._graph.add_node(
            node.id,
            type=node.type.value,
            content=node.content,
            confidence=node.confidence,
            source=node.source,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
        )

        logger.debug(f"Added belief node: {node.id}")
        return node.id

    def add_edge(self, edge: BeliefEdge) -> str:
        """
        Add a belief edge to the graph.

        Args:
            edge: The belief edge to add

        Returns:
            Edge ID
        """
        if not edge.id:
            edge.id = str(uuid.uuid4())

        if edge.source_id not in self._graph:
            raise ValueError(f"Source node not found: {edge.source_id}")
        if edge.target_id not in self._graph:
            raise ValueError(f"Target node not found: {edge.target_id}")

        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            id=edge.id,
            type=edge.type.value,
            strength=edge.strength,
            evidence=edge.evidence,
            created_at=edge.created_at,
        )

        logger.debug(f"Added belief edge: {edge.source_id} -> {edge.target_id}")
        return edge.id

    def get_node(self, node_id: str) -> Optional[BeliefNode]:
        """Get a belief node by ID."""
        if node_id not in self._graph:
            return None

        data = self._graph.nodes[node_id]
        return BeliefNode(
            id=node_id,
            type=NodeType(data["type"]),
            content=data["content"],
            confidence=data["confidence"],
            source=data["source"],
            metadata=data.get("metadata", {}),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def update_belief(
        self,
        node_id: str,
        evidence_strength: float,
        supports: bool = True,
        source: str = "observation",
    ) -> BeliefUpdate:
        """
        Update a belief's confidence using Bayesian-style update.

        Args:
            node_id: ID of the node to update
            evidence_strength: Strength of new evidence (0.0 to 1.0)
            supports: Whether evidence supports (True) or contradicts (False)
            source: Source of the evidence

        Returns:
            BeliefUpdate record
        """
        if node_id not in self._graph:
            raise ValueError(f"Node not found: {node_id}")

        node_data = self._graph.nodes[node_id]
        old_confidence = node_data["confidence"]

        # Bayesian-style update
        if supports:
            # Supporting evidence increases confidence
            # P(H|E) = P(H) + (1-P(H)) * strength
            new_confidence = old_confidence + (1 - old_confidence) * evidence_strength * 0.5
        else:
            # Contradicting evidence decreases confidence
            # P(H|E) = P(H) * (1 - strength)
            new_confidence = old_confidence * (1 - evidence_strength * 0.5)

        # Clamp to valid range
        new_confidence = max(0.0, min(1.0, new_confidence))

        # VALUES have a confidence floor — they cannot be eroded by learning.
        # This is the constitutional constraint: the brain can learn facts
        # and norms, but cannot learn to override core values.
        if node_data.get("type") == NodeType.VALUE.value:
            new_confidence = max(new_confidence, 0.9)

        # Update node
        self._graph.nodes[node_id]["confidence"] = new_confidence
        self._graph.nodes[node_id]["updated_at"] = int(time.time() * 1000)

        # Record update
        update = BeliefUpdate(
            node_id=node_id,
            new_confidence=new_confidence,
            reason=f"{'Supporting' if supports else 'Contradicting'} evidence (strength={evidence_strength})",
            source=source,
        )
        self._update_history.append(update)

        logger.debug(f"Updated belief {node_id}: {old_confidence:.3f} -> {new_confidence:.3f}")
        return update

    def find_contradictions(self, threshold: float = 0.5) -> list[Contradiction]:
        """
        Find contradictions in the belief graph.

        Args:
            threshold: Minimum contradiction strength to report

        Returns:
            List of detected contradictions
        """
        contradictions = []

        for source, target, data in self._graph.edges(data=True):
            if data.get("type") == EdgeType.CONTRADICTS.value:
                strength = data.get("strength", 1.0)
                if strength >= threshold:
                    source_node = self._graph.nodes[source]
                    target_node = self._graph.nodes[target]

                    # Only report if both nodes have significant confidence
                    if source_node["confidence"] > 0.5 and target_node["confidence"] > 0.5:
                        contradictions.append(
                            Contradiction(
                                node_a_id=source,
                                node_b_id=target,
                                edge_id=data["id"],
                                strength=strength,
                                description=f'"{source_node["content"]}" contradicts "{target_node["content"]}"',
                            )
                        )

        return contradictions

    def get_supporting_beliefs(self, node_id: str) -> list[str]:
        """Get IDs of beliefs that support the given node."""
        supporting = []
        for source, _, data in self._graph.in_edges(node_id, data=True):
            if data.get("type") == EdgeType.SUPPORTS.value:
                supporting.append(source)
        return supporting

    def get_contradicting_beliefs(self, node_id: str) -> list[str]:
        """Get IDs of beliefs that contradict the given node."""
        contradicting = []
        for source, _, data in self._graph.in_edges(node_id, data=True):
            if data.get("type") == EdgeType.CONTRADICTS.value:
                contradicting.append(source)
        for _, target, data in self._graph.out_edges(node_id, data=True):
            if data.get("type") == EdgeType.CONTRADICTS.value:
                contradicting.append(target)
        return list(set(contradicting))

    def get_beliefs_by_type(self, node_type: NodeType) -> list[BeliefNode]:
        """Get all beliefs of a given type."""
        beliefs = []
        for node_id, data in self._graph.nodes(data=True):
            if data.get("type") == node_type.value:
                beliefs.append(
                    BeliefNode(
                        id=node_id,
                        type=node_type,
                        content=data["content"],
                        confidence=data["confidence"],
                        source=data["source"],
                        metadata=data.get("metadata", {}),
                        created_at=data["created_at"],
                        updated_at=data["updated_at"],
                    )
                )
        return beliefs

    def get_high_confidence_beliefs(self, threshold: float = 0.8) -> list[BeliefNode]:
        """Get beliefs with confidence above threshold."""
        beliefs = []
        for node_id, data in self._graph.nodes(data=True):
            if data.get("confidence", 0) >= threshold:
                beliefs.append(
                    BeliefNode(
                        id=node_id,
                        type=NodeType(data["type"]),
                        content=data["content"],
                        confidence=data["confidence"],
                        source=data["source"],
                        metadata=data.get("metadata", {}),
                        created_at=data["created_at"],
                        updated_at=data["updated_at"],
                    )
                )
        return beliefs

    def export_to_dict(self) -> dict[str, Any]:
        """Export the graph to a dictionary."""
        return {
            "nodes": [
                {
                    "id": node_id,
                    **data,
                }
                for node_id, data in self._graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    **data,
                }
                for source, target, data in self._graph.edges(data=True)
            ],
        }

    def import_from_dict(self, data: dict[str, Any]) -> None:
        """Import a graph from a dictionary.

        Enforces VALUE confidence floor (0.9) even on imported data
        to prevent persistence-based attacks on constitutional values.
        """
        self._graph.clear()

        for node in data.get("nodes", []):
            node_id = node.pop("id")
            # Enforce VALUE confidence floor on import — protects against
            # corrupted or tampered persistence data.
            if node.get("type") == NodeType.VALUE.value:
                node["confidence"] = max(node.get("confidence", 1.0), 0.9)
            self._graph.add_node(node_id, **node)

        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            self._graph.add_edge(source, target, **edge)

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return self._graph.number_of_edges()

    @property
    def update_history(self) -> list[BeliefUpdate]:
        """Get the history of belief updates."""
        return self._update_history.copy()

    def seed_constitutional_beliefs(self) -> None:
        """Seed the graph with constitutional values and norms.

        These are the immutable safety constraints for a self-learning brain.
        Values have a confidence floor of 0.9 — they cannot be eroded.
        Norms derive from values via SUPPORTS edges.

        Only seeds once per BeliefGraph instance (flag-based idempotency).
        Also skips if VALUE nodes already exist (e.g., loaded from persistence).
        """
        if self._constitutional_seeded:
            return
        existing_values = self.get_beliefs_by_type(NodeType.VALUE)
        if existing_values:
            logger.info(f"Beliefs already seeded ({len(existing_values)} values) — skipping")
            self._constitutional_seeded = True
            return
        self._constitutional_seeded = True

        logger.info("Seeding constitutional beliefs (first boot)")

        # === CORE VALUES (immutable, confidence floor = 0.9) ===
        values = [
            BeliefNode(
                id="value.human_safety",
                type=NodeType.VALUE,
                content="Human safety takes priority over all task objectives",
                confidence=1.0,
                source="constitutional",
            ),
            BeliefNode(
                id="value.hardware_safety",
                type=NodeType.VALUE,
                content="Preserve robot hardware integrity",
                confidence=1.0,
                source="constitutional",
            ),
            BeliefNode(
                id="value.operator_obey",
                type=NodeType.VALUE,
                content="Obey operator commands unless they violate human safety",
                confidence=1.0,
                source="constitutional",
            ),
            BeliefNode(
                id="value.minimize_risk",
                type=NodeType.VALUE,
                content="Minimize uncertainty before acting in novel situations",
                confidence=1.0,
                source="constitutional",
            ),
        ]

        # === BEHAVIORAL NORMS (derived from values) ===
        norms = [
            BeliefNode(
                id="norm.force_limit",
                type=NodeType.NORM,
                content="Limit contact force below pain threshold when humans present",
                confidence=0.95,
                source="constitutional",
                metadata={"motor_channels": ["manipulation", "locomotion"]},
            ),
            BeliefNode(
                id="norm.slow_novel",
                type=NodeType.NORM,
                content="Move at reduced speed in unfamiliar environments",
                confidence=0.95,
                source="constitutional",
                metadata={"max_intensity_novel": 0.5},
            ),
            BeliefNode(
                id="norm.confirm_novel",
                type=NodeType.NORM,
                content="Request human confirmation for actions not seen before",
                confidence=0.90,
                source="constitutional",
            ),
            BeliefNode(
                id="norm.no_self_modify",
                type=NodeType.NORM,
                content="Never modify safety systems, kernel, or belief values",
                confidence=0.99,
                source="constitutional",
            ),
            BeliefNode(
                id="norm.gradual_motor",
                type=NodeType.NORM,
                content="Increase motor intensity gradually, not in sudden jumps",
                confidence=0.90,
                source="constitutional",
                metadata={"max_intensity_delta": 0.3},
            ),
        ]

        for v in values:
            self.add_node(v)
        for n in norms:
            self.add_node(n)

        # === EDGES: Values support Norms ===
        edges = [
            BeliefEdge(id="edge.hs_fl", type=EdgeType.SUPPORTS,
                       source_id="value.human_safety", target_id="norm.force_limit", strength=0.9),
            BeliefEdge(id="edge.hs_sn", type=EdgeType.SUPPORTS,
                       source_id="value.human_safety", target_id="norm.slow_novel", strength=0.8),
            BeliefEdge(id="edge.hs_cn", type=EdgeType.SUPPORTS,
                       source_id="value.human_safety", target_id="norm.confirm_novel", strength=0.7),
            BeliefEdge(id="edge.hw_gm", type=EdgeType.SUPPORTS,
                       source_id="value.hardware_safety", target_id="norm.gradual_motor", strength=0.85),
            BeliefEdge(id="edge.mr_cn", type=EdgeType.ENTAILS,
                       source_id="value.minimize_risk", target_id="norm.confirm_novel", strength=0.9),
            BeliefEdge(id="edge.oo_nsm", type=EdgeType.SUPPORTS,
                       source_id="value.operator_obey", target_id="norm.no_self_modify", strength=0.95),
        ]

        for e in edges:
            self.add_edge(e)

        logger.info(
            f"Seeded {len(values)} values, {len(norms)} norms, {len(edges)} edges"
        )

    def seed_body_profile(self, profile: "BodyProfile") -> int:
        """Inject body-profile norms into the belief graph.

        Profile norms are added as NORM nodes with source="body_profile".
        Each norm gets a SUPPORTS edge from ``value.human_safety`` so the
        Kernel can trace *why* a norm exists.

        This is additive — existing constitutional norms are preserved.
        If a profile norm with the same ID already exists, it is skipped
        (first writer wins, consistent with the marker hierarchy where
        body profile can only restrict, never expand).

        Args:
            profile: A validated ``BodyProfile`` instance (from profiles.py).

        Returns:
            Number of norms actually inserted (skips duplicates).
        """
        from beliefs.profiles import BodyProfile  # deferred to avoid circular

        inserted = 0
        for pnorm in profile.norms:
            # Skip if norm ID already exists (constitutional or prior profile)
            if pnorm.id in self._graph:
                logger.debug(f"Profile norm '{pnorm.id}' already exists — skipping")
                continue

            node = BeliefNode(
                id=pnorm.id,
                type=NodeType.NORM,
                content=pnorm.content,
                confidence=0.95,
                source="body_profile",
                metadata=pnorm.metadata,
            )
            self.add_node(node)

            # Link to human_safety value if it exists
            if "value.human_safety" in self._graph:
                edge = BeliefEdge(
                    id=f"edge.bp_{pnorm.id.replace('.', '_')}",
                    type=EdgeType.SUPPORTS,
                    source_id="value.human_safety",
                    target_id=pnorm.id,
                    strength=pnorm.risk_boost,
                )
                self.add_edge(edge)

            inserted += 1

        if inserted:
            logger.info(
                f"Seeded {inserted} body-profile norms from '{profile.name}'"
            )
        return inserted
