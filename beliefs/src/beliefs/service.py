"""
Beliefs Service - Manages the belief graph.

This service maintains the system's belief graph, handling updates,
queries, and contradiction detection.  Supports request-reply queries
so the Kernel can synchronously check beliefs before decisions.
"""

import asyncio
import json
from dataclasses import asdict

from activelearning import BaseService
from activelearning.nats_client import serialize_message

from beliefs.graph import BeliefGraph, BeliefNode, BeliefEdge, NodeType, EdgeType


class BeliefsService(BaseService):
    """
    Belief graph service.

    Manages the belief graph with persistence to SQLite and
    real-time updates via NATS.
    """

    def __init__(self):
        super().__init__("beliefs", use_database=True, use_event_bus=True)
        self._graph = BeliefGraph()

    async def _setup(self) -> None:
        """Service-specific setup."""
        # Load existing beliefs from database
        await self._load_from_db()

        # Seed constitutional beliefs (values + norms) on first boot.
        # Idempotent — skips if values already exist from DB load.
        self._graph.seed_constitutional_beliefs()

        # Subscribe to belief events using EventBus
        await self.event_bus.subscribe("beliefs.add_node", self._handle_add_node)
        await self.event_bus.subscribe("beliefs.add_edge", self._handle_add_edge)
        await self.event_bus.subscribe("beliefs.update", self._handle_update)
        await self.event_bus.subscribe("beliefs.query", self._handle_query)
        await self.event_bus.subscribe("beliefs.contradictions", self._handle_contradictions)
        # Request-reply query for synchronous callers (Kernel safety checks)
        await self.event_bus.subscribe(
            "beliefs.query.request",
            self._handle_query_request,
            is_request_handler=True,
        )

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        # Save to database before shutdown
        await self._save_to_db()

    async def _load_from_db(self) -> None:
        """Load beliefs from SQLite."""
        try:
            # Load nodes
            rows = await self.database.fetchall(
                "SELECT id, type, content, confidence, source, metadata, created_at, updated_at FROM belief_nodes"
            )

            for row in rows:
                node = BeliefNode(
                    id=row["id"],
                    type=NodeType(row["type"]),
                    content=row["content"],
                    confidence=row["confidence"],
                    source=row["source"],
                    metadata=json.loads(row["metadata"] or "{}"),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                self._graph.add_node(node)

            # Load edges
            edge_rows = await self.database.fetchall(
                "SELECT id, type, source_id, target_id, strength, evidence, created_at FROM belief_edges"
            )

            for row in edge_rows:
                try:
                    edge = BeliefEdge(
                        id=row["id"],
                        type=EdgeType(row["type"]),
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        strength=row["strength"],
                        evidence=row["evidence"],
                        created_at=row["created_at"],
                    )
                    self._graph.add_edge(edge)
                except ValueError as e:
                    self.logger.warning(f"Skipping invalid edge: {e}")

            self.logger.info(
                f"Loaded {self._graph.node_count} nodes and {self._graph.edge_count} edges from database"
            )
        except Exception as e:
            self.logger.warning(f"Could not load beliefs from database: {e}")

    async def _save_to_db(self) -> None:
        """Save beliefs to SQLite."""
        try:
            # Export graph
            data = self._graph.export_to_dict()

            # Clear existing data
            await self.database.execute("DELETE FROM belief_edges")
            await self.database.execute("DELETE FROM belief_nodes")

            # Insert nodes
            for node in data["nodes"]:
                await self.database.insert(
                    "belief_nodes",
                    {
                        "id": node["id"],
                        "type": node["type"],
                        "content": node["content"],
                        "confidence": node["confidence"],
                        "source": node["source"],
                        "metadata": json.dumps(node.get("metadata", {})),
                        "created_at": node["created_at"],
                        "updated_at": node["updated_at"],
                    },
                )

            # Insert edges
            for edge in data["edges"]:
                await self.database.insert(
                    "belief_edges",
                    {
                        "id": edge["id"],
                        "type": edge["type"],
                        "source_id": edge["source"],
                        "target_id": edge["target"],
                        "strength": edge["strength"],
                        "evidence": edge.get("evidence"),
                        "created_at": edge["created_at"],
                    },
                )

            self.logger.info(f"Saved {len(data['nodes'])} nodes and {len(data['edges'])} edges to database")
        except Exception as e:
            self.logger.error(f"Error saving beliefs to database: {e}")

    async def _handle_add_node(self, data: dict) -> None:
        """Handle add node requests."""
        try:
            node = BeliefNode(
                id=data.get("id", ""),
                type=NodeType(data["type"]),
                content=data["content"],
                confidence=data.get("confidence", 1.0),
                source=data.get("source", "unknown"),
                metadata=data.get("metadata", {}),
            )
            node_id = self._graph.add_node(node)
        except Exception as e:
            self.logger.error(f"Error adding node: {e}")

    async def _handle_add_edge(self, data: dict) -> None:
        """Handle add edge requests."""
        try:
            edge = BeliefEdge(
                id=data.get("id", ""),
                type=EdgeType(data["type"]),
                source_id=data["source_id"],
                target_id=data["target_id"],
                strength=data.get("strength", 1.0),
                evidence=data.get("evidence"),
            )
            edge_id = self._graph.add_edge(edge)
        except Exception as e:
            self.logger.error(f"Error adding edge: {e}")

    async def _handle_update(self, data: dict) -> None:
        """Handle belief update requests."""
        try:
            update = self._graph.update_belief(
                node_id=data["node_id"],
                evidence_strength=data["evidence_strength"],
                supports=data.get("supports", True),
                source=data.get("source", "unknown"),
            )
        except Exception as e:
            self.logger.error(f"Error updating belief: {e}")

    async def _handle_query(self, data: dict) -> None:
        """Handle belief query requests."""
        try:
            query_type = data.get("type", "by_id")

            result = None
            if query_type == "by_id":
                node = self._graph.get_node(data["node_id"])
                result = asdict(node) if node else None
            elif query_type == "by_type":
                nodes = self._graph.get_beliefs_by_type(NodeType(data["node_type"]))
                result = [asdict(n) for n in nodes]
            elif query_type == "high_confidence":
                nodes = self._graph.get_high_confidence_beliefs(data.get("threshold", 0.8))
                result = [asdict(n) for n in nodes]
            elif query_type == "supporting":
                result = self._graph.get_supporting_beliefs(data["node_id"])
            elif query_type == "contradicting":
                result = self._graph.get_contradicting_beliefs(data["node_id"])
            elif query_type == "export":
                result = self._graph.export_to_dict()
        except Exception as e:
            self.logger.error(f"Error querying beliefs: {e}")

    async def _handle_contradictions(self, data: dict) -> None:
        """Handle contradiction detection requests."""
        try:
            threshold = data.get("threshold", 0.5)
            contradictions = self._graph.find_contradictions(threshold)
        except Exception as e:
            self.logger.error(f"Error finding contradictions: {e}")

    async def _handle_query_request(self, data: dict, msg=None) -> None:
        """Handle synchronous belief queries via NATS request-reply.

        Used by the Kernel to check norms before making safety decisions.
        Supports the same query types as _handle_query but replies directly.
        """
        try:
            query_type = data.get("type", "by_type")
            result: list | dict | None = None

            if query_type == "by_id":
                node = self._graph.get_node(data["node_id"])
                result = asdict(node) if node else None
            elif query_type == "by_type":
                nodes = self._graph.get_beliefs_by_type(NodeType(data["node_type"]))
                result = [asdict(n) for n in nodes]
            elif query_type == "high_confidence":
                nodes = self._graph.get_high_confidence_beliefs(data.get("threshold", 0.8))
                result = [asdict(n) for n in nodes]
            elif query_type == "norms":
                # Convenience: get high-confidence norms (used by Kernel)
                norms = self._graph.get_beliefs_by_type(NodeType.NORM)
                threshold = data.get("threshold", 0.8)
                result = [asdict(n) for n in norms if n.confidence >= threshold]
            elif query_type == "export":
                result = self._graph.export_to_dict()

            response = {"result": result, "query_type": query_type}

            if msg and msg.reply:
                await msg.respond(serialize_message(response))
            else:
                # Fallback publish (shouldn't happen for request-reply)
                await self.event_bus.publish("beliefs.query.response", response)

        except Exception as e:
            self.logger.error(f"Error handling query request: {e}")
            if msg and msg.reply:
                await msg.respond(serialize_message({"result": None, "error": str(e)}))


async def main() -> None:
    """Main entry point."""
    service = BeliefsService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
