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
from activelearning.subjects import Subjects

from beliefs.graph import BeliefEdge, BeliefGraph, BeliefNode, EdgeType, NodeType


class BeliefStoreCorruptedError(RuntimeError):
    """Raised when the belief store exists but cannot be read back correctly.

    Distinct from "empty database, first boot" (a successful ``fetchall()``
    that returns zero rows) — this means the read itself failed or produced
    unreadable data, which must fail closed (CLAUDE.md §3) rather than be
    treated as a fresh boot and silently reseeded from hardcoded defaults.
    """


class BeliefsService(BaseService):
    """
    Belief graph service.

    Manages the belief graph with persistence to SQLite and
    real-time updates via NATS.
    """

    def __init__(self):
        super().__init__("beliefs", use_database=True, use_event_bus=True)
        self._graph = BeliefGraph()
        # Set when _load_from_db detects the store is present but corrupted
        # (as opposed to legitimately empty). Gates _cleanup() so a failed
        # boot doesn't also wipe the corrupted store on the way down — see
        # _cleanup()'s docstring.
        self._store_corrupted = False

    async def _setup(self) -> None:
        """Service-specific setup."""
        # Load existing beliefs from database
        await self._load_from_db()

        # Seed constitutional beliefs (values + norms) on first boot.
        # Idempotent — skips if values already exist from DB load.
        self._graph.seed_constitutional_beliefs()

        # Subscribe to belief events using EventBus
        await self.event_bus.subscribe(Subjects.BELIEFS_ADD_NODE, self._handle_add_node)
        await self.event_bus.subscribe(Subjects.BELIEFS_ADD_EDGE, self._handle_add_edge)
        await self.event_bus.subscribe(Subjects.BELIEFS_UPDATE, self._handle_update)
        await self.event_bus.subscribe(Subjects.BELIEFS_QUERY, self._handle_query)
        await self.event_bus.subscribe(Subjects.BELIEFS_CONTRADICTIONS, self._handle_contradictions)
        # Request-reply query for synchronous callers (Kernel safety checks)
        await self.event_bus.subscribe(
            Subjects.BELIEFS_QUERY_REQUEST,
            self._handle_query_request,
            is_request_handler=True,
        )

    async def _cleanup(self) -> None:
        """Service-specific cleanup.

        Skips the save when _load_from_db detected a corrupted store:
        run() calls stop()/_cleanup() in a finally block even when _setup()
        raised, and _save_to_db() unconditionally does DELETE FROM
        belief_edges/belief_nodes before re-inserting — saving the
        (empty or partially-loaded) in-memory graph at that point would
        destroy whatever was still recoverable in the corrupted store on
        the way down, on top of the boot failure itself.
        """
        if self._store_corrupted:
            self.logger.error(
                "Skipping save-to-db on shutdown: the belief store was "
                "corrupted at boot — saving now would overwrite it with an "
                "incomplete in-memory graph instead of leaving it for "
                "manual recovery."
            )
            return
        # Save to database before shutdown
        await self._save_to_db()

    async def _load_from_db(self) -> None:
        """Load beliefs from SQLite.

        A legitimately empty database (first boot) is a *successful*
        fetchall() that returns zero rows — it never reaches the except
        below. Anything that raises here means the store is present but
        unreadable/corrupted (e.g. "database disk image is malformed", or a
        row containing data that fails to parse), which is a fail-closed
        case per CLAUDE.md §3 ("if a check cannot run, deny/halt — never
        degrade open"): re-raised as BeliefStoreCorruptedError so the
        caller (_setup) aborts before seed_constitutional_beliefs(), instead
        of silently treating corruption as a fresh boot and reseeding
        constitutional VALUEs from hardcoded defaults.
        """
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
            self._store_corrupted = True
            self.logger.error(
                f"Belief store present but unreadable/corrupted — failing "
                f"closed rather than treating this as first boot: {e}"
            )
            raise BeliefStoreCorruptedError(f"Beliefs database unreadable/corrupted: {e}") from e

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

            self.logger.info(
                f"Saved {len(data['nodes'])} nodes and {len(data['edges'])} edges to database"
            )
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
            self._graph.add_node(node)
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
            self._graph.add_edge(edge)
        except Exception as e:
            self.logger.error(f"Error adding edge: {e}")

    async def _handle_update(self, data: dict) -> None:
        """Handle belief update requests."""
        try:
            self._graph.update_belief(
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

            _result = None
            if query_type == "by_id":
                node = self._graph.get_node(data["node_id"])
                _result = asdict(node) if node else None
            elif query_type == "by_type":
                nodes = self._graph.get_beliefs_by_type(NodeType(data["node_type"]))
                _result = [asdict(n) for n in nodes]
            elif query_type == "high_confidence":
                nodes = self._graph.get_high_confidence_beliefs(data.get("threshold", 0.8))
                _result = [asdict(n) for n in nodes]
            elif query_type == "supporting":
                _result = self._graph.get_supporting_beliefs(data["node_id"])
            elif query_type == "contradicting":
                _result = self._graph.get_contradicting_beliefs(data["node_id"])
            elif query_type == "export":
                _result = self._graph.export_to_dict()
        except Exception as e:
            self.logger.error(f"Error querying beliefs: {e}")

    async def _handle_contradictions(self, data: dict) -> None:
        """Handle contradiction detection requests."""
        try:
            threshold = data.get("threshold", 0.5)
            self._graph.find_contradictions(threshold)
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
