"""
External API Manager - Handles external knowledge queries safely.

Safety rules:
1. Check local knowledge first
2. Request Kernel approval before external queries
3. No sensitive data sent to external APIs
4. Compare external with local knowledge
5. Human escalation for conflicts
"""

import logging
import os
from typing import Any

from activelearning import EventBus, current_timestamp, generate_trace_id

from external_api.conflict_detection import detect_knowledge_conflict

logger = logging.getLogger(__name__)


class ExternalAPIManager:
    """
    Manages external API queries with safety checks.
    """

    def __init__(self, event_bus: EventBus, db: Any, embedding_service: Any | None = None):
        self.event_bus = event_bus
        self.db = db
        self._embedding_service = embedding_service

        # API keys from environment
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        # Initialize clients if keys are available
        self._claude_client = None
        self._openai_client = None

        if self.anthropic_api_key:
            try:
                from anthropic import AsyncAnthropic

                self._claude_client = AsyncAnthropic(api_key=self.anthropic_api_key)
                logger.info("Claude API client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Claude client: {e}")

        if self.openai_api_key:
            try:
                from openai import AsyncOpenAI

                self._openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI API client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

    async def query_external(
        self,
        query: str,
        context: dict | None = None,
        local_knowledge: dict | None = None,
    ) -> dict:
        """
        Query external APIs with safety checks.

        Flow:
        1. Check if external queries are allowed
        2. Request Kernel approval
        3. Query external API (Claude or OpenAI)
        4. Compare with local knowledge
        5. Detect conflicts
        6. Store validated knowledge

        Returns:
            dict with:
                - success: bool
                - response: str (if success)
                - conflict: bool (if contradicts local knowledge)
                - error: str (if failed)
        """
        try:
            # Step 1: Request Kernel approval for external query
            kernel_approved = await self._request_kernel_approval(query)
            if not kernel_approved:
                return {
                    "success": False,
                    "response": "",
                    "conflict": False,
                    "error": "Kernel denied external query",
                }

            # Step 2: Query external API
            if self._claude_client:
                response = await self._query_claude(query, context)
            elif self._openai_client:
                response = await self._query_openai(query, context)
            else:
                return {
                    "success": False,
                    "response": "",
                    "conflict": False,
                    "error": "No external API configured (ANTHROPIC_API_KEY or OPENAI_API_KEY required)",
                }

            # Step 3: Compare with local knowledge
            conflict = False
            if local_knowledge:
                conflict = await self._detect_conflict(response, local_knowledge)

            # Step 4: Log query
            await self._log_query(query, response, conflict)

            # Step 5: If conflict, escalate to human
            if conflict:
                logger.warning(f"Knowledge conflict detected: {query[:50]}...")
                await self._escalate_conflict(query, response, local_knowledge)

            return {
                "success": True,
                "response": response,
                "conflict": conflict,
                "error": None,
            }

        except Exception as e:
            logger.error(f"External query error: {e}", exc_info=True)
            return {
                "success": False,
                "response": "",
                "conflict": False,
                "error": str(e),
            }

    async def _request_kernel_approval(self, query: str) -> bool:
        """Request Kernel approval for external query."""
        try:
            trace_id = generate_trace_id()

            proposal = {
                "trace_id": trace_id,
                "type": "external_query",
                "query": query[:200],  # Truncate for safety
            }

            # Publish to Kernel
            await self.event_bus.publish("proposal.new", proposal)

            # Wait for signed decision via EventBus
            decision = await self.event_bus.wait_for_decision(trace_id, timeout=10.0)
            return decision.get("type") == "ALLOW"
        except TimeoutError:
            logger.error("Timeout waiting for Kernel decision")
            return False
        except Exception as e:
            logger.error(f"Error requesting Kernel approval: {e}")
            return False

    async def _query_claude(self, query: str, context: dict | None) -> str:
        """Query Claude API."""
        try:
            # Build prompt
            prompt = query
            if context:
                prompt = f"Context: {context}\n\nQuery: {query}"

            message = await self._claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            return message.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def _query_openai(self, query: str, context: dict | None) -> str:
        """Query OpenAI API."""
        try:
            # Build prompt
            prompt = query
            if context:
                prompt = f"Context: {context}\n\nQuery: {query}"

            response = await self._openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _detect_conflict(self, external_response: str, local_knowledge: dict) -> bool:
        """Detect if external response conflicts with local knowledge."""
        return await detect_knowledge_conflict(
            external_response,
            local_knowledge,
            embedding_service=self._embedding_service,
        )

    async def _log_query(self, query: str, response: str, conflict: bool) -> None:
        """Log external API query."""
        try:
            await self.db.execute(
                """
                INSERT INTO external_api_queries
                (id, query, response, conflict, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    generate_trace_id(),
                    query,
                    response,
                    conflict,
                    current_timestamp(),
                ),
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error logging query: {e}")

    async def _escalate_conflict(
        self,
        query: str,
        external_response: str,
        local_knowledge: dict,
    ) -> None:
        """Escalate knowledge conflict to human."""
        try:
            trace_id = generate_trace_id()

            await self.event_bus.publish(
                "approval.request",
                {
                    "trace_id": trace_id,
                    "type": "knowledge_conflict",
                    "query": query,
                    "external_response": external_response,
                    "local_knowledge": local_knowledge,
                    "message": "Conflicting knowledge detected. Which should be trusted?",
                },
            )

            logger.info(f"Conflict escalated to human: {trace_id}")

        except Exception as e:
            logger.error(f"Error escalating conflict: {e}")
