"""Cognitive Bridge — connects the SNN brain to external LLMs via NATS.

Listens for cognitive.execute events (forwarded by Kernel after ALLOW
decision) and queries a local Ollama instance. Publishes the response
to the Kernel validation gate before it reaches the brain.

Flow: brain fires cognitive motor → proposal.new → Kernel ALLOW →
      cognitive.execute → this bridge → Ollama →
      cognitive.response.validate → Kernel validates →
      cognitive.response.validated → brain injects

This service is intentionally simple — the brain decides WHEN to ask (via
learned prediction-error pathways), the Kernel decides IF it's allowed,
this bridge just executes the query, and the Kernel validates the response
before injection.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from activelearning import BaseService
from activelearning.subjects import Subjects

if TYPE_CHECKING:  # pragma: no cover - typing only
    from activelearning.llm import LLMClient

logger = logging.getLogger(__name__)


class CognitiveBridgeService(BaseService):
    """Bridges SNN cognitive queries to local Ollama LLM."""

    def __init__(self) -> None:
        """Load Ollama/NATS settings from env and initialize BaseService."""
        super().__init__("cognitive-bridge", use_database=False, use_event_bus=True)
        self.ollama_url = os.environ.get("OLLAMA_URL", self.config.ollama_url)
        self.model = os.environ.get("OLLAMA_CODE_MODEL", "deepseek-coder:6.7b")
        self.max_tokens = int(os.environ.get("COGNITIVE_MAX_TOKENS", "256"))
        self.timeout = int(os.environ.get("COGNITIVE_TIMEOUT", "30"))

        # Built lazily on first query: the SDK is not installed in the
        # neuromorphic-only test environment, so the import is deferred to
        # runtime (where the launcher puts the SDK on PYTHONPATH).
        self._llm: LLMClient | None = None

        # Rate limiting — max 1 query per N seconds
        self._min_interval = float(os.environ.get("COGNITIVE_MIN_INTERVAL", "5.0"))
        self._last_query_time: float = 0.0

        # System prompt that frames queries for the LLM
        self._system_prompt = (
            "You are an assistant embedded in a neuromorphic robot brain. "
            "The brain's spiking neural network has detected something it "
            "cannot predict. Provide a concise, factual explanation that "
            "helps the brain learn. Keep responses under 100 words."
        )

    async def _setup(self) -> None:
        """Subscribe to Kernel-gated and legacy cognitive query subjects."""
        event_bus = self.event_bus
        if event_bus is None:
            raise RuntimeError("Event bus is not initialized")

        # Kernel-gated channel (forwarded on ALLOW).
        await event_bus.subscribe(Subjects.COGNITIVE_EXECUTE, self._handle_query)
        # Legacy subject for backward compat during rollout.
        await event_bus.subscribe(Subjects.COGNITIVE_QUERY, self._handle_query)
        self.logger.info(
            "CognitiveBridge ready — model=%s, ollama=%s",
            self.model,
            self.ollama_url,
        )

    async def _cleanup(self) -> None:
        """Release the lazy LLM client session."""
        if self._llm is not None:
            await self._llm.close()
            self._llm = None

    async def _handle_query(self, data: dict[str, Any]) -> None:
        """Handle a cognitive query from the brain."""
        # Rate limiting
        now = asyncio.get_running_loop().time()
        if now - self._last_query_time < self._min_interval:
            self.logger.debug("Cognitive query rate-limited, skipping")
            return
        self._last_query_time = now

        prediction_error = data.get("prediction_error", 0.0)
        step = data.get("step", 0)
        drives = data.get("drives", {})

        context = self._build_context(prediction_error, step, drives)

        self.logger.info(
            "Cognitive query at step %s (pred_error=%.2f)",
            step,
            prediction_error,
        )

        response = await self._query_ollama(context)

        if response:
            event_bus = self.event_bus
            if event_bus is None:
                raise RuntimeError("Event bus is not initialized")

            # Route through Kernel validation gate before brain injection.
            await event_bus.publish(
                Subjects.COGNITIVE_RESPONSE_VALIDATE,
                {
                    "response_text": response,
                    "trace_id": data.get("trace_id", ""),
                    "query_step": step,
                    "prediction_error": prediction_error,
                    "model": self.model,
                },
            )
            self.logger.info(
                "Submitted cognitive response for validation (%d chars)",
                len(response),
            )

    def _build_context(
        self,
        prediction_error: float,
        step: int,
        drives: dict[str, Any],
    ) -> str:
        """Build a query context string from brain state."""
        parts = [
            f"The brain is confused (prediction error: {prediction_error:.2f}).",
        ]

        if drives:
            critical = []
            for drive, level in drives.items():
                if isinstance(level, (int, float)) and level < 0.3:
                    critical.append(f"{drive} is low ({level:.2f})")
                elif isinstance(level, (int, float)) and level > 0.8:
                    critical.append(f"{drive} is high ({level:.2f})")
            if critical:
                parts.append("Drive state: " + ", ".join(critical) + ".")

        parts.append(
            "What might explain this unexpected input? "
            "Describe the most likely situation briefly."
        )

        return " ".join(parts)

    async def _query_ollama(self, prompt: str) -> str | None:
        """Query local Ollama via the shared SDK client; ``None`` on failure."""
        # Deferred import: the SDK is absent in the neuromorphic-only test env.
        from activelearning.llm import LLMClient, LLMConfig, LLMError

        if self._llm is None:
            self._llm = LLMClient(
                LLMConfig(
                    host=self.ollama_url,
                    model=self.model,
                    timeout=self.timeout,
                    options={"num_predict": self.max_tokens},
                )
            )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            return await self._llm.chat(messages)
        except LLMError as e:
            self.logger.warning("Cognitive query failed: %s", e)
            return None


async def main() -> None:
    """Run the cognitive bridge service."""
    service = CognitiveBridgeService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
