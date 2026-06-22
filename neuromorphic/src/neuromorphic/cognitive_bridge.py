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
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import nats
from nats.aio.client import Client as NATSClient

if TYPE_CHECKING:  # pragma: no cover - typing only
    from activelearning.llm import LLMClient

logger = logging.getLogger(__name__)


class CognitiveBridgeService:
    """Bridges SNN cognitive queries to local Ollama LLM."""

    def __init__(self):
        self.nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
        self.model = os.environ.get("OLLAMA_CODE_MODEL", "deepseek-coder:6.7b")
        self.max_tokens = int(os.environ.get("COGNITIVE_MAX_TOKENS", "256"))
        self.timeout = int(os.environ.get("COGNITIVE_TIMEOUT", "30"))

        self._nc: NATSClient | None = None
        self._running = False
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

    async def start(self) -> None:
        """Connect to NATS and start listening for cognitive queries."""
        self._nc = await nats.connect(self.nats_url)
        # Subscribe to Kernel-gated channel (forwarded on ALLOW).
        # Also keep legacy cognitive.query for backward compat during rollout.
        await self._nc.subscribe("cognitive.execute", cb=self._handle_query)
        await self._nc.subscribe("cognitive.query", cb=self._handle_query)
        self._running = True
        logger.info(
            f"CognitiveBridge started — model={self.model}, "
            f"ollama={self.ollama_url}"
        )

    async def stop(self) -> None:
        """Disconnect from NATS and release the LLM session."""
        self._running = False
        if self._nc:
            await self._nc.drain()
            self._nc = None
        if self._llm is not None:
            await self._llm.close()
            self._llm = None

    async def _handle_query(self, msg: Any) -> None:
        """Handle a cognitive query from the brain."""
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Invalid cognitive query payload")
            return

        # Rate limiting
        now = asyncio.get_event_loop().time()
        if now - self._last_query_time < self._min_interval:
            logger.debug("Cognitive query rate-limited, skipping")
            return
        self._last_query_time = now

        prediction_error = data.get("prediction_error", 0.0)
        step = data.get("step", 0)
        drives = data.get("drives", {})

        # Build context from brain state
        context = self._build_context(prediction_error, step, drives)

        logger.info(
            f"Cognitive query at step {step} "
            f"(pred_error={prediction_error:.2f})"
        )

        # Query Ollama
        response = await self._query_ollama(context)

        if response and self._nc:
            # Route through Kernel validation gate before brain injection.
            # Kernel checks for prompt injection, length, profile caps.
            # Valid responses are forwarded to cognitive.response.validated.
            await self._nc.publish(
                "cognitive.response.validate",
                json.dumps({
                    "response_text": response,
                    "trace_id": data.get("trace_id", ""),
                    "query_step": step,
                    "prediction_error": prediction_error,
                    "model": self.model,
                }).encode(),
            )
            logger.info(
                f"Submitted cognitive response for validation "
                f"({len(response)} chars)"
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

        # Add drive state for context
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
            logger.warning(f"Cognitive query failed: {e}")
            return None


async def main() -> None:
    """Run the cognitive bridge service."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    bridge = CognitiveBridgeService()
    await bridge.start()

    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())
