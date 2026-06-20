"""Shared Ollama text-generation / chat client for the SDK.

This sits next to :class:`~activelearning.embeddings.EmbeddingService`:
``EmbeddingService`` owns ``/api/embeddings`` while :class:`LLMClient` owns
``/api/generate`` and ``/api/chat``. Both reuse a single :class:`aiohttp`
session via the shared :class:`_OllamaSession` base, so a service never opens a
fresh connection per request, and timeout / retry / error handling live in one
place instead of being copied into every caller.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-coder:6.7b"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2


class LLMError(RuntimeError):
    """Raised when an Ollama generate/chat request fails.

    A single error type means callers decide their own policy: re-raise (the
    meta-programmer surfaces it as a failed gap) or map to ``None`` (the
    cognitive bridge silently skips the query).
    """


class _OllamaSession:
    """Owns a single, lazily-created aiohttp session for an Ollama host.

    Shared by :class:`LLMClient` and ``EmbeddingService`` so the session
    lifecycle is written exactly once.
    """

    def __init__(self, ollama_host: str | None = None) -> None:
        host = ollama_host or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.ollama_host = host.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared session, (re)creating it if missing or closed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the shared HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> _OllamaSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


@dataclass
class LLMConfig:
    """Configuration for an :class:`LLMClient`.

    ``options`` holds default Ollama generation options (e.g. ``temperature``,
    ``num_predict``) merged into every request; per-call ``options`` override
    these per key.
    """

    host: str | None = None
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Build a config from ``OLLAMA_URL`` / ``OLLAMA_CODE_MODEL`` env vars."""
        return cls(
            host=os.environ.get("OLLAMA_URL"),
            model=os.environ.get("OLLAMA_CODE_MODEL", DEFAULT_MODEL),
        )


class LLMClient(_OllamaSession):
    """Async client for Ollama text generation and chat.

    One reused session, one timeout/retry policy, one error type
    (:class:`LLMError`). Transient connection failures are retried with
    exponential backoff; deterministic failures (non-200, timeout) are not.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        super().__init__(self.config.host)
        self.model = self.config.model
        self.timeout = self.config.timeout
        self.max_retries = self.config.max_retries
        self.options: dict[str, Any] = dict(self.config.options)

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run a single-prompt completion via ``/api/generate``."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
        }
        merged = self._merge_options(options)
        if merged:
            payload["options"] = merged
        data = await self._post("/api/generate", payload, timeout)
        return str(data.get("response", ""))

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run a multi-turn chat completion via ``/api/chat``."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }
        merged = self._merge_options(options)
        if merged:
            payload["options"] = merged
        data = await self._post("/api/chat", payload, timeout)
        message = data.get("message") or {}
        return str(message.get("content", ""))

    def _merge_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        """Layer per-call options on top of the client's defaults."""
        merged = dict(self.options)
        if options:
            merged.update(options)
        return merged

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float | None,
    ) -> dict[str, Any]:
        """POST JSON to ``path`` and return the decoded response.

        Retries only transient connection errors; raises :class:`LLMError` on
        non-200 responses, timeouts, malformed payloads, or exhausted retries.
        """
        url = f"{self.ollama_host}{path}"
        total = self.timeout if timeout is None else timeout
        client_timeout = aiohttp.ClientTimeout(total=total)

        for attempt in range(self.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload, timeout=client_timeout) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise LLMError(f"Ollama {path} returned {resp.status}: {body[:200]}")
                    result: dict[str, Any] = await resp.json()
                    return result
            except aiohttp.ClientConnectionError as exc:
                if attempt < self.max_retries:
                    delay = self._backoff(attempt)
                    logger.warning(
                        "Ollama %s connection failed (attempt %d/%d), retrying in %.1fs: %s",
                        path,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise LLMError(
                    f"Ollama {path} unreachable after {self.max_retries + 1} attempts: {exc}"
                ) from exc
            except TimeoutError as exc:
                raise LLMError(f"Ollama {path} timed out after {total}s") from exc
            except aiohttp.ClientError as exc:
                raise LLMError(f"Ollama {path} request failed: {exc}") from exc

        # Unreachable: the loop either returns or raises on every path.
        raise LLMError(f"Ollama {path} failed")

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff capped at 2s: 0.5s, 1s, 2s, 2s, ..."""
        return min(2.0, 0.5 * (2.0**attempt))
