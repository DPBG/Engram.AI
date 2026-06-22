"""Tests for the shared SDK LLM client and the no-duplicate-session guard."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import aiohttp
import pytest

from activelearning.llm import (
    DEFAULT_MODEL,
    LLMClient,
    LLMConfig,
    LLMError,
)


# --------------------------------------------------------------------------- #
# Test doubles: a fake aiohttp session that replays a scripted list of
# behaviours (a response to yield, or an exception to raise) per POST.
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: dict | None = None,
        text: str = "",
        json_exc: Exception | None = None,
    ):
        self.status = status
        self._payload = payload if payload is not None else {}
        self._text = text
        self._json_exc = json_exc

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def json(self) -> dict:
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload

    async def text(self) -> str:
        return self._text


class FakeSession:
    def __init__(self, behaviours: list[Any]):
        self._behaviours = list(behaviours)
        self.closed = False
        self.calls: list[dict] = []

    def post(self, url: str, json: dict | None = None, timeout: Any = None) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        behaviour = self._behaviours.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour

    async def close(self) -> None:
        self.closed = True


def _client(behaviours: list[Any], **cfg: Any) -> LLMClient:
    client = LLMClient(LLMConfig(host="http://ollama:11434", **cfg))
    client._session = FakeSession(behaviours)  # type: ignore[assignment]
    return client


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Make retry backoff instant so tests don't actually wait."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("activelearning.llm.asyncio.sleep", _instant)


class TestGenerate:
    @pytest.mark.asyncio
    async def test_returns_response_text(self):
        client = _client(
            [FakeResponse(200, {"response": "print('hi')"})],
            model="deepseek-coder:6.7b",
            options={"temperature": 0.2},
        )
        out = await client.generate("write code", options={"top_p": 0.9})

        assert out == "print('hi')"
        call = client._session.calls[0]  # type: ignore[attr-defined]
        assert call["url"] == "http://ollama:11434/api/generate"
        assert call["json"]["model"] == "deepseek-coder:6.7b"
        assert call["json"]["stream"] is False
        # Per-call options layer on top of the client defaults.
        assert call["json"]["options"] == {"temperature": 0.2, "top_p": 0.9}

    @pytest.mark.asyncio
    async def test_missing_response_key_yields_empty_string(self):
        client = _client([FakeResponse(200, {})])
        assert await client.generate("x") == ""


class TestChat:
    @pytest.mark.asyncio
    async def test_returns_message_content(self):
        client = _client([FakeResponse(200, {"message": {"content": "hello"}})])
        out = await client.chat([{"role": "user", "content": "hi"}])

        assert out == "hello"
        call = client._session.calls[0]  # type: ignore[attr-defined]
        assert call["url"] == "http://ollama:11434/api/chat"
        assert call["json"]["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_missing_message_yields_empty_string(self):
        client = _client([FakeResponse(200, {})])
        assert await client.chat([{"role": "user", "content": "hi"}]) == ""


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_non_200_raises_without_retry(self):
        client = _client([FakeResponse(500, text="boom")], max_retries=2)
        with pytest.raises(LLMError) as excinfo:
            await client.generate("x")
        assert "500" in str(excinfo.value)
        # Deterministic server errors are not retried.
        assert len(client._session.calls) == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_timeout_raises_without_retry(self):
        client = _client([TimeoutError()], max_retries=2)
        with pytest.raises(LLMError) as excinfo:
            await client.generate("x")
        assert "timed out" in str(excinfo.value)
        assert len(client._session.calls) == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_invalid_json_raises_without_retry(self):
        # A 200 whose body does not parse as JSON (json.JSONDecodeError is a
        # ValueError) must surface as LLMError, not leak a raw ValueError.
        client = _client(
            [FakeResponse(200, json_exc=json.JSONDecodeError("Expecting value", "", 0))],
            max_retries=2,
        )
        with pytest.raises(LLMError) as excinfo:
            await client.generate("x")
        assert "invalid JSON" in str(excinfo.value)
        assert len(client._session.calls) == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_connection_error_retries_then_succeeds(self):
        client = _client(
            [
                aiohttp.ClientConnectionError("refused"),
                FakeResponse(200, {"response": "ok"}),
            ],
            max_retries=2,
        )
        assert await client.generate("x") == "ok"
        assert len(client._session.calls) == 2  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_connection_error_exhausts_retries(self):
        client = _client(
            [aiohttp.ClientConnectionError("refused")] * 3,
            max_retries=2,
        )
        with pytest.raises(LLMError) as excinfo:
            await client.generate("x")
        assert "unreachable" in str(excinfo.value)
        # 1 initial attempt + 2 retries.
        assert len(client._session.calls) == 3  # type: ignore[attr-defined]


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_reuses_single_session(self):
        # `async with` guarantees the session is closed even if an assertion
        # fails, so a real aiohttp session can never leak out of the test.
        async with LLMClient() as client:
            session_one = await client._get_session()
            session_two = await client._get_session()
            assert session_one is session_two
        assert client._session is None


class TestConfig:
    def test_defaults(self):
        client = LLMClient()
        assert client.model == DEFAULT_MODEL
        assert client.ollama_host  # resolved from env/default

    def test_trailing_slash_stripped(self):
        client = LLMClient(LLMConfig(host="http://host:1234/"))
        assert client.ollama_host == "http://host:1234"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://env-host:9999")
        monkeypatch.setenv("OLLAMA_CODE_MODEL", "custom-model")
        cfg = LLMConfig.from_env()
        assert cfg.host == "http://env-host:9999"
        assert cfg.model == "custom-model"


# --------------------------------------------------------------------------- #
# Guard: no service may open its own Ollama session. After migration, every
# service reaches Ollama through the SDK (LLMClient / EmbeddingService), so the
# raw endpoint paths must not appear in service source at all.
# --------------------------------------------------------------------------- #
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_MIGRATED_SERVICE_FILES = [
    "neuromorphic/src/neuromorphic/cognitive_bridge.py",
    "meta-programmer/src/meta_programmer/agents.py",
    "cache/src/cache/llm_cache.py",
    "coordinator/src/coordinator/task_coordinator.py",
]
_OLLAMA_ENDPOINTS = ("/api/generate", "/api/chat", "/api/embeddings")


@pytest.mark.parametrize("rel_path", _MIGRATED_SERVICE_FILES)
def test_service_has_no_direct_ollama_endpoint(rel_path):
    path = REPO_ROOT / rel_path
    assert path.exists(), f"expected migrated service file at {path}"
    source = path.read_text()
    hits = [endpoint for endpoint in _OLLAMA_ENDPOINTS if endpoint in source]
    assert not hits, (
        f"{rel_path} talks to Ollama directly ({', '.join(hits)}). Route text "
        "generation/chat through activelearning.llm.LLMClient and embeddings "
        "through activelearning.EmbeddingService instead of opening a session."
    )


@pytest.mark.parametrize(
    "rel_path",
    [
        "neuromorphic/src/neuromorphic/cognitive_bridge.py",
        "meta-programmer/src/meta_programmer/agents.py",
    ],
)
def test_pure_llm_service_opens_no_aiohttp_session(rel_path):
    """Services whose only HTTP use was Ollama must not build a session at all."""
    source = (REPO_ROOT / rel_path).read_text()
    assert (
        "aiohttp.ClientSession" not in source
    ), f"{rel_path} should not open its own aiohttp session; use LLMClient."
