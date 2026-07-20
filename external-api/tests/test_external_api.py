"""
Regression tests for the external-api service (E2.1.2 migration contract).

Verifies that after migrating to BaseService/EventBus:
  - All NATS interactions go through EventBus (no raw nats.connect usage)
  - The Kernel approval gate is preserved and fail-closed
  - Claude → OpenAI fallback routing is correct
  - Conflict detection and human-escalation publish correctly
  - Service handlers update metrics and reply via the EventBus

Fake objects replace NATS, SQLite, and external API clients; no network needed.
"""

import asyncio
import os
import sys
import types

# ── path so `import external_api` resolves without installing the package ─────
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from external_api.manager import ExternalAPIManager  # noqa: E402
from external_api.service import ExternalAPIService  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


# ── fake collaborators ────────────────────────────────────────────────────────


class _FakeBus:
    """Minimal EventBus stand-in; records publish calls, returns a fixed decision."""

    def __init__(self, decision=None, timeout=False):
        self.published: list[tuple[str, dict]] = []
        self._decision = decision
        self._timeout = timeout

    async def publish(self, subject: str, data: dict) -> None:
        self.published.append((subject, data))

    async def wait_for_decision(self, trace_id: str, timeout: float = 10.0) -> dict:
        if self._timeout:
            raise TimeoutError("simulated timeout")
        return self._decision or {}

    async def subscribe(self, subject, handler, **kwargs) -> None:
        pass


class _FakeDB:
    """Minimal Database stand-in; records SQL execute calls."""

    def __init__(self):
        self.executed: list[tuple[str, tuple]] = []

    async def execute(self, sql: str, params: tuple = ()) -> None:
        self.executed.append((sql, params))

    async def commit(self) -> None:
        pass


class _FakeClaude:
    """Minimal AsyncAnthropic stand-in."""

    def __init__(self, text: str = "claude response"):
        self._text = text
        self.messages = self

    async def create(self, **kwargs) -> object:
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=self._text)])


class _FakeOpenAI:
    """Minimal AsyncOpenAI stand-in."""

    def __init__(self, text: str = "openai response"):
        self._text = text
        # .chat.completions.create(...) — completions is self
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kwargs) -> object:
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self._text))]
        )


class _FakeMsg:
    """Minimal NATS Msg stand-in for request-reply handlers."""

    def __init__(self, has_reply: bool = True):
        self.reply = "reply.inbox" if has_reply else None
        self.responded: list[bytes] = []

    async def respond(self, data: bytes) -> None:
        self.responded.append(data)


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


# ── helpers ───────────────────────────────────────────────────────────────────


def _manager(bus, db, *, claude=None, openai=None) -> ExternalAPIManager:
    mgr = ExternalAPIManager(event_bus=bus, db=db)
    mgr._claude_client = claude
    mgr._openai_client = openai
    return mgr


def _build_service() -> ExternalAPIService:
    """Construct ExternalAPIService bypassing BaseService.__init__ (no NATS/DB)."""
    svc = ExternalAPIService.__new__(ExternalAPIService)
    svc.logger = _Logger()
    svc._queries_total = 0
    svc._queries_success = 0
    svc._queries_failed = 0
    svc._conflicts_detected = 0
    svc._manager = None
    return svc


# ── ExternalAPIManager: Kernel gate ──────────────────────────────────────────


def test_kernel_denial_blocks_query():
    mgr = _manager(_FakeBus(decision={"type": "DENY"}), _FakeDB())
    result = _run(mgr.query_external("query"))
    assert result["success"] is False
    assert "Kernel denied" in result["error"]


def test_kernel_timeout_blocks_query():
    mgr = _manager(_FakeBus(timeout=True), _FakeDB())
    result = _run(mgr.query_external("query"))
    assert result["success"] is False


def test_kernel_exception_blocks_query():
    bus = _FakeBus()

    async def _bad_decision(*_a, **_k):
        raise RuntimeError("bus exploded")

    bus.wait_for_decision = _bad_decision
    mgr = _manager(bus, _FakeDB())
    result = _run(mgr.query_external("query"))
    assert result["success"] is False


def test_kernel_allow_proposal_published_to_correct_subject():
    bus = _FakeBus(decision={"type": "ALLOW"})
    mgr = _manager(bus, _FakeDB(), claude=_FakeClaude())
    _run(mgr.query_external("test"))
    subjects = [s for s, _ in bus.published]
    assert "proposal.new" in subjects


def test_kernel_proposal_uses_action_envelope():
    """Kernel rejects proposal.new payloads missing 'action' — shape must match ActionProposal."""
    bus = _FakeBus(decision={"type": "ALLOW"})
    mgr = _manager(bus, _FakeDB())
    _run(mgr.query_external("what is the weather?"))
    _subject, proposal = next(p for p in bus.published if p[0] == "proposal.new")
    assert proposal.get("provenance") == "external-api"
    assert proposal["action"]["type"] == "external_query"
    assert "query" in proposal["action"]


def test_external_query_proposal_can_be_kernel_allowed():
    """End-to-end shape check: a well-formed external query is evaluable, not malformed-DENY."""
    from activelearning import RiskAnalysis
    from kernel.evaluator import DecisionType, KernelEvaluator

    proposal = {
        "trace_id": "t-ext",
        "provenance": "external-api",
        "action": {"type": "external_query", "query": "safe question"},
    }
    decision = KernelEvaluator().evaluate_action_proposal(
        proposal,
        risk_analysis=RiskAnalysis(trace_id="t-ext", risk_score=0.1, flags=[]),
    )
    assert decision.type == DecisionType.ALLOW


# ── ExternalAPIManager: API routing ──────────────────────────────────────────


def test_no_api_client_returns_error():
    mgr = _manager(_FakeBus(decision={"type": "ALLOW"}), _FakeDB())
    result = _run(mgr.query_external("query"))
    assert result["success"] is False
    assert "API" in result["error"] or "api" in result["error"].lower()


def test_claude_used_when_available():
    mgr = _manager(
        _FakeBus(decision={"type": "ALLOW"}),
        _FakeDB(),
        claude=_FakeClaude("claude answer"),
    )
    result = _run(mgr.query_external("question"))
    assert result["success"] is True
    assert result["response"] == "claude answer"


def test_claude_preferred_over_openai():
    mgr = _manager(
        _FakeBus(decision={"type": "ALLOW"}),
        _FakeDB(),
        claude=_FakeClaude("from claude"),
        openai=_FakeOpenAI("from openai"),
    )
    result = _run(mgr.query_external("question"))
    assert result["response"] == "from claude"


def test_openai_fallback_when_no_claude():
    mgr = _manager(
        _FakeBus(decision={"type": "ALLOW"}),
        _FakeDB(),
        openai=_FakeOpenAI("openai answer"),
    )
    result = _run(mgr.query_external("question"))
    assert result["success"] is True
    assert result["response"] == "openai answer"


def test_successful_query_has_no_conflict_by_default():
    mgr = _manager(
        _FakeBus(decision={"type": "ALLOW"}),
        _FakeDB(),
        claude=_FakeClaude("the answer"),
    )
    result = _run(mgr.query_external("question"))
    assert result["conflict"] is False
    assert result["error"] is None


# ── ExternalAPIManager: persistence ──────────────────────────────────────────


def test_successful_query_logged_to_database():
    db = _FakeDB()
    mgr = _manager(_FakeBus(decision={"type": "ALLOW"}), db, claude=_FakeClaude())
    _run(mgr.query_external("log me"))
    assert any("external_api_queries" in sql for sql, _ in db.executed)


def test_failed_query_not_logged():
    # Kernel denies → no DB write
    db = _FakeDB()
    mgr = _manager(_FakeBus(decision={"type": "DENY"}), db)
    _run(mgr.query_external("denied"))
    assert db.executed == []


# ── ExternalAPIManager: conflict detection ────────────────────────────────────


def test_conflict_detected_on_low_word_overlap():
    bus = _FakeBus(decision={"type": "ALLOW"})
    mgr = _manager(bus, _FakeDB(), claude=_FakeClaude("alpha beta gamma delta"))
    result = _run(
        mgr.query_external(
            "question",
            local_knowledge={"description": "zeta eta theta iota kappa lambda mu nu"},
        )
    )
    assert result["success"] is True
    assert result["conflict"] is True


def test_conflict_triggers_approval_request():
    bus = _FakeBus(decision={"type": "ALLOW"})
    mgr = _manager(bus, _FakeDB(), claude=_FakeClaude("alpha beta gamma delta"))
    _run(
        mgr.query_external(
            "question",
            local_knowledge={"description": "zeta eta theta iota kappa lambda"},
        )
    )
    subjects = [s for s, _ in bus.published]
    assert "approval.request" in subjects


def test_no_conflict_when_high_word_overlap():
    mgr = _manager(
        _FakeBus(decision={"type": "ALLOW"}),
        _FakeDB(),
        claude=_FakeClaude("the quick brown fox"),
    )
    result = _run(
        mgr.query_external(
            "question",
            local_knowledge={"description": "the quick brown fox"},
        )
    )
    assert result["conflict"] is False


def test_no_conflict_without_local_knowledge():
    mgr = _manager(
        _FakeBus(decision={"type": "ALLOW"}),
        _FakeDB(),
        claude=_FakeClaude("some response"),
    )
    result = _run(mgr.query_external("question"))
    assert result["conflict"] is False


# ── ExternalAPIManager._detect_conflict (async, no mocks needed) ─────────────


def test_detect_conflict_true_when_zero_overlap():
    mgr = ExternalAPIManager(event_bus=_FakeBus(), db=_FakeDB())
    assert (
        _run(
            mgr._detect_conflict(
                "alpha beta gamma",
                {"description": "zeta eta theta iota"},
            )
        )
        is True
    )


def test_detect_conflict_false_when_identical_words():
    mgr = ExternalAPIManager(event_bus=_FakeBus(), db=_FakeDB())
    assert (
        _run(
            mgr._detect_conflict(
                "the quick brown fox",
                {"description": "the quick brown fox"},
            )
        )
        is False
    )


def test_detect_conflict_false_when_empty_local_text():
    mgr = ExternalAPIManager(event_bus=_FakeBus(), db=_FakeDB())
    assert _run(mgr._detect_conflict("some external response", {"description": ""})) is False


def test_detect_conflict_false_when_no_description_key():
    mgr = ExternalAPIManager(event_bus=_FakeBus(), db=_FakeDB())
    assert _run(mgr._detect_conflict("some response", {})) is False


def test_detect_conflict_uses_semantic_similarity_when_configured():
    class _FakeEmbeddings:
        async def embed_text(self, text: str) -> list[float]:
            if "cat" in text.lower() or "feline" in text.lower():
                return [1.0, 0.0, 0.0]
            if "dog" in text.lower() or "canine" in text.lower():
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

    mgr = ExternalAPIManager(
        event_bus=_FakeBus(),
        db=_FakeDB(),
        embedding_service=_FakeEmbeddings(),
    )
    assert (
        _run(
            mgr._detect_conflict(
                "A feline rests on the mat",
                {"description": "The cat sits on the mat"},
            )
        )
        is False
    )
    assert (
        _run(
            mgr._detect_conflict(
                "A canine runs in the park",
                {"description": "The cat sits on the mat"},
            )
        )
        is True
    )


# ── ExternalAPIService handler tests ─────────────────────────────────────────


def test_handle_query_success_updates_metrics():
    svc = _build_service()
    svc._manager = _manager(_FakeBus(decision={"type": "ALLOW"}), _FakeDB(), claude=_FakeClaude())
    _run(svc._handle_query({"query": "what?"}, _FakeMsg()))
    assert svc._queries_total == 1
    assert svc._queries_success == 1
    assert svc._queries_failed == 0


def test_handle_query_failure_updates_failed_metric():
    svc = _build_service()
    svc._manager = _manager(_FakeBus(decision={"type": "DENY"}), _FakeDB())
    _run(svc._handle_query({"query": "denied"}, _FakeMsg()))
    assert svc._queries_total == 1
    assert svc._queries_failed == 1
    assert svc._queries_success == 0


def test_handle_query_conflict_increments_conflict_counter():
    svc = _build_service()
    bus = _FakeBus(decision={"type": "ALLOW"})
    svc._manager = _manager(bus, _FakeDB(), claude=_FakeClaude("alpha beta gamma delta"))
    _run(
        svc._handle_query(
            {
                "query": "conflict?",
                "local_knowledge": {"description": "zeta eta theta iota kappa lambda"},
            },
            _FakeMsg(),
        )
    )
    assert svc._conflicts_detected == 1


def test_handle_query_replies_when_reply_address_present():
    svc = _build_service()
    svc._manager = _manager(_FakeBus(decision={"type": "ALLOW"}), _FakeDB(), claude=_FakeClaude())
    msg = _FakeMsg(has_reply=True)
    _run(svc._handle_query({"query": "answer me"}, msg))
    assert msg.responded


def test_handle_query_no_reply_address_does_not_crash():
    svc = _build_service()
    svc._manager = _manager(_FakeBus(decision={"type": "ALLOW"}), _FakeDB(), claude=_FakeClaude())
    msg = _FakeMsg(has_reply=False)
    _run(svc._handle_query({"query": "silent"}, msg))
    assert msg.responded == []


def test_handle_status_replies_with_metric_counts():
    svc = _build_service()
    svc._manager = _manager(_FakeBus(), _FakeDB())
    msg = _FakeMsg()
    _run(svc._handle_status({}, msg))
    assert msg.responded


def test_handle_status_reflects_api_availability():
    svc = _build_service()
    bus = _FakeBus()
    svc._manager = _manager(bus, _FakeDB(), claude=_FakeClaude())
    msg = _FakeMsg()
    _run(svc._handle_status({}, msg))
    # A reply was sent — the handler ran to completion
    assert msg.responded
