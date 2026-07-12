"""Fail-closed regression tests for issue #205 (M1.12).

``BeliefsService._load_from_db`` used to wrap the entire node/edge load in a
bare ``except Exception: log warning, continue``, then fall straight through
to ``seed_constitutional_beliefs()`` — corruption got silently treated as
"first boot" and constitutional VALUEs got reseeded from hardcoded defaults
rather than halting. This directly contradicts CLAUDE.md §3's fail-closed
rule ("if a check cannot run, deny/halt — never degrade open").

These tests stub ``self.database`` rather than corrupting a real SQLite
file: reliably reproducing "database disk image is malformed" via raw byte
corruption turns out to be non-deterministic (SQLite recovers from a lot of
naive corruption attempts), while a stub deterministically reproduces the
exact code path this issue is about — ``fetchall()`` raising — the same
style ``kernel/tests/test_service.py`` already uses for the Kernel's
analogous fail-closed-on-internal-error coverage.

Uses ``asyncio.run()`` rather than ``@pytest.mark.asyncio`` since the
Governance CI job runs ``beliefs/tests`` with ``--with pytest --with
networkx`` only — no pytest-asyncio installed in that step.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# beliefs/src on path so `import beliefs.*` resolves without installing,
# matching test_belief_floor.py's convention.
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from beliefs.graph import BeliefGraph  # noqa: E402
from beliefs.service import BeliefsService, BeliefStoreCorruptedError  # noqa: E402


class _FakeDatabaseEmpty:
    """A legitimately empty database — first boot, not corruption."""

    async def fetchall(self, query: str, *args, **kwargs) -> list:
        return []


class _FakeDatabaseCorrupted:
    """Present but unreadable — simulates a corrupted SQLite store.

    "database disk image is malformed" is SQLite's real message for page
    -level corruption; the exact exception type doesn't matter to
    _load_from_db (it catches broadly), only that it's raised instead of a
    successful empty result.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def fetchall(self, query: str, *args, **kwargs) -> list:
        self.calls += 1
        raise Exception("database disk image is malformed")


class _SpySaveDatabase:
    """Tracks whether a save (DELETE + re-insert) was ever attempted."""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.inserted: list[tuple] = []

    async def execute(self, query: str, *args, **kwargs) -> None:
        self.executed.append(query)

    async def insert(self, table: str, values: dict) -> None:
        self.inserted.append((table, values))


def _make_service(database) -> BeliefsService:
    """BeliefsService with __init__ bypassed (no real NATS/SQLite needed),
    matching kernel/tests/test_service.py's convention."""
    svc = BeliefsService.__new__(BeliefsService)
    svc.logger = logging.getLogger("test-beliefs")
    svc._graph = BeliefGraph()
    svc._store_corrupted = False
    svc.database = database
    return svc


# ── _load_from_db: empty vs corrupted ──────────────────────────────────────


def test_empty_database_is_first_boot_not_corruption():
    svc = _make_service(_FakeDatabaseEmpty())
    asyncio.run(svc._load_from_db())
    assert svc._store_corrupted is False
    assert svc._graph.node_count == 0


def test_corrupted_database_raises_and_marks_corrupted():
    svc = _make_service(_FakeDatabaseCorrupted())
    try:
        asyncio.run(svc._load_from_db())
        raise AssertionError("expected BeliefStoreCorruptedError")
    except BeliefStoreCorruptedError:
        pass
    assert svc._store_corrupted is True
    assert svc._graph.node_count == 0


# ── _setup: the core regression ────────────────────────────────────────────


def test_setup_aborts_before_reseeding_on_corruption():
    """Corruption must NOT fall through to seed_constitutional_beliefs()
    and silently reseed constitutional VALUEs as if this were a fresh boot
    — the exact bug this issue fixes."""
    svc = _make_service(_FakeDatabaseCorrupted())
    svc.event_bus = None  # _setup must abort before ever touching this

    try:
        asyncio.run(svc._setup())
        raise AssertionError("expected BeliefStoreCorruptedError")
    except BeliefStoreCorruptedError:
        pass

    assert svc._graph.node_count == 0, (
        "seed_constitutional_beliefs() ran despite corruption — VALUEs were "
        "silently reseeded from hardcoded defaults instead of failing closed"
    )


# ── _cleanup: don't wipe a corrupted store on the way down ────────────────


def test_cleanup_skips_save_when_corrupted():
    """run() calls stop()/_cleanup() in a finally block even when _setup()
    raised. _save_to_db() unconditionally DELETEs then re-inserts — saving
    the (empty) in-memory graph here would destroy whatever was still
    recoverable in the corrupted store, on top of the boot failure itself."""
    db = _SpySaveDatabase()
    svc = _make_service(db)
    svc._store_corrupted = True

    asyncio.run(svc._cleanup())

    assert db.executed == [], "DELETE ran against a corrupted store on shutdown"
    assert db.inserted == []


def test_cleanup_saves_normally_when_not_corrupted():
    """Companion sanity check: normal shutdown still saves as before."""
    db = _SpySaveDatabase()
    svc = _make_service(db)
    svc._store_corrupted = False

    asyncio.run(svc._cleanup())

    assert any("DELETE FROM belief_edges" in q for q in db.executed)
    assert any("DELETE FROM belief_nodes" in q for q in db.executed)
