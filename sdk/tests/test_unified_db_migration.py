"""Regression tests: unified.db schema migration path (issue #262).

Proves that a database with an outdated column set is migrated correctly on
startup — ALTER TABLE statements in _MIGRATIONS apply to existing tables, and
existing rows survive untouched.

Async code is driven through asyncio.run() so pytest-asyncio is not required.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from activelearning.database import _MIGRATIONS, SCHEMA_VERSION, Database

# DDL for kernel_decisions as it existed at schema v1 (before the latency_ms
# column was added in v2).  A deployed instance that never picked up the new
# schema.sql would have exactly this shape.
_KERNEL_DECISIONS_V1 = """
CREATE TABLE IF NOT EXISTS kernel_decisions (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    proposal_type TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    source TEXT,
    reason TEXT,
    risk_score REAL NOT NULL DEFAULT 0.0,
    flags TEXT,
    norm_violations TEXT,
    body_profile TEXT,
    issued_at INTEGER NOT NULL,
    expires_at INTEGER,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000)
);
"""


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def _user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def test_alter_table_migration_adds_column_preserves_rows() -> None:
    """An old-schema DB at v1 gets latency_ms added without losing existing rows.

    This is the core regression for issue #262: CREATE TABLE IF NOT EXISTS
    alone would silently skip the new column on a deployed instance.
    The v1→v2 ALTER TABLE migration in _MIGRATIONS must patch the live table.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Seed: kernel_decisions at v1 schema with one existing decision row.
        with sqlite3.connect(db_path) as seed:
            seed.executescript(_KERNEL_DECISIONS_V1)
            seed.execute(
                "INSERT INTO kernel_decisions "
                "(id, trace_id, proposal_type, decision_type, risk_score, issued_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("row-1", "trace-abc", "action", "ALLOW", 0.1, 1_000_000, 1_000_000),
            )
            seed.execute("PRAGMA user_version = 1")
            seed.commit()

        # Run initialize() — should apply the v1→v2 migration.
        db = Database(db_path)
        asyncio.run(db.initialize())
        asyncio.run(db.close())

        with sqlite3.connect(db_path) as check:
            cols = _column_names(check, "kernel_decisions")
            assert (
                "latency_ms" in cols
            ), f"latency_ms column not added by migration; columns present: {cols}"
            row = check.execute(
                "SELECT id, trace_id, decision_type FROM kernel_decisions WHERE id = 'row-1'"
            ).fetchone()
            assert row is not None, "existing row was deleted during migration"
            assert row[1] == "trace-abc", f"row data corrupted after migration: {row}"
            assert row[2] == "ALLOW", f"row data corrupted after migration: {row}"
            assert _user_version(check) == SCHEMA_VERSION
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_fresh_database_reaches_current_schema_version() -> None:
    """A brand-new DB ends at SCHEMA_VERSION with all tables and columns present."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = Database(db_path)
        asyncio.run(db.initialize())
        asyncio.run(db.close())

        with sqlite3.connect(db_path) as check:
            assert _user_version(check) == SCHEMA_VERSION

            # kernel_decisions has the v2 column.
            cols = _column_names(check, "kernel_decisions")
            assert "latency_ms" in cols

            # llm_cache was dropped (v0→v1) and recreated with the current shape.
            llm_cols = _column_names(check, "llm_cache")
            assert "model" in llm_cols
            assert "tags" in llm_cols
            assert "cached_at" in llm_cols
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_schema_version_equals_migration_count() -> None:
    """SCHEMA_VERSION tracks len(_MIGRATIONS) — the contract that keeps _migrate() correct."""
    assert SCHEMA_VERSION == len(_MIGRATIONS), (
        f"SCHEMA_VERSION ({SCHEMA_VERSION}) does not equal len(_MIGRATIONS) "
        f"({len(_MIGRATIONS)}). Update SCHEMA_VERSION when adding a migration entry."
    )


def test_newer_database_raises_on_open() -> None:
    """A DB from a future build (user_version > SCHEMA_VERSION) is refused, not silently opened."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        with sqlite3.connect(db_path) as seed:
            seed.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
            seed.commit()

        db = Database(db_path)
        try:
            asyncio.run(db.initialize())
            assert False, "expected RuntimeError for newer schema version"
        except RuntimeError as exc:
            assert "newer" in str(exc).lower(), f"unexpected error message: {exc}"
        finally:
            asyncio.run(db.close())
    finally:
        Path(db_path).unlink(missing_ok=True)
