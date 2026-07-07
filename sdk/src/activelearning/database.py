"""
Database module for unified SQLite access.

Provides async SQLite operations with proper schema initialization
for all ActiveLearningAI tables.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import aiosqlite

logger = logging.getLogger(__name__)


def default_sqlite_path(db_name: str = "unified.db") -> str:
    """Return the default SQLite path, honoring SQLITE_PATH when set.

    Windows standalone uses a user-writable AppData path; Linux/macOS keep
    the Docker-oriented /data/sqlite default when SQLITE_PATH is unset.
    """
    if env_path := os.environ.get("SQLITE_PATH"):
        return env_path
    if sys.platform == "win32":
        local = os.environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"),
        )
        return str(Path(local) / "Engram" / "sqlite" / db_name)
    return f"/data/sqlite/{db_name}"


def _load_schema() -> str:
    """Load SQL schema from external file."""
    schema_path = Path(__file__).parent / "schema.sql"
    if schema_path.exists():
        return schema_path.read_text()
    else:
        logger.warning(f"Schema file not found at {schema_path}, using minimal schema")
        # Minimal fallback schema
        return """
        CREATE TABLE IF NOT EXISTS audit_entries (
            id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            component TEXT,
            action TEXT,
            details TEXT,
            created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000)
        );
        """


SCHEMA_SQL = _load_schema()

# Ordered migrations applied before the (declarative) schema is reapplied. Entry
# i upgrades a database from user_version i to i+1. Because the schema is all
# CREATE ... IF NOT EXISTS, a migration only needs to drop or alter objects whose
# shape changed; executescript then recreates them. This keeps the table DDL in
# schema.sql as the single source of truth.
_MIGRATIONS: list[tuple[str, ...]] = [
    # v1: llm_cache changed shape (added model/tags/cached_at, dropped legacy
    # columns). It is a disposable index over the Qdrant cache, so drop the old
    # table and let the schema recreate it with the current columns.
    ("DROP TABLE IF EXISTS llm_cache",),
]
SCHEMA_VERSION = len(_MIGRATIONS)


class Database:
    """
    Async SQLite database client.

    Provides connection pooling and schema initialization for
    the unified ActiveLearningAI database.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or default_sqlite_path()
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database and create schema."""
        # Ensure directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Connect and create schema
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrency
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")

        # Bring an older database up to date, then (re)create the schema
        await self._migrate()
        await self._connection.executescript(SCHEMA_SQL)
        await self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await self._connection.commit()

        logger.info(f"Database initialized at {self.db_path}")

    async def _migrate(self) -> None:
        """Run pending migrations so existing databases adopt schema changes.

        Tracks progress with PRAGMA user_version; a fresh database starts at 0
        and the pending statements are no-ops on it.
        """
        if not self._connection:
            raise RuntimeError("Database not initialized")
        cursor = await self._connection.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        version = row[0] if row else 0

        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {version} is newer than this build "
                f"supports ({SCHEMA_VERSION}); upgrade the application"
            )

        for statements in _MIGRATIONS[version:]:
            for statement in statements:
                await self._connection.execute(statement)

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    async def execute(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> aiosqlite.Cursor:
        """Execute a SQL statement."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")
        return await self._connection.execute(sql, params)

    async def executemany(
        self,
        sql: str,
        params_list: list[tuple[Any, ...]],
    ) -> aiosqlite.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")
        return await self._connection.executemany(sql, params_list)

    async def fetchone(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> aiosqlite.Row | None:
        """Execute and fetch one result."""
        cursor = await self.execute(sql, params)
        return await cursor.fetchone()

    async def fetchall(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[aiosqlite.Row]:
        """Execute and fetch all results."""
        cursor = await self.execute(sql, params)
        return cast(list[Any], await cursor.fetchall())

    async def commit(self) -> None:
        """Commit the current transaction."""
        if self._connection:
            await self._connection.commit()

    async def insert(
        self,
        table: str,
        data: dict[str, Any],
    ) -> str:
        """
        Insert a row into a table.

        Args:
            table: Table name
            data: Column-value mapping

        Returns:
            The ID of the inserted row
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        await self.execute(sql, tuple(data.values()))
        await self.commit()
        return cast(str, data.get("id", ""))

    async def update(
        self,
        table: str,
        data: dict[str, Any],
        where: str,
        where_params: tuple[Any, ...] = (),
    ) -> int:
        """
        Update rows in a table.

        Args:
            table: Table name
            data: Column-value mapping
            where: WHERE clause
            where_params: Parameters for WHERE clause

        Returns:
            Number of rows updated
        """
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        cursor = await self.execute(sql, tuple(data.values()) + where_params)
        await self.commit()
        return cursor.rowcount


# Global database instance
_db: Database | None = None


async def get_database() -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database()
        await _db.initialize()
    return _db
