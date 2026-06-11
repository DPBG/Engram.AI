"""
Database module for unified SQLite access.

Provides async SQLite operations with proper schema initialization
for all ActiveLearningAI tables.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)


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


class Database:
    """
    Async SQLite database client.

    Provides connection pooling and schema initialization for
    the unified ActiveLearningAI database.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or os.environ.get(
            "SQLITE_PATH", "/data/sqlite/unified.db"
        )
        self._connection: Optional[aiosqlite.Connection] = None

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

        # Create schema
        await self._connection.executescript(SCHEMA_SQL)
        await self._connection.commit()

        logger.info(f"Database initialized at {self.db_path}")

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
        if not self._connection:
            raise RuntimeError("Database not initialized")
        return await self._connection.execute(sql, params)

    async def executemany(
        self,
        sql: str,
        params_list: list[tuple[Any, ...]],
    ) -> aiosqlite.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        if not self._connection:
            raise RuntimeError("Database not initialized")
        return await self._connection.executemany(sql, params_list)

    async def fetchone(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> Optional[aiosqlite.Row]:
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
        return await cursor.fetchall()

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
        return data.get("id", "")

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
_db: Optional[Database] = None


async def get_database() -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database()
        await _db.initialize()
    return _db
