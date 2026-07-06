"""Body skill library -- per-manifest motor weight snapshots.

Phase 2 of docs/CROSS-BRAIN-TRANSFER-DESIGN.md. The library is a simple
SQLite key-value store keyed by ``body_manifest_id``. Each entry holds an
opaque blob (the brain's motor cortex weights at the moment of save) plus
metadata.

The library does not know what a "motor weight" is. It stores bytes and
hands them back. The brain serializes / deserializes its own weights via
numpy savez_compressed. This keeps the library body-agnostic: a drone,
walker, or humanoid all use the same store contract.

Iron rule (CLAUDE.md): the library never interprets actuator tag strings,
channel names, or anything body-specific. Manifest id in, blob out.

SCOPE BOUNDARY vs ``persistence.py`` (single source of truth note):
  ``persistence.py`` is the canonical CURRENT-STATE bgsave + restore for
  the running brain. It saves the FULL synapse weight set every save
  cycle and overwrites the previous snapshot. It is the SoT for "what
  state is the brain in right now."

  This module is the canonical BODY-KEYED ARCHIVE. It stores ONLY the
  motor-coupled subset of weights, tagged by ``body_manifest_id``, and
  ACCUMULATES one entry per body the brain has worn. It is the SoT for
  "what motor skills has the brain learned across body changes."

  The two stores never own the same row of bytes. ``persistence.py``
  owns "current"; the library owns "archived per body". On body swap,
  the brain reads current motor weights from memory (which match the
  last bgsave snapshot), writes them HERE tagged by the OLD manifest
  id, then asks HERE for any entry tagged by the NEW manifest id to
  bootstrap motor cortex on the new body. ``persistence.py`` keeps
  rolling forward with new-body weights.

  Do not move weight storage between the two modules. Different scopes,
  different lifetimes.

Lifecycle:
  At brain shutdown (or scheduled save), the brain calls
  ``library.save(current_manifest_id, weights_blob, brain_name)``.
  At brain startup with a body manifest, the brain calls
  ``library.load(current_manifest_id)`` to see if there is a prior skill
  to bootstrap motor cortex with.

For body hot-swap (mid-run body change), the brain saves under the OLD
manifest id, re-allocates motor cortex for the new body, loads under the
NEW manifest id if present.

"Motor weights" = the synapse groups in
``HomeostasisScheduler.motor_groups`` (network.py:226). That frozenset is
the canonical SoT for what is motor-coupled in the brain. Phase 2.b.wire
reads from there, never re-derives.
"""

from __future__ import annotations

import io
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS bodies (
    manifest_id TEXT PRIMARY KEY,
    motor_weights BLOB NOT NULL,
    brain_name TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    weight_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_bodies_updated_at ON bodies(updated_at);
CREATE INDEX IF NOT EXISTS idx_bodies_brain_name ON bodies(brain_name);
"""


@dataclass
class SkillEntry:
    """One stored skill row."""

    manifest_id: str
    brain_name: str
    created_at: float
    updated_at: float
    weight_count: int


class BodySkillLibrary:
    """SQLite-backed per-manifest motor weight store.

    Usage:
        lib = BodySkillLibrary("/data/sqlite/body_skills.db")
        # Save the current motor cortex state when the brain is about to
        # switch bodies or shut down:
        lib.save("walker_4dof_v1", weights_dict, brain_name="oscen-neuromorphic")
        # At brain startup with a (possibly new) body:
        weights = lib.load("walker_4dof_v1")  # None if first time
        if weights is not None:
            self._network.restore_motor_weights(weights)
        lib.close()

    The weights argument to save() is a ``dict[str, np.ndarray]`` mapping
    synapse group name to weight array. The library serializes this with
    numpy.savez_compressed (single npz blob), so any tensor shape works
    and storage is reasonably compact.
    """

    DEFAULT_PATH = "/data/sqlite/body_skills.db"

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.environ.get(
            "NEURO_BODY_SKILL_DB",
            self.DEFAULT_PATH,
        )
        if self._db_path != ":memory:":
            db_dir = os.path.dirname(self._db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def db_path(self) -> str:
        return self._db_path

    # -- Save / load --

    def save(
        self,
        manifest_id: str,
        weights: dict[str, np.ndarray],
        brain_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a motor weights snapshot for ``manifest_id``.

        Overwrites any prior entry for the same manifest_id; only one
        snapshot per body. If the brain wants version history, it can
        bump the manifest_id (e.g. walker_4dof_v2).
        """
        if not manifest_id:
            raise ValueError("manifest_id must be non-empty")
        if not isinstance(weights, dict):
            raise TypeError(f"weights must be dict, got {type(weights)}")
        blob = self._serialize(weights)
        now = time.time()
        import json

        metadata_json = json.dumps(metadata or {})
        self._conn.execute(
            """
            INSERT INTO bodies
              (manifest_id, motor_weights, brain_name,
               created_at, updated_at, weight_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(manifest_id) DO UPDATE SET
              motor_weights = excluded.motor_weights,
              brain_name = excluded.brain_name,
              updated_at = excluded.updated_at,
              weight_count = excluded.weight_count,
              metadata_json = excluded.metadata_json
            """,
            (
                manifest_id,
                blob,
                brain_name,
                now,
                now,
                len(weights),
                metadata_json,
            ),
        )
        self._conn.commit()
        logger.info(
            "BodySkillLibrary: saved %s (%d groups, %d bytes, brain=%s)",
            manifest_id,
            len(weights),
            len(blob),
            brain_name or "(none)",
        )

    def load(self, manifest_id: str) -> dict[str, np.ndarray] | None:
        """Return the stored weights for ``manifest_id`` or None."""
        if not manifest_id:
            return None
        row = self._conn.execute(
            "SELECT motor_weights FROM bodies WHERE manifest_id = ?",
            (manifest_id,),
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row[0])

    def load_actuator_manifest(self, manifest_id: str) -> dict | None:
        """Return the actuator manifest stored alongside the weights for
        ``manifest_id``, or None if no entry or no manifest. Used by
        Phase 3.5 projection at body-swap time.

        Convention: callers pass actuator_manifest in the ``metadata``
        argument to ``save`` under the key ``"actuator_manifest"``.
        """
        if not manifest_id:
            return None
        row = self._conn.execute(
            "SELECT metadata_json FROM bodies WHERE manifest_id = ?",
            (manifest_id,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        import json

        try:
            metadata = json.loads(row[0])
        except (ValueError, TypeError):
            return None
        return metadata.get("actuator_manifest")

    def has_skill(self, manifest_id: str) -> bool:
        if not manifest_id:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM bodies WHERE manifest_id = ? LIMIT 1",
            (manifest_id,),
        ).fetchone()
        return row is not None

    def delete(self, manifest_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM bodies WHERE manifest_id = ?",
            (manifest_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_skills(self) -> list[SkillEntry]:
        rows = self._conn.execute(
            """
            SELECT manifest_id, brain_name, created_at, updated_at, weight_count
            FROM bodies
            ORDER BY updated_at DESC
            """,
        ).fetchall()
        return [
            SkillEntry(
                manifest_id=r[0],
                brain_name=r[1],
                created_at=r[2],
                updated_at=r[3],
                weight_count=r[4],
            )
            for r in rows
        ]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # -- Serialization (numpy compressed npz) --

    @staticmethod
    def _serialize(weights: dict[str, np.ndarray]) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, **weights)
        return buf.getvalue()

    @staticmethod
    def _deserialize(blob: bytes) -> dict[str, np.ndarray]:
        buf = io.BytesIO(blob)
        with np.load(buf) as npz:
            return {k: npz[k] for k in npz.files}


__all__ = ["BodySkillLibrary", "SkillEntry"]
