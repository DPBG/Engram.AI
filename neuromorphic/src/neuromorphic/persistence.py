"""Save/load neural state — hybrid SQLite + filesystem.

Metadata, neuron states, and small data go in SQLite.
Synapse arrays (weights, eligibility, etc.) are saved as .npy files on disk
for speed — no blob size limits and direct numpy save/load.

Background saves use ``os.fork()`` (Redis BGSAVE pattern): the child process
inherits the full address space via copy-on-write and writes to disk while
the parent continues training with near-zero pause.

Layout::

    /data/sqlite/neuromorphic.db         <- SQLite (metadata, neurons, drives)
    /data/sqlite/synapses/               <- numpy .npy files per synapse group
      sensory_association/
        weights_data.npy
        weights_indices.npy
        ...
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import signal
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
import aiosqlite

logger = logging.getLogger(__name__)

# Keys saved/loaded from the network_aux JSON blob.
# Defined once to avoid divergence between sync and async save paths.
_AUX_KEYS = (
    "sensory_allocator", "sensory_buffer",
    "cached_myel_fraction", "feature_stdp_peak",
    "speech_decoder", "cognitive_decoder",
    "watchdog",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS neuron_state (
    region TEXT PRIMARY KEY,
    state BLOB,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS synapse_meta (
    name TEXT PRIMARY KEY,
    shape_rows INTEGER,
    shape_cols INTEGER,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS drive_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS network_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    step_count INTEGER,
    config_json TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS neuromodulation_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS auditory_stm_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS network_aux (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT,
    updated_at REAL
);
"""

# Fields saved per synapse group -> (filename, state-dict key, numpy dtype)
_SYNAPSE_FIELDS = [
    ("weights_data", "weights_data", np.float32),
    ("weights_indices", "weights_indices", np.int32),
    ("weights_indptr", "weights_indptr", np.int32),
    ("eligibility", "eligibility", np.float32),
    ("bcm_theta", "bcm_theta", np.float32),
    ("stability_counter", "stability_counter", np.int32),
    ("myelinated", "myelinated", np.bool_),
    ("identity_tag", "identity", np.bool_),
    ("prune_survival", "prune_survival_count", np.int32),
    ("elig_active", "elig_active", np.int32),
]


# ------------------------------------------------------------------ #
#  Child-process save (runs after fork — NO asyncio, NO parent fds)  #
# ------------------------------------------------------------------ #

def _child_save(db_path: str, syn_dir: str, state: dict[str, Any]) -> None:
    """Write state to disk from a forked child process.

    This function runs in a forked child.  It must NOT touch:
    - The parent's asyncio event loop
    - The parent's aiosqlite connection (file descriptors are shared)
    - Any asyncio primitives

    It opens its own *synchronous* sqlite3 connection and writes .npy
    files using atomic rename (write to .tmp, then rename).
    """
    # Ignore SIGTERM so we finish writing even if Docker sends stop signal.
    # The parent handles SIGTERM and will waitpid() for us.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    t0 = time.time()
    step = state.get("step_count", 0)
    now = t0

    # ---- Synapse arrays -> .npy files (atomic write) ---- #
    syn_base = Path(syn_dir)
    for syn_name, sstate in state.get("synapses", {}).items():
        group_dir = syn_base / syn_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for file_key, state_key, dtype in _SYNAPSE_FIELDS:
            arr = sstate.get(state_key)
            fpath = group_dir / f"{file_key}.npy"
            # np.save keeps .npy suffix as-is, so _tmp.npy won't get doubled
            tmp = group_dir / f"{file_key}_tmp.npy"
            if arr is not None:
                np.save(tmp, arr)
                os.replace(tmp, fpath)  # atomic on POSIX
            elif fpath.exists():
                fpath.unlink()

    # ---- Metadata -> SQLite (fresh synchronous connection) ---- #
    db = sqlite3.connect(db_path)
    try:
        db.execute("BEGIN IMMEDIATE")

        db.execute(
            "INSERT OR REPLACE INTO network_meta (id, step_count, updated_at) "
            "VALUES (1, ?, ?)", (step, now),
        )

        drives = state.get("drives", {})
        db.execute(
            "INSERT OR REPLACE INTO drive_state (id, state, updated_at) "
            "VALUES (1, ?, ?)", (json.dumps(drives), now),
        )

        for region_name, rstate in state.get("regions", {}).items():
            buf = io.BytesIO()
            np.savez_compressed(buf, **rstate)
            db.execute(
                "INSERT OR REPLACE INTO neuron_state (region, state, updated_at) "
                "VALUES (?, ?, ?)", (region_name, buf.getvalue(), now),
            )

        neuromod = state.get("neuromodulation", {})
        if neuromod:
            db.execute(
                "INSERT OR REPLACE INTO neuromodulation_state (id, state, updated_at) "
                "VALUES (1, ?, ?)", (json.dumps(neuromod), now),
            )

        # Save AuditorySTM state (tiny JSON — ~2 KB)
        auditory_stm = state.get("auditory_stm")
        if auditory_stm:
            db.execute(
                "INSERT OR REPLACE INTO auditory_stm_state (id, state, updated_at) "
                "VALUES (1, ?, ?)", (json.dumps(auditory_stm), now),
            )

        # Save auxiliary state (small but critical for deterministic restart).
        aux: dict[str, Any] = {}
        for key in _AUX_KEYS:
            val = state.get(key)
            if val is not None:
                aux[key] = val
        if aux:
            db.execute(
                "INSERT OR REPLACE INTO network_aux (id, state, updated_at) "
                "VALUES (1, ?, ?)", (json.dumps(aux), now),
            )

        for syn_name, sstate in state.get("synapses", {}).items():
            shape = sstate["shape"]
            db.execute(
                "INSERT OR REPLACE INTO synapse_meta "
                "(name, shape_rows, shape_cols, updated_at) VALUES (?, ?, ?, ?)",
                (syn_name, shape[0], shape[1], now),
            )

        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    finally:
        db.close()

    elapsed = time.time() - t0
    # Write to stderr directly — logger may not be safe after fork
    os.write(2, f"[bgsave-child] step {step} saved in {elapsed:.1f}s\n".encode())


# ------------------------------------------------------------------ #
#  Main persistence class                                            #
# ------------------------------------------------------------------ #

class NeuromorphicPersistence:
    """Saves and loads neuromorphic network state to/from SQLite + disk.

    Supports two save modes:

    * ``save_state(state)`` — synchronous (blocking) save, used for
      final save on shutdown and for small-scale testing.
    * ``save_background(state)`` — fork-based background save.  The
      parent returns immediately; a child process writes the data.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._syn_dir = Path(db_path).parent / "synapses"

        # Background save child PID (0 = no child running)
        self._bg_child_pid: int = 0

    async def open(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        self._syn_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Persistence opened: {self.db_path}")

    async def close(self) -> None:
        # Wait for any background save to finish before closing
        await self.wait_bg_save()
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------ #
    #  Background (fork-based) save                                      #
    # ------------------------------------------------------------------ #

    def is_bg_save_running(self) -> bool:
        """Check if a background save child is still running."""
        if self._bg_child_pid == 0:
            return False
        # Non-blocking waitpid: reap if finished, else check alive
        try:
            pid, status = os.waitpid(self._bg_child_pid, os.WNOHANG)
        except ChildProcessError:
            # Child already reaped or doesn't exist
            self._bg_child_pid = 0
            return False
        if pid != 0:
            # Child finished — reap it
            self._bg_child_pid = 0
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
                logger.error(f"Background save child exited with status {os.WEXITSTATUS(status)}")
            elif os.WIFSIGNALED(status):
                logger.error(f"Background save child killed by signal {os.WTERMSIG(status)}")
            return False
        return True  # Still running

    def _reap_zombie(self) -> None:
        """Reap any finished child process to prevent zombie accumulation.

        Called proactively before each save attempt.  is_bg_save_running()
        reaps the child too, but only when _bg_child_pid is set.  This
        method handles the edge case where a child PID was cleared but
        the process wasn't fully reaped by the OS yet.
        """
        if self._bg_child_pid == 0:
            return
        try:
            pid, status = os.waitpid(self._bg_child_pid, os.WNOHANG)
        except ChildProcessError:
            self._bg_child_pid = 0
            return
        if pid != 0:
            self._bg_child_pid = 0
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
                logger.error(f"Reaped zombie save child (exit status {os.WEXITSTATUS(status)})")
            elif os.WIFSIGNALED(status):
                logger.error(f"Reaped zombie save child (killed by signal {os.WTERMSIG(status)})")
            else:
                logger.debug(f"Reaped finished save child pid={pid}")

    async def wait_bg_save(self, timeout: float = 600) -> None:
        """Wait for background save child to finish (up to timeout seconds)."""
        if self._bg_child_pid == 0:
            return
        deadline = time.monotonic() + timeout
        while self.is_bg_save_running():
            if time.monotonic() > deadline:
                logger.warning(f"Background save child {self._bg_child_pid} timed out, killing")
                try:
                    os.kill(self._bg_child_pid, signal.SIGKILL)
                    os.waitpid(self._bg_child_pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
                self._bg_child_pid = 0
                return
            await asyncio.sleep(1.0)

    def save_background(self, state: dict[str, Any]) -> bool:
        """Fork a child process to save state in the background.

        The caller must hold no asyncio locks when calling this — the
        child inherits the parent's memory via copy-on-write, so the
        state dict (which references numpy arrays in the network) is
        visible to the child without any explicit copy.

        Returns True if the fork succeeded, False if a previous save
        is still running (caller should skip or wait).
        """
        # Reap any finished child to prevent zombie accumulation.
        # Even if is_bg_save_running() returns False, try reaping —
        # a child may have exited between saves without being reaped.
        self._reap_zombie()

        if self.is_bg_save_running():
            logger.warning("Previous background save still running, skipping this save")
            return False

        pid = os.fork()
        if pid == 0:
            # ---- CHILD PROCESS ---- #
            # Close inherited file descriptors we don't need.
            # CRITICAL: Do NOT touch self._db (aiosqlite) — it shares
            # the parent's fd and event loop.  We open a fresh sqlite3.
            try:
                _child_save(self.db_path, str(self._syn_dir), state)
            except BaseException as exc:
                os.write(2, f"[bgsave-child] FAILED: {exc}\n".encode())
                os._exit(1)
            os._exit(0)
            # NEVER REACHED — os._exit kills the child immediately.
            # No finally blocks, no atexit handlers, no asyncio cleanup.
        else:
            # ---- PARENT PROCESS ---- #
            self._bg_child_pid = pid
            logger.info(f"Background save started (child pid={pid}, step={state.get('step_count', '?')})")
            return True

    # ------------------------------------------------------------------ #
    #  Synchronous save (used for final save on shutdown / tests)        #
    # ------------------------------------------------------------------ #

    async def save_state(self, state: dict[str, Any]) -> None:
        """Save full network state synchronously (blocks until complete)."""
        if not self._db:
            raise RuntimeError("Persistence not open")

        t0 = time.time()
        now = t0
        step = state.get("step_count", 0)

        # Save synapse arrays to disk (in thread pool for I/O)
        loop = asyncio.get_running_loop()
        syn_futures = []
        for syn_name, sstate in state.get("synapses", {}).items():
            fut = loop.run_in_executor(
                None, self._save_synapse_arrays, syn_name, sstate
            )
            syn_futures.append((syn_name, sstate, fut))

        # Save small data to SQLite while synapses write
        try:
            await self._db.execute("BEGIN IMMEDIATE")

            await self._db.execute(
                "INSERT OR REPLACE INTO network_meta (id, step_count, updated_at) "
                "VALUES (1, ?, ?)", (step, now),
            )

            drives = state.get("drives", {})
            await self._db.execute(
                "INSERT OR REPLACE INTO drive_state (id, state, updated_at) "
                "VALUES (1, ?, ?)", (json.dumps(drives), now),
            )

            for region_name, rstate in state.get("regions", {}).items():
                buf = io.BytesIO()
                np.savez_compressed(buf, **rstate)
                await self._db.execute(
                    "INSERT OR REPLACE INTO neuron_state (region, state, updated_at) "
                    "VALUES (?, ?, ?)", (region_name, buf.getvalue(), now),
                )

            neuromod = state.get("neuromodulation", {})
            if neuromod:
                await self._db.execute(
                    "INSERT OR REPLACE INTO neuromodulation_state (id, state, updated_at) "
                    "VALUES (1, ?, ?)", (json.dumps(neuromod), now),
                )

            auditory_stm = state.get("auditory_stm")
            if auditory_stm:
                await self._db.execute(
                    "INSERT OR REPLACE INTO auditory_stm_state (id, state, updated_at) "
                    "VALUES (1, ?, ?)", (json.dumps(auditory_stm), now),
                )

            # Save auxiliary state (small but critical for deterministic restart).
            aux: dict[str, Any] = {}
            for key in _AUX_KEYS:
                val = state.get(key)
                if val is not None:
                    aux[key] = val
            if aux:
                await self._db.execute(
                    "INSERT OR REPLACE INTO network_aux (id, state, updated_at) "
                    "VALUES (1, ?, ?)", (json.dumps(aux), now),
                )

            for syn_name, sstate, fut in syn_futures:
                await fut
                shape = sstate["shape"]
                await self._db.execute(
                    "INSERT OR REPLACE INTO synapse_meta "
                    "(name, shape_rows, shape_cols, updated_at) VALUES (?, ?, ?, ?)",
                    (syn_name, shape[0], shape[1], now),
                )

            await self._db.execute("COMMIT")
            elapsed = time.time() - t0
            logger.info(f"State saved: step {step} ({elapsed:.1f}s)")
        except BaseException:
            try:
                await self._db.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def _save_synapse_arrays(self, syn_name: str, sstate: dict[str, Any]) -> None:
        """Save synapse arrays as .npy files with atomic rename."""
        group_dir = self._syn_dir / syn_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for file_key, state_key, dtype in _SYNAPSE_FIELDS:
            arr = sstate.get(state_key)
            fpath = group_dir / f"{file_key}.npy"
            tmp = group_dir / f"{file_key}_tmp.npy"
            if arr is not None:
                np.save(tmp, arr)
                os.replace(tmp, fpath)  # atomic on POSIX
            elif fpath.exists():
                fpath.unlink()

    # ------------------------------------------------------------------ #
    #  Load                                                               #
    # ------------------------------------------------------------------ #

    async def load_state(self) -> dict[str, Any] | None:
        """Load full network state from SQLite + disk."""
        if not self._db:
            raise RuntimeError("Persistence not open")

        async with self._db.execute("SELECT step_count FROM network_meta WHERE id=1") as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            step_count = row[0]

        state: dict[str, Any] = {"step_count": step_count}

        # Load drives
        async with self._db.execute("SELECT state FROM drive_state WHERE id=1") as cursor:
            row = await cursor.fetchone()
            if row:
                state["drives"] = json.loads(row[0])

        # Load neuron states
        state["regions"] = {}
        async with self._db.execute("SELECT region, state FROM neuron_state") as cursor:
            async for row in cursor:
                region_name, blob = row
                buf = io.BytesIO(blob)
                npz = np.load(buf)
                state["regions"][region_name] = dict(npz)

        # Load synapse states from disk
        state["synapses"] = {}
        async with self._db.execute("SELECT name, shape_rows, shape_cols FROM synapse_meta") as cursor:
            async for row in cursor:
                name, nrows, ncols = row
                sstate: dict[str, Any] = {"shape": (nrows, ncols)}

                group_dir = self._syn_dir / name
                if not group_dir.is_dir():
                    logger.warning(f"Synapse dir missing for {name}, skipping")
                    continue

                for file_key, state_key, dtype in _SYNAPSE_FIELDS:
                    fpath = group_dir / f"{file_key}.npy"
                    if fpath.exists():
                        arr = np.load(fpath)
                        sstate[state_key] = arr

                state["synapses"][name] = sstate

        # Load neuromodulation state
        try:
            async with self._db.execute("SELECT state FROM neuromodulation_state WHERE id=1") as cursor:
                row = await cursor.fetchone()
                if row:
                    state["neuromodulation"] = json.loads(row[0])
        except Exception:
            pass

        # Load auditory STM state
        try:
            async with self._db.execute("SELECT state FROM auditory_stm_state WHERE id=1") as cursor:
                row = await cursor.fetchone()
                if row:
                    state["auditory_stm"] = json.loads(row[0])
        except Exception:
            pass

        # Load auxiliary state (sensory_allocator, sensory_buffer, speech_decoder, etc.)
        try:
            async with self._db.execute("SELECT state FROM network_aux WHERE id=1") as cursor:
                row = await cursor.fetchone()
                if row:
                    aux = json.loads(row[0])
                    for key, val in aux.items():
                        state[key] = val
        except Exception:
            pass

        logger.info(f"State loaded: step {step_count}")
        return state

    async def has_saved_state(self) -> bool:
        """Check if there's a saved state."""
        if not self._db:
            return False
        async with self._db.execute("SELECT COUNT(*) FROM network_meta") as cursor:
            row = await cursor.fetchone()
            return row[0] > 0 if row else False
