"""
Queue state persistence — saves video training queue to JSON on disk.

Writes on every state change (add, remove, complete, advance).
On gateway restart, checks for saved state and offers to resume.
Completed videos are skipped; the active video restarts from loop 0
(frame-level resume not worth the complexity — extra repetition is
fine for STDP).
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "queue_state.json"


class QueueStateManager:
    """Persists video training queue state to a JSON file."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else DEFAULT_STATE_PATH
        self._dirty = False

    @property
    def path(self) -> Path:
        return self._path

    def save(
        self,
        sessions: dict[str, Any],
        queue: list[str],
        active_id: str | None,
    ) -> None:
        """Save current queue state to disk.

        Args:
            sessions: dict of session_id -> VideoTrainingSession objects.
            queue: ordered list of session_ids in the queue.
            active_id: session_id currently playing (or None).
        """
        entries = []
        # Save active first, then queued, then recently completed
        all_ids = []
        if active_id and active_id in sessions:
            all_ids.append(active_id)
        for sid in queue:
            if sid not in all_ids:
                all_ids.append(sid)
        # Also save completed sessions for history
        for sid, s in sessions.items():
            if sid not in all_ids and s.status in ("completed", "stopped", "error"):
                all_ids.append(sid)

        for sid in all_ids:
            s = sessions.get(sid)
            if not s:
                continue
            entry = {
                "session_id": s.session_id,
                "filepath": s.filepath,
                "title": s.title,
                "url": s.url,
                "duration_s": s.duration_s,
                "fps": s.fps,
                "loop": s.loop,
                "transcript": s.transcript,
                "target_loops": s.target_loops,
                "category": getattr(s, "category", ""),
                "status": s.status,
                "error": s.error,
                "created_at": s.created_at,
            }
            # Add progress info if the session has a running video sensor
            if hasattr(s, "video_sensor") and s.video_sensor is not None:
                entry["completed_loops"] = s.video_sensor.loop_count
                entry["frames_emitted"] = s.video_sensor.frames_emitted
            else:
                entry["completed_loops"] = 0
                entry["frames_emitted"] = 0
            entries.append(entry)

        state = {
            "version": 1,
            "last_updated": time.time(),
            "active_id": active_id,
            "queue_order": queue.copy(),
            "sessions": entries,
        }

        try:
            # Atomic write: write to temp file, then rename (prevents corruption on crash)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent, suffix=".tmp", prefix=".queue_state_"
            )
            try:
                with open(tmp_fd, "w") as f:
                    json.dump(state, f, indent=2)
                Path(tmp_path).replace(self._path)
            except BaseException:
                Path(tmp_path).unlink(missing_ok=True)
                raise
            logger.debug(f"Queue state saved: {len(entries)} sessions to {self._path}")
        except OSError as e:
            logger.error(f"Failed to save queue state: {e}")

    def load(self) -> dict | None:
        """Load saved queue state from disk.

        Returns:
            Parsed state dict, or None if no saved state exists.
        """
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text())
            if data.get("version") != 1:
                logger.warning(f"Unknown queue state version: {data.get('version')}")
                return None
            logger.info(
                f"Loaded queue state: {len(data.get('sessions', []))} sessions "
                f"from {self._path}"
            )
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load queue state: {e}")
            # Remove corrupted file so it doesn't fail on every restart
            try:
                self._path.unlink(missing_ok=True)
                logger.info(f"Removed corrupted state file: {self._path}")
            except OSError:
                pass
            return None

    def clear(self) -> None:
        """Remove the state file."""
        try:
            if self._path.exists():
                self._path.unlink()
                logger.info(f"Queue state cleared: {self._path}")
        except OSError as e:
            logger.error(f"Failed to clear queue state: {e}")

    def has_resumable(self) -> list[dict]:
        """Check if there are sessions that can be resumed.

        Returns:
            List of session dicts with status 'queued' or 'running'.
        """
        state = self.load()
        if not state:
            return []
        return [
            s for s in state.get("sessions", [])
            if s.get("status") in ("queued", "running", "pending", "starting")
        ]
