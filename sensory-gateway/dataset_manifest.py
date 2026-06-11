"""
Dataset manifest — tracks which videos trained which brain checkpoint.

Provides reproducibility: given a manifest, you can recreate the exact
training run. Each video records its source, category, loops played,
learning score, and quality validation result.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


class DatasetManifest:
    """Records all videos used in a training run for reproducibility."""

    def __init__(self, manifest_dir: str | Path | None = None):
        self._dir = Path(manifest_dir) if manifest_dir else DEFAULT_MANIFEST_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._run_id: str = ""
        self._started_at: float = 0.0
        self._entries: list[dict[str, Any]] = []

    def start_run(self, run_id: str) -> None:
        """Begin tracking a new training run."""
        self._run_id = run_id
        self._started_at = time.time()
        self._entries = []
        logger.info(f"Dataset manifest started: run {run_id}")

    def record_video(
        self,
        session_id: str,
        filepath: str,
        title: str,
        url: str | None,
        category: str,
        target_loops: int,
        completed_loops: int,
        frames_emitted: int,
        learning_score: float,
        avg_learning_rate: float,
        quality_result: dict | None,
        video_hash: str,
        status: str,
        duration_s: float,
        elapsed_s: float,
    ) -> None:
        """Record a completed video training entry."""
        self._entries.append({
            "session_id": session_id,
            "filepath": filepath,
            "title": title,
            "url": url,
            "category": category,
            "video_hash": video_hash,
            "target_loops": target_loops,
            "completed_loops": completed_loops,
            "frames_emitted": frames_emitted,
            "learning_score": round(learning_score, 6),
            "avg_learning_rate": round(avg_learning_rate, 6),
            "duration_s": round(duration_s, 1),
            "elapsed_s": round(elapsed_s, 1),
            "status": status,
            "quality_valid": quality_result.get("valid") if quality_result else None,
            "quality_issues": quality_result.get("issues") if quality_result else [],
            "recorded_at": time.time(),
        })

    def save(self) -> Path:
        """Save the manifest to disk and return the path."""
        if not self._run_id:
            self._run_id = f"run_{int(time.time())}"

        manifest = {
            "version": 1,
            "run_id": self._run_id,
            "started_at": self._started_at,
            "completed_at": time.time(),
            "total_videos": len(self._entries),
            "total_loops": sum(e["completed_loops"] for e in self._entries),
            "total_frames": sum(e["frames_emitted"] for e in self._entries),
            "total_learning_score": round(
                sum(e["learning_score"] for e in self._entries), 6
            ),
            "categories": sorted(set(
                e["category"] for e in self._entries if e["category"]
            )),
            "videos": self._entries,
        }

        path = self._dir / f"manifest_{self._run_id}.json"
        path.write_text(json.dumps(manifest, indent=2))
        logger.info(f"Dataset manifest saved: {path} ({len(self._entries)} videos)")
        return path

    def list_runs(self) -> list[dict]:
        """List all saved training run manifests."""
        runs = []
        for p in sorted(self._dir.glob("manifest_*.json"), reverse=True):
            try:
                data = json.loads(p.read_text())
                runs.append({
                    "run_id": data.get("run_id", p.stem),
                    "started_at": data.get("started_at", 0),
                    "total_videos": data.get("total_videos", 0),
                    "total_loops": data.get("total_loops", 0),
                    "total_learning_score": data.get("total_learning_score", 0),
                    "path": str(p),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return runs

    def load_run(self, run_id: str) -> dict | None:
        """Load a specific manifest by run_id."""
        path = self._dir / f"manifest_{run_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
