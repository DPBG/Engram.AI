"""
Video quality gate — validates videos before they enter the training queue.

Checks: codec integrity, min duration, non-blank frames, audio presence,
hash-based deduplication, and blacklist filtering.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BLACKLIST_PATH = Path(__file__).resolve().parent / "blacklist.json"


class VideoQualityGate:
    """Pre-training video validation and deduplication."""

    def __init__(
        self,
        min_duration_s: float = 5.0,
        min_frames: int = 10,
        blank_threshold: float = 0.02,
        max_blank_ratio: float = 0.8,
        sample_frames: int = 10,
        blacklist_path: str | Path | None = None,
    ):
        self.min_duration_s = min_duration_s
        self.min_frames = min_frames
        self.blank_threshold = blank_threshold  # mean pixel value below this = blank
        self.max_blank_ratio = max_blank_ratio  # >80% blank frames = bad video
        self.sample_frames = sample_frames  # how many frames to sample for quality check
        self._blacklist_path = Path(blacklist_path) if blacklist_path else DEFAULT_BLACKLIST_PATH
        self._blacklist: dict[str, dict[str, Any]] = {}
        self._known_hashes: dict[str, str] = {}  # hash -> session_id or filepath
        self._load_blacklist()

    def _load_blacklist(self) -> None:
        """Load blacklist from disk."""
        if self._blacklist_path.exists():
            try:
                data = json.loads(self._blacklist_path.read_text())
                self._blacklist = data.get("blacklisted", {})
                self._known_hashes = data.get("known_hashes", {})
                logger.info(f"Loaded blacklist: {len(self._blacklist)} entries")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load blacklist: {e}")

    def _save_blacklist(self) -> None:
        """Persist blacklist to disk atomically (tempfile + rename)."""
        try:
            data = {
                "blacklisted": self._blacklist,
                "known_hashes": self._known_hashes,
            }
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self._blacklist_path.parent, suffix=".tmp", prefix=".blacklist_"
            )
            try:
                with open(tmp_fd, "w") as f:
                    json.dump(data, f, indent=2)
                Path(tmp_path).replace(self._blacklist_path)
            except BaseException:
                Path(tmp_path).unlink(missing_ok=True)
                raise
        except OSError as e:
            logger.error(f"Failed to save blacklist: {e}")

    def compute_hash(self, filepath: str) -> str:
        """Compute a fingerprint from first, middle, and last frames.

        Uses SHA256 of concatenated frame pixels for fast dedup detection.
        """
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return ""

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return ""

        hasher = hashlib.sha256()
        positions = [0, total // 2, max(0, total - 1)]
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                small = cv2.resize(frame, (32, 32))
                hasher.update(small.tobytes())
        cap.release()
        return hasher.hexdigest()[:16]

    def validate(self, filepath: str) -> dict[str, Any]:
        """Run all quality checks on a video file.

        Returns:
            dict with keys: valid (bool), issues (list[str]), metadata (dict)
        """
        issues: list[str] = []
        metadata: dict[str, Any] = {"filepath": filepath}

        # Check file exists
        path = Path(filepath)
        if not path.exists():
            return {"valid": False, "issues": ["File not found"], "metadata": metadata}

        if path.stat().st_size == 0:
            return {"valid": False, "issues": ["Empty file"], "metadata": metadata}

        # Check blacklist
        video_hash = self.compute_hash(filepath)
        metadata["hash"] = video_hash
        if video_hash and video_hash in self._blacklist:
            reason = self._blacklist[video_hash].get("reason", "unknown")
            return {
                "valid": False,
                "issues": [f"Blacklisted: {reason}"],
                "metadata": metadata,
            }

        # Check for duplicates
        if video_hash and video_hash in self._known_hashes:
            existing = self._known_hashes[video_hash]
            issues.append(f"Duplicate of {existing}")
            # Not a hard fail — caller decides whether to skip duplicates

        # Open with OpenCV
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {"valid": False, "issues": ["Cannot open video file"], "metadata": metadata}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_s = total_frames / fps if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata.update(
            {
                "total_frames": total_frames,
                "fps": round(fps, 1),
                "duration_s": round(duration_s, 1),
                "width": width,
                "height": height,
            }
        )

        # Duration check
        if duration_s < self.min_duration_s:
            issues.append(f"Too short: {duration_s:.1f}s < {self.min_duration_s}s")

        # Frame count check
        if total_frames < self.min_frames:
            issues.append(f"Too few frames: {total_frames} < {self.min_frames}")

        # Sample frames for quality
        blank_count = 0
        corrupt_count = 0
        step = max(1, total_frames // self.sample_frames)
        sampled = 0
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                corrupt_count += 1
                sampled += 1
                continue
            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if gray.mean() < self.blank_threshold or gray.std() < 0.01:
                blank_count += 1
            if sampled >= self.sample_frames:
                break

        cap.release()

        metadata["sampled_frames"] = sampled
        metadata["blank_frames"] = blank_count
        metadata["corrupt_frames"] = corrupt_count

        if sampled > 0:
            blank_ratio = blank_count / sampled
            metadata["blank_ratio"] = round(blank_ratio, 2)
            if blank_ratio > self.max_blank_ratio:
                issues.append(
                    f"Too many blank frames: {blank_ratio:.0%} > {self.max_blank_ratio:.0%}"
                )

        if corrupt_count > 0 and sampled > 0 and corrupt_count / sampled > 0.5:
            issues.append(f"Corrupt frames: {corrupt_count}/{sampled} unreadable")

        # Check audio track exists (via ffprobe)
        has_audio = self._check_audio(filepath)
        metadata["has_audio"] = has_audio
        if not has_audio:
            issues.append("No audio track (training benefits from audio+visual)")

        # Determine validity: hard failures vs warnings
        hard_fails = [
            i
            for i in issues
            if any(
                keyword in i.lower()
                for keyword in [
                    "blacklisted",
                    "cannot open",
                    "empty file",
                    "file not found",
                    "too short",
                    "too few frames",
                    "too many blank",
                    "corrupt",
                ]
            )
        ]
        valid = len(hard_fails) == 0
        metadata["is_duplicate"] = any("duplicate" in i.lower() for i in issues)

        return {"valid": valid, "issues": issues, "metadata": metadata}

    def _check_audio(self, filepath: str) -> bool:
        """Check if video file has an audio track using ffprobe."""
        if not shutil.which("ffprobe"):
            return True  # assume yes if ffprobe not available

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    filepath,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "audio" in result.stdout
        except (subprocess.TimeoutExpired, OSError):
            return True  # assume yes on error

    def register_hash(self, filepath: str, identifier: str, video_hash: str | None = None) -> None:
        """Register a video hash after it passes validation.

        If video_hash is provided (e.g. from a prior validate() call), reuses it
        to avoid recomputing the hash (which opens the video file again).
        """
        if not video_hash:
            video_hash = self.compute_hash(filepath)
        if video_hash:
            self._known_hashes[video_hash] = identifier
            self._save_blacklist()

    def blacklist(self, filepath: str, reason: str, video_hash: str | None = None) -> None:
        """Add a video to the blacklist."""
        if not video_hash:
            video_hash = self.compute_hash(filepath)
        if not video_hash:
            logger.warning(f"Cannot compute hash for {filepath}, blacklisting by path")
            video_hash = filepath

        self._blacklist[video_hash] = {
            "filepath": filepath,
            "reason": reason,
            "filename": Path(filepath).name,
        }
        self._save_blacklist()
        logger.info(f"Blacklisted: {Path(filepath).name} — {reason}")

    def unblacklist(self, video_hash: str) -> bool:
        """Remove a video from the blacklist."""
        if video_hash in self._blacklist:
            removed = self._blacklist.pop(video_hash)
            self._save_blacklist()
            logger.info(f"Unblacklisted: {removed.get('filename', video_hash)}")
            return True
        return False

    def get_blacklist(self) -> list[dict]:
        """Return the current blacklist."""
        return [{"hash": h, **info} for h, info in self._blacklist.items()]

    def is_blacklisted(self, filepath: str) -> bool:
        """Check if a video is blacklisted."""
        video_hash = self.compute_hash(filepath)
        return video_hash in self._blacklist if video_hash else False

    def is_duplicate(self, filepath: str) -> str | None:
        """Check if video is a duplicate. Returns the existing identifier or None."""
        video_hash = self.compute_hash(filepath)
        if video_hash and video_hash in self._known_hashes:
            return self._known_hashes[video_hash]
        return None
