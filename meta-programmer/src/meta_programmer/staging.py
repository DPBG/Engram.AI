"""
Staging Manager - Manages code staging flow.

Flow:
1. pending/     - CodeGen writes here
2. testing/     - After Kernel ALLOW
3. human_review/ - After Kernel DEFER
4. approved/    - Tests passed
5. rejected/    - Denied or failed
"""

import json
import logging
import os
import shutil
import time
from typing import Optional

logger = logging.getLogger(__name__)


def is_review_expired(metadata: dict, now_ms: int, ttl_ms: int) -> bool:
    """True if a human-review item has waited longer than its TTL (Phase 1.9).

    Fail-closed: an item with no ``created_at`` (legacy / pre-timestamp) is
    treated as expired, so an un-aged pending approval can't dodge the sweep.
    """
    created = metadata.get("created_at")
    if not isinstance(created, (int, float)):
        return True
    return (now_ms - created) >= ttl_ms


class StagingManager:
    """
    Manages the staging directory structure for code generation.

    Tracks code through its lifecycle from generation to deployment.
    """

    def __init__(self, staging_root: str = "/data/staging"):
        self.staging_root = staging_root
        self.pending_dir = os.path.join(staging_root, "pending")
        self.testing_dir = os.path.join(staging_root, "testing")
        self.human_review_dir = os.path.join(staging_root, "human_review")
        self.approved_dir = os.path.join(staging_root, "approved")
        self.rejected_dir = os.path.join(staging_root, "rejected")

    def initialize(self) -> None:
        """Create staging directories if they don't exist."""
        for directory in [
            self.pending_dir,
            self.testing_dir,
            self.human_review_dir,
            self.approved_dir,
            self.rejected_dir,
        ]:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Staging directory ready: {directory}")

    def stage_pending(
        self,
        trace_id: str,
        target_path: str,
        code: str,
        tests: Optional[str] = None,
    ) -> str:
        """
        Stage code in pending directory.

        Args:
            trace_id: Unique trace ID for this code
            target_path: Final deployment path
            code: Code content
            tests: Optional test content

        Returns:
            Path to staged code file
        """
        stage_dir = os.path.join(self.pending_dir, trace_id)
        os.makedirs(stage_dir, exist_ok=True)

        # Write code
        code_path = os.path.join(stage_dir, "code.py")
        with open(code_path, "w") as f:
            f.write(code)

        # Write tests if provided
        if tests:
            tests_path = os.path.join(stage_dir, "tests.py")
            with open(tests_path, "w") as f:
                f.write(tests)

        # Write metadata. created_at lets the human-review sweep fail an
        # unanswered DEFER closed after a TTL (Phase 1.9); it is preserved
        # across stage moves since _move_stage only rewrites the `stage` field.
        metadata = {
            "trace_id": trace_id,
            "target_path": target_path,
            "stage": "pending",
            "created_at": int(time.time() * 1000),
        }
        metadata_path = os.path.join(stage_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Staged pending code: {trace_id}")
        return code_path

    def stage_testing(self, trace_id: str) -> None:
        """Move code from pending to testing."""
        self._move_stage(trace_id, self.pending_dir, self.testing_dir, "testing")

    def stage_human_review(self, trace_id: str) -> None:
        """Move code from pending to human_review."""
        self._move_stage(trace_id, self.pending_dir, self.human_review_dir, "human_review")

    def stage_approved(self, trace_id: str) -> None:
        """Move code from testing to approved."""
        self._move_stage(trace_id, self.testing_dir, self.approved_dir, "approved")

    def stage_rejected(self, trace_id: str, reason: str) -> None:
        """Move code to rejected with reason."""
        source_dir = None
        for check_dir in [self.pending_dir, self.testing_dir, self.human_review_dir]:
            candidate = os.path.join(check_dir, trace_id)
            if os.path.exists(candidate):
                source_dir = check_dir
                break

        if not source_dir:
            logger.warning(f"Cannot reject {trace_id}: not found in pending or testing")
            return

        dest_dir = os.path.join(self.rejected_dir, trace_id)
        source = os.path.join(source_dir, trace_id)

        try:
            shutil.move(source, dest_dir)

            # Update metadata
            metadata_path = os.path.join(dest_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata["stage"] = "rejected"
                metadata["rejection_reason"] = reason
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Staged rejected: {trace_id} - {reason}")

        except Exception as e:
            logger.error(f"Error staging rejected: {e}")

    def _move_stage(self, trace_id: str, from_dir: str, to_dir: str, stage_name: str) -> None:
        """Internal helper to move between stages."""
        source = os.path.join(from_dir, trace_id)
        dest = os.path.join(to_dir, trace_id)

        if not os.path.exists(source):
            logger.warning(f"Cannot move {trace_id}: not found in {from_dir}")
            return

        try:
            shutil.move(source, dest)

            # Update metadata
            metadata_path = os.path.join(dest, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata["stage"] = stage_name
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Moved to {stage_name}: {trace_id}")

        except Exception as e:
            logger.error(f"Error moving stage: {e}")

    def list_human_review(self) -> list[dict]:
        """Return metadata for every item awaiting human review (Phase 1.9)."""
        items = []
        if not os.path.isdir(self.human_review_dir):
            return items
        for trace_id in os.listdir(self.human_review_dir):
            metadata_path = os.path.join(self.human_review_dir, trace_id, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        items.append(json.load(f))
                except (OSError, json.JSONDecodeError):
                    # A trace whose metadata can't be read is failed closed by
                    # the caller; surface it with just its id so it isn't lost.
                    items.append({"trace_id": trace_id})
        return items

    def expired_reviews(self, now_ms: int, ttl_ms: int) -> list[str]:
        """trace_ids of human-review items that have exceeded their TTL."""
        return [
            m["trace_id"]
            for m in self.list_human_review()
            if "trace_id" in m and is_review_expired(m, now_ms, ttl_ms)
        ]

    def get_metadata(self, trace_id: str) -> Optional[dict]:
        """Get metadata for a trace_id (searches all stages)."""
        for stage_dir in [
            self.pending_dir,
            self.testing_dir,
            self.human_review_dir,
            self.approved_dir,
            self.rejected_dir,
        ]:
            metadata_path = os.path.join(stage_dir, trace_id, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    return json.load(f)
        return None
