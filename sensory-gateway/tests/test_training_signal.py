"""Unit tests for the training-session-complete predicate (issue #324).

Pure logic, zero dependencies — no cv2/activelearning stubbing needed.
"""

from __future__ import annotations

import os
import sys

_GATEWAY_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _GATEWAY_ROOT not in sys.path:
    sys.path.insert(0, _GATEWAY_ROOT)

from training_signal import should_publish_training_complete  # noqa: E402


class TestShouldPublishTrainingComplete:
    def test_false_when_no_videos_completed(self):
        """Startup with an already-empty queue must not fire the signal."""
        assert should_publish_training_complete(0) is False

    def test_true_after_at_least_one_video_completed(self):
        assert should_publish_training_complete(1) is True
        assert should_publish_training_complete(5) is True

    def test_false_for_negative_counts_defensively(self):
        assert should_publish_training_complete(-1) is False
