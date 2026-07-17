"""Pure predicate for the training-session-complete signal (issue #324).

Split out from gateway.py so it can be unit tested without stubbing cv2,
activelearning, and gateway.py's other heavy dependencies.
"""

from __future__ import annotations


def should_publish_training_complete(videos_completed_since_drain: int) -> bool:
    """True only when the queue has just drained after processing at least one
    video — not on startup with an already-empty queue, and not on every
    subsequent poll while the queue stays empty.
    """
    return videos_completed_since_drain > 0
