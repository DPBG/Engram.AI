"""Shared pytest configuration for neuromorphic/tests/.

Ensures tests/helpers/ (issue #438's shared equivalence-test harness) is
importable as `helpers.*` regardless of pytest's import-mode -- a
conftest.py's own directory is always added to sys.path by pytest, which
is a more reliable guarantee than relying on the default "prepend"
import-mode's rootdir-insertion behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
