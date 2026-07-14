"""Keep EventBus._is_safety_critical() in sync with ADR 0001 §3 (issue #252).

ADR 0001 §3 ("The privileged Kernel publisher set") documents the subjects
only the Kernel may publish — the authoritative source for "what is
safety-critical enough to require guaranteed, durable JetStream delivery
instead of fire-and-forget core NATS." If a new Kernel-privileged subject is
added to that table without also teaching
``EventBus._is_safety_critical()``/``_SAFETY_STREAM_SUBJECTS`` (in
``activelearning.nats_client``) to route it through JetStream, it becomes
silently non-persistent: a consumer that is mid-reconnect at the moment it's
published simply never receives it, with no redelivery. This test fails CI on
exactly that drift, mirroring ``test_adr_subject_matrix.py``'s doc-vs-code
sync pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from activelearning.nats_client import EventBus

# repo_root/sdk/tests/this_file -> repo_root/docs/adr/0001-nats-authz.md
ADR_PATH = Path(__file__).resolve().parents[2] / "docs" / "adr" / "0001-nats-authz.md"

_SECTION_HEADING = "### 3. The privileged Kernel publisher set"
_TABLE_ROW = re.compile(r"^\| `([^`]+)` \|", re.MULTILINE)


def _privileged_subjects() -> list[str]:
    """Extract the backtick-quoted subject patterns from ADR 0001 §3's table."""
    text = ADR_PATH.read_text(encoding="utf-8")
    start = text.index(_SECTION_HEADING)
    next_heading = text.find("\n### ", start + len(_SECTION_HEADING))
    section = text[start:] if next_heading == -1 else text[start:next_heading]
    return _TABLE_ROW.findall(section)


def _concrete_subject(pattern: str) -> str:
    """Turn an ADR wildcard pattern into a concrete subject to check.

    ``decision.>`` / ``code.decision.>`` -> append a fake trace id.
    ``policy.*`` -> a concrete one-token subject under the policy namespace.
    Anything without a NATS wildcard is already concrete.
    """
    if pattern.endswith(".>"):
        return f"{pattern[:-1]}example-trace-id"
    if pattern.endswith(".*"):
        return f"{pattern[:-1]}example"
    return pattern


def test_adr_section_exists():
    assert ADR_PATH.is_file(), f"ADR not found at {ADR_PATH}"
    assert _SECTION_HEADING in ADR_PATH.read_text(encoding="utf-8"), (
        f"{_SECTION_HEADING!r} heading not found — did ADR 0001 §3 get renamed? "
        "Update _SECTION_HEADING in this test to match."
    )


def test_privileged_subject_table_is_non_empty():
    # Guards against the regex silently matching nothing if the ADR's table
    # format ever changes (e.g. reflowed, different column order).
    assert _privileged_subjects(), (
        "No privileged-subject rows parsed from ADR 0001 §3 — the table "
        "format may have changed; update _TABLE_ROW in this test."
    )


@pytest.mark.parametrize("pattern", _privileged_subjects())
def test_privileged_subject_is_safety_critical(pattern: str):
    concrete = _concrete_subject(pattern)
    assert EventBus._is_safety_critical(concrete), (
        f"ADR 0001 §3 declares {pattern!r} Kernel-privileged, but "
        f"EventBus._is_safety_critical({concrete!r}) is False — this subject "
        f"would be published fire-and-forget over core NATS instead of the "
        f"durable SAFETY_CRITICAL JetStream stream. Add it to "
        f"_SAFETY_CRITICAL_EXACT / _SAFETY_CRITICAL_PREFIXES in nats_client.py."
    )
