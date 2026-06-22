"""Keep ADR 0001 (NATS authz) in sync with the subject registry.

ADR 0001 documents a publish/subscribe permission matrix that must cover every
subject in the codebase. The registry in ``activelearning.subjects.Subjects`` is
the canonical source of named subjects, so every constant there must appear in
the ADR. This test fails CI if a subject is added to the registry without being
documented in the matrix, preventing silent drift between code and the ADR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from activelearning.subjects import Subjects

# repo_root/sdk/tests/this_file -> repo_root/docs/adr/0001-nats-authz.md
ADR_PATH = Path(__file__).resolve().parents[2] / "docs" / "adr" / "0001-nats-authz.md"


def _registry_subjects() -> list[tuple[str, str]]:
    """Return (constant_name, subject_value) for each public subject constant."""
    out: list[tuple[str, str]] = []
    for name in dir(Subjects):
        if name.startswith("_"):
            continue
        value = getattr(Subjects, name)
        if isinstance(value, str):
            out.append((name, value))
    return out


def _normalize(subject: str) -> str:
    """Reduce a subject to the stable prefix the ADR documents.

    Trailing wildcard/prefix markers (``*``, ``>``, ``.``) vary between the
    registry (e.g. ``decision.``) and the ADR (e.g. ``decision.>``), so compare
    on the wildcard-free stem.
    """
    return subject.rstrip(".*>")


def test_adr_exists():
    assert ADR_PATH.is_file(), f"ADR not found at {ADR_PATH}"


@pytest.mark.parametrize("name, subject", _registry_subjects())
def test_subject_documented_in_adr(name: str, subject: str):
    adr_text = ADR_PATH.read_text(encoding="utf-8")
    stem = _normalize(subject)
    assert stem and stem in adr_text, (
        f"Subjects.{name} ({subject!r}) is not documented in the ADR 0001 "
        f"permission matrix. Add a row for it to {ADR_PATH.name}."
    )
