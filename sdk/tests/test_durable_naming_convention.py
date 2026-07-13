"""Keep JetStream durable-consumer names collision-free and documented.

docs/JETSTREAM-DURABLE-NAMING.md documents a <service>-<purpose> convention
for static durable consumer names (EventBus.js_subscribe's `durable` kwarg)
so that two services — or two consumers in the same service — can never
silently collide on the same durable name and share/steal each other's
JetStream delivery cursor (issue #255).

This test is the audit the issue asked for, made permanent: it scans every
``*.py`` file in the repo for a literal ``durable="..."`` argument, and
fails CI if a new one is added that collides with an existing name, doesn't
match the convention, or isn't reflected in the doc's table — the same
doc-vs-code sync pattern test_adr_subject_matrix.py already uses for the
NATS subject registry.
"""

from __future__ import annotations

import re
from pathlib import Path

# repo_root/sdk/tests/this_file -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "JETSTREAM-DURABLE-NAMING.md"
REGISTRY_PATH = REPO_ROOT / "launcher" / "registry.py"

# Directories that are never source code (build artifacts, deps, VCS).
_SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".ruff_cache",
    ".pytest_cache",
}

_DURABLE_LITERAL_RE = re.compile(r'durable\s*=\s*"([^"]+)"')

# meta-programmer is intentionally absent from launcher/registry.py (it needs
# the Docker socket, so it only runs via `docker compose --profile extra up`)
# but is still a real service name a durable consumer could legitimately be
# prefixed with — see deploy/scripts/gen-creds.sh's ALL_SERVICES comment.
_EXTRA_SERVICE_NAMES = {"meta-programmer"}


_THIS_FILE = Path(__file__).resolve()


def _iter_source_files() -> list[Path]:
    out = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if path == _THIS_FILE:
            continue  # our own docstrings/regex source mention durable="..." as prose
        out.append(path)
    return out


def _find_static_durables() -> dict[str, list[Path]]:
    """Return {durable_name: [files it appears in]} for every literal
    ``durable="..."`` found in the repo (production code and tests alike —
    a collision is a collision regardless of which file introduces it)."""
    found: dict[str, list[Path]] = {}
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _DURABLE_LITERAL_RE.finditer(text):
            found.setdefault(match.group(1), []).append(path)
    return found


def _known_service_names() -> set[str]:
    registry_text = REGISTRY_PATH.read_text(encoding="utf-8")
    names = set(re.findall(r'name="([^"]+)"', registry_text))
    return names | _EXTRA_SERVICE_NAMES


_STATIC_DURABLES = _find_static_durables()


def test_registry_and_doc_exist():
    assert REGISTRY_PATH.is_file(), f"service registry not found at {REGISTRY_PATH}"
    assert DOC_PATH.is_file(), f"naming convention doc not found at {DOC_PATH}"


def test_at_least_the_known_kernel_durables_were_found():
    # Sanity check that the scanner itself works — if this starts failing,
    # the regex or the file walk broke, not the convention.
    assert "kernel-action-proposals" in _STATIC_DURABLES
    assert "kernel-code-proposals" in _STATIC_DURABLES


def test_no_two_files_share_a_durable_name():
    collisions = {name: files for name, files in _STATIC_DURABLES.items() if len(files) > 1}
    assert not collisions, (
        "Multiple call sites use the same JetStream durable name — they will "
        f"silently share/steal each other's delivery cursor: {collisions}. "
        "See docs/JETSTREAM-DURABLE-NAMING.md."
    )


def test_static_durables_follow_service_purpose_convention():
    known_services = _known_service_names()
    for name in _STATIC_DURABLES:
        prefix_matches = [
            svc for svc in known_services if name == svc or name.startswith(svc + "-")
        ]
        assert prefix_matches, (
            f"Durable {name!r} doesn't start with a known service name from "
            f"launcher/registry.py (or meta-programmer). Convention is "
            f"<service>-<purpose> — see docs/JETSTREAM-DURABLE-NAMING.md."
        )
        purpose = name[len(prefix_matches[0]) :].lstrip("-")
        assert purpose, (
            f"Durable {name!r} has no <purpose> suffix beyond the service name "
            "— see docs/JETSTREAM-DURABLE-NAMING.md."
        )
        assert re.fullmatch(
            r"[a-z0-9]+(-[a-z0-9]+)*", name
        ), f"Durable {name!r} isn't kebab-case — see docs/JETSTREAM-DURABLE-NAMING.md."


def test_every_static_durable_is_documented():
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    undocumented = [name for name in _STATIC_DURABLES if name not in doc_text]
    assert not undocumented, (
        f"Durable name(s) {undocumented} appear in code but not in "
        f"{DOC_PATH.name}'s table. Add a row — see the doc's 'Adding a new "
        "static durable' section."
    )


def test_dynamic_waiter_pattern_is_present_and_documented():
    # The ephemeral wait_for_decision() consumer name is an f-string, not a
    # literal, so the scanner above can't see it — check its source and the
    # doc separately.
    nats_client = (REPO_ROOT / "sdk" / "src" / "activelearning" / "nats_client.py").read_text(
        encoding="utf-8"
    )
    assert 'f"waiter-{' in nats_client, (
        "wait_for_decision()'s durable-naming pattern changed — update "
        "docs/JETSTREAM-DURABLE-NAMING.md's §2 to match."
    )
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    assert "waiter-<type>-<trace_id>" in doc_text or "waiter-" in doc_text
