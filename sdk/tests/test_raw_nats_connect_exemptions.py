"""Keep the raw nats.connect() exemption list exhaustive (issue #229).

Nearly every service should reach NATS through activelearning.EventBus,
which wraps nats.connect() with JetStream setup, decision signing, and
reconnection policy. A handful of files are intentionally exempt because
they cannot or should not depend on the SDK -- see
docs/nats-raw-connect-exemptions.md for the reasoning behind each one.

This scans the whole repo (via AST, not a plain grep, so it isn't fooled by
which alias a file imports `nats`/`Client` under) and fails CI if a *new*
raw connect call site appears anywhere outside the reviewed list, or if a
listed exemption no longer contains a raw connect call (so the list can't
silently rot into an aspirational allowlist either).
"""

from __future__ import annotations

import ast
from pathlib import Path

# repo_root/sdk/tests/this_file -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
EXEMPTIONS_DOC = REPO_ROOT / "docs" / "nats-raw-connect-exemptions.md"

_EXCLUDE_DIR_NAMES = {
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".localrun",
    "sandbox",
}

# Reviewed, intentional exceptions to "use EventBus, not raw nats.connect()".
# Each entry is documented (with rationale) in docs/nats-raw-connect-exemptions.md.
# Adding a file here is a deliberate, reviewed decision -- don't add one just
# to silence this test; add a row to the doc first.
APPROVED_RAW_CONNECT_FILES = frozenset(
    {
        "sdk/src/activelearning/nats_client.py",
        "sdk/tests/test_event_bus_validation.py",
        "sdk/tests/conftest.py",
        "neuromorphic/src/neuromorphic/cognitive_bridge.py",
        "neuromorphic/scripts/train_pump.py",
        "dashboard/src/dashboard/api.py",
        "deploy/gateway_restart_listener.py",
        "test-runner/src/test_runner/tests/conftest.py",
        "test-runner/src/test_runner/tests/test_jetstream_durability.py",
    }
)


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in _EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        yield path


def _raw_connect_bindings(tree: ast.AST) -> set[str]:
    """Names in this module bound to the raw `nats` module or `nats.aio.client.Client`."""
    bindings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nats":
                    bindings.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "nats.aio.client":
                for alias in node.names:
                    if alias.name == "Client":
                        bindings.add(alias.asname or alias.name)
    return bindings


def _has_raw_connect_call(tree: ast.AST, bindings: set[str]) -> bool:
    if not bindings:
        return False

    # Track local variables assigned directly from a bound Client
    # constructor, e.g. `nc = NATSClient()`, so `nc.connect(...)` on a later
    # line is still recognized as a raw connect.
    client_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id in bindings:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        client_vars.add(target.id)

    raw_receivers = bindings | client_vars
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "connect":
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Name) and receiver.id in raw_receivers:
            return True
        # Direct chain: NATSClient().connect(...)
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id in bindings
        ):
            return True
    return False


def _files_with_raw_connect() -> set[str]:
    found: set[str] = set()
    for path in _iter_python_files(REPO_ROOT):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        bindings = _raw_connect_bindings(tree)
        if _has_raw_connect_call(tree, bindings):
            found.add(str(path.relative_to(REPO_ROOT).as_posix()))
    return found


def test_exemptions_doc_exists():
    assert EXEMPTIONS_DOC.is_file(), f"Exemption list doc not found at {EXEMPTIONS_DOC}"


def test_no_new_raw_nats_connect_call_sites():
    found = _files_with_raw_connect()
    unapproved = sorted(found - APPROVED_RAW_CONNECT_FILES)
    assert not unapproved, (
        "New raw nats.connect()/Client().connect() call site(s) found outside the "
        f"reviewed exemption list: {unapproved}. Use activelearning.EventBus instead, "
        "or -- if this file has a genuine reason to bypass the SDK -- add it to "
        "APPROVED_RAW_CONNECT_FILES here and document why in "
        f"{EXEMPTIONS_DOC.relative_to(REPO_ROOT)}."
    )


def test_exemption_list_has_no_stale_entries():
    found = _files_with_raw_connect()
    stale = sorted(APPROVED_RAW_CONNECT_FILES - found)
    assert not stale, (
        "These exemption-list entries no longer contain a raw nats.connect() call "
        f"and should be removed from APPROVED_RAW_CONNECT_FILES: {stale}"
    )
