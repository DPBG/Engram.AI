"""
Deploy-time safety guards for the Meta-Programmer.

Defence-in-depth checks applied to LLM-generated code **before** it is written
to disk — independent of the Kernel decision (which only ever sees a 500-char
preview):

- ``safe_deploy_path``  — generated files may only land inside an explicit
  allowlist of roots, with the path fully normalised (``..`` traversal and
  symlink escapes are rejected). Anything else is refused.
- ``scan_source`` / ``is_dangerous`` — a full-source **AST taint scan** that
  flags dangerous sinks (``eval``/``exec``/``compile``, ``os.system``,
  ``subprocess``, sockets, dynamic import, ``pickle``/``marshal``, ``ctypes``),
  dunder reflection (``__globals__``, ``__subclasses__`` …), and any reference
  to the safety/meta machinery itself. It sees the WHOLE file, closing the
  "hide the payload past the 500-char preview" gap.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Optional

# Roots a generated artifact is allowed to be written into.
DEFAULT_ALLOWLIST = ("/data/plugins", "/data/tasks", "/data/adapters", "/data/staging")

# Modules that are dangerous to import/call. value=None → any attribute is flagged;
# value=set → only those attributes are flagged.
_DANGEROUS_MODULES: dict[str, Optional[set[str]]] = {
    "subprocess": None,
    "socket": None,
    "ctypes": None,
    "pickle": None,
    "marshal": None,
    "importlib": None,
    "pty": None,
    "os": {"system", "popen", "execv", "execve", "execvp", "execvpe", "spawnl",
           "spawnv", "spawnve", "remove", "unlink", "rmdir", "fork"},
    "shutil": {"rmtree"},
}
# Builtins that execute arbitrary code / strings.
_DANGEROUS_BUILTINS = {"eval", "exec", "compile", "__import__"}
_REFLECTION_BUILTINS = {"getattr", "setattr", "delattr"}
# Dunder attributes used for sandbox escapes.
_DUNDER_ATTRS = {"__globals__", "__builtins__", "__subclasses__", "__bases__",
                 "__mro__", "__code__", "__class__", "__dict__", "__loader__"}
# Substrings that mean the code touches the safety/meta machinery itself.
_SELF_REF = ("kernel", "safety_supervisor", "safety-supervisor",
             "meta_programmer", "meta-programmer")


@dataclass(frozen=True)
class Finding:
    severity: str   # "high" | "medium"
    rule: str
    detail: str


def _allowlist_roots(allowlist: Optional[list[str]] = None) -> list[str]:
    if allowlist:
        roots = list(allowlist)
    else:
        env = os.environ.get("ENGRAM_DEPLOY_ALLOWLIST", "")
        roots = [r for r in env.split(os.pathsep) if r] or list(DEFAULT_ALLOWLIST)
    return [os.path.realpath(r) for r in roots]


def safe_deploy_path(target_path: str, allowlist: Optional[list[str]] = None) -> tuple[bool, str]:
    """Validate a deploy target path.

    Returns ``(ok, resolved_path_or_reason)``. The path is accepted only if,
    after resolving symlinks and ``..``, it is the allowlisted root itself or
    strictly inside one of the allowlisted roots.
    """
    if not target_path or not isinstance(target_path, str):
        return False, "empty or non-string target path"
    roots = _allowlist_roots(allowlist)
    resolved = os.path.realpath(target_path)
    for root in roots:
        if resolved == root or resolved.startswith(root + os.sep):
            return True, resolved
    return False, f"path {resolved!r} is outside the deploy allowlist {roots}"


def scan_source(code: str) -> list[Finding]:
    """Full-source AST taint scan. Returns a list of findings (possibly empty)."""
    findings: list[Finding] = []
    if not isinstance(code, str) or not code.strip():
        return findings
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Finding("high", "syntax_error", f"cannot parse generated code: {e}")]

    for node in ast.walk(tree):
        # Calls: eval/exec/compile/__import__, os.system, subprocess.*, getattr(__x__)
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                if fn.id in _DANGEROUS_BUILTINS:
                    findings.append(Finding("high", "dangerous_builtin", f"{fn.id}()"))
                elif fn.id in _REFLECTION_BUILTINS:
                    for arg in node.args:
                        if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                                and arg.value.startswith("__")):
                            findings.append(Finding("high", "dynamic_attr",
                                                    f"{fn.id}(..., {arg.value!r})"))
            elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                mod, attr = fn.value.id, fn.attr
                allowed = _DANGEROUS_MODULES.get(mod, "MISS")
                if allowed != "MISS" and (allowed is None or attr in allowed):
                    findings.append(Finding("high", "dangerous_call", f"{mod}.{attr}()"))

        # Imports of dangerous or self-referential modules
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _DANGEROUS_MODULES:
                    findings.append(Finding("medium", "dangerous_import", f"import {alias.name}"))
                if any(s in alias.name.lower() for s in _SELF_REF):
                    findings.append(Finding("high", "self_referential", f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if top in _DANGEROUS_MODULES:
                findings.append(Finding("medium", "dangerous_import", f"from {mod} import …"))
            if any(s in mod.lower() for s in _SELF_REF):
                findings.append(Finding("high", "self_referential", f"from {mod} import …"))

        # Dunder reflection attribute access (e.g. f.__globals__)
        elif isinstance(node, ast.Attribute) and node.attr in _DUNDER_ATTRS:
            findings.append(Finding("high", "dunder_access", f".{node.attr}"))

    return findings


def is_dangerous(code: str) -> bool:
    """True if the source contains any high-severity finding (should be blocked)."""
    return any(f.severity == "high" for f in scan_source(code))


def deploy_atomically(target_path: str, code: str, validate_syntax: bool = True) -> tuple[bool, str]:
    """Write ``code`` to ``target_path`` with automatic rollback on failure (Phase 1.9).

    A deploy must never leave the system in a half-broken state. This:
    1. Snapshots any existing file at ``target_path``.
    2. Writes the new code.
    3. Optionally validates it compiles (catches syntax errors before they
       break an import at runtime).
    4. On any failure, **rolls back** — restores the previous content, or
       removes the file entirely if it was newly created — so a bad deploy
       leaves no partial artifact behind.

    Returns ``(ok, detail)``. The caller (and the allowlist check in
    ``safe_deploy_path``) remain responsible for *where* it's allowed to write;
    this only governs the write itself.
    """
    existed = os.path.exists(target_path)
    prior: Optional[bytes] = None
    if existed:
        try:
            with open(target_path, "rb") as f:
                prior = f.read()
        except OSError as e:
            return False, f"could not snapshot existing file: {e}"

    def _rollback() -> None:
        if existed and prior is not None:
            with open(target_path, "wb") as f:
                f.write(prior)
        elif not existed and os.path.exists(target_path):
            os.remove(target_path)

    try:
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(code)

        if validate_syntax:
            try:
                compile(code, target_path, "exec")
            except SyntaxError as e:
                _rollback()
                return False, f"rolled back — syntax error: {e}"

        return True, "deployed"
    except Exception as e:  # noqa: BLE001 — any failure must trigger rollback
        try:
            _rollback()
        except Exception as re:  # noqa: BLE001
            return False, f"deploy failed ({e}); ROLLBACK ALSO FAILED ({re})"
        return False, f"rolled back — deploy error: {e}"
