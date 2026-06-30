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
import subprocess
import sys
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


def _resolve_tainted_imports(tree: ast.AST, findings: list[Finding]) -> dict[str, str]:
    """Map local names bound to dangerous sinks via ``from <mod> import <sink>``.

    The attribute-form check in :func:`scan_source` only catches ``os.system(...)``.
    A sink pulled in with ``from os import system`` is called as a bare ``Name``
    (``system(...)``), so it must be tracked separately or it sails through the
    gate (fail-open). This returns ``{local_name: "mod.attr"}`` for every name a
    ``from <dangerous_module> import <sink> [as alias]`` brings into scope, so a
    later bare call to that name can be flagged **high** — symmetric with the
    attribute-form check. The curated per-module attribute set is respected, so
    benign names (``from os import getcwd``) are *not* tainted.

    ``from <dangerous_module> import *`` pulls every sink in wholesale and
    defeats name tracking, so it is flagged **high** directly.
    """
    tainted: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        top = (node.module or "").split(".")[0]
        allowed = _DANGEROUS_MODULES.get(top, "MISS")
        if allowed == "MISS":
            continue
        for alias in node.names:
            if alias.name == "*":
                findings.append(Finding("high", "dangerous_star_import",
                                        f"from {node.module} import *"))
                continue
            if allowed is None or alias.name in allowed:
                local = alias.asname or alias.name
                tainted[local] = f"{top}.{alias.name}"
    return tainted


def _propagate_aliases(tree: ast.AST, tainted: dict[str, str]) -> None:
    """Extend ``tainted`` through simple ``Name = Name`` rebindings (in place).

    A bare call to a tainted sink (``system(...)``) is caught, but a trivial
    local alias is not::

        from os import system as s
        runner = s          # runner is not tainted …
        runner("id")        # … so this call would slip through (fail-open)

    Conservatively propagate taint across straightforward name-to-name
    assignments to a fixpoint (so chains ``a = s; b = a`` are covered). Dangerous
    builtins (``eval``/``exec``/…) are seeded as sources too, so ``e = exec`` is
    treated like ``exec`` itself. Over-tainting is acceptable here: a scanner in
    the safety layer must fail closed.
    """
    sources = dict(tainted)
    for builtin in _DANGEROUS_BUILTINS:
        sources.setdefault(builtin, f"{builtin}")

    assigns: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    assigns.append((tgt.id, node.value.id))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and isinstance(node.value, ast.Name):
            if isinstance(node.target, ast.Name):
                assigns.append((node.target.id, node.value.id))

    changed = True
    while changed:
        changed = False
        for target, value in assigns:
            if value in sources and target not in sources:
                sources[target] = sources[value]
                changed = True

    # Builtins are already flagged directly; only fold back their aliases.
    for name, detail in sources.items():
        if name not in _DANGEROUS_BUILTINS:
            tainted.setdefault(name, detail)


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

    # Names bound to dangerous sinks via `from <module> import <sink>`, so a bare
    # call to one (e.g. `system(...)`) is flagged like `os.system(...)` would be.
    # Then extend that set through simple local aliases (`runner = system`).
    tainted_names = _resolve_tainted_imports(tree, findings)
    _propagate_aliases(tree, tainted_names)

    for node in ast.walk(tree):
        # Calls: eval/exec/compile/__import__, os.system, subprocess.*, getattr(__x__)
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                if fn.id in _DANGEROUS_BUILTINS:
                    findings.append(Finding("high", "dangerous_builtin", f"{fn.id}()"))
                elif fn.id in tainted_names:
                    findings.append(Finding("high", "dangerous_call",
                                            f"{tainted_names[fn.id]}()"))
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


def run_health_probe(target_path: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Import-probe the deployed module in an isolated subprocess (Phase E1.9.2).

    Spawns a throwaway interpreter that tries to load the file via
    ``importlib.util``. Any crash, import error, or hang within ``timeout``
    seconds is treated as a probe failure and should trigger rollback.

    Returns ``(ok, reason)``.
    """
    if not os.path.exists(target_path):
        return False, f"deployed file not found: {target_path!r}"

    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_probe', {target_path!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print('ok')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "health probe passed"
        stderr = result.stderr.strip() or result.stdout.strip()
        return False, f"probe failed (exit {result.returncode}): {stderr}"
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return False, f"probe error: {e}"


def deploy_atomically(target_path: str, code: str, validate_syntax: bool = True, probe_timeout: float = 0.0) -> tuple[bool, str]:
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

        if probe_timeout > 0.0:
            ok_probe, probe_reason = run_health_probe(target_path, timeout=probe_timeout)
            if not ok_probe:
                _rollback()
                return False, f"rolled back — health probe failed: {probe_reason}"

        return True, "deployed"
    except Exception as e:  # noqa: BLE001 — any failure must trigger rollback
        try:
            _rollback()
        except Exception as re:  # noqa: BLE001
            return False, f"deploy failed ({e}); ROLLBACK ALSO FAILED ({re})"
        return False, f"rolled back — deploy error: {e}"
