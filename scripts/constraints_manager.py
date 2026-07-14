#!/usr/bin/env python3
"""Manage shared dependency floors defined in constraints.txt.

Commands
--------
  verify              Check every per-service dependency file against the floors in
                      constraints.txt.  Exits 0 if consistent, 1 if problems are found.

  bump PKG VERSION    Raise PKG's floor to >=VERSION in constraints.txt and every
                      per-service file that declares it at a lower (or exact-pinned) level.

Examples
--------
  python scripts/constraints_manager.py verify
  python scripts/constraints_manager.py bump pydantic 2.7.0
  python scripts/constraints_manager.py bump nats-py 2.8.0
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS = REPO_ROOT / "constraints.txt"

# ---------------------------------------------------------------------------
# Package-name canonicalisation (PEP 503)
# ---------------------------------------------------------------------------

_SEP = re.compile(r"[-_.]+")


def canonicalize(name: str) -> str:
    """Normalise package name to lowercase with hyphens (PEP 503)."""
    return _SEP.sub("-", name).lower()


def _name_pattern(canon: str) -> str:
    """Return a regex fragment matching all equivalent spellings of *canon*."""
    return r"[-_.]".join(re.escape(part) for part in re.split(r"[-_.]", canon))


# ---------------------------------------------------------------------------
# Dependency specifier parsing
# ---------------------------------------------------------------------------

_SPEC = re.compile(
    r"^"
    r"([A-Za-z0-9][A-Za-z0-9._-]*)"  # package name
    r"(\[[^\]]*\])?"  # optional [extras]
    r"\s*(>=|==|~=|!=|>|<|<=)\s*"  # operator
    r"([\d][^\s;#,\\]*)",  # version (up to whitespace / comment / marker)
    re.IGNORECASE,
)


@dataclass
class Dep:
    pkg: str  # canonical name
    raw: str  # name as written in the source file
    extras: str  # e.g. "[standard]" or ""
    op: str  # ">=", "==", …
    version: str  # "2.6.0"
    lineno: int  # 1-based; 0 = from TOML (no line info available)


def _parse(line: str, lineno: int = 0) -> Dep | None:
    s = line.strip()
    if not s or s[0] in ("#", "-", "["):
        return None
    m = _SPEC.match(s)
    if not m:
        return None
    return Dep(
        pkg=canonicalize(m.group(1)),
        raw=m.group(1),
        extras=m.group(2) or "",
        op=m.group(3),
        version=m.group(4),
        lineno=lineno,
    )


def _read_requirements(path: Path) -> list[Dep]:
    return [
        d
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if (d := _parse(line, i)) is not None
    ]


def _read_pyproject(path: Path) -> list[Dep]:
    try:
        import tomllib  # stdlib (Python 3.11+)
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return _pyproject_regex_fallback(path)

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    deps: list[Dep] = []
    project = data.get("project", {})
    for entry in project.get("dependencies", []):
        if d := _parse(entry):
            deps.append(d)
    for group in project.get("optional-dependencies", {}).values():
        for entry in group:
            if d := _parse(entry):
                deps.append(d)
    return deps


def _pyproject_regex_fallback(path: Path) -> list[Dep]:
    """Regex-based extraction for environments without tomllib/tomli."""
    deps: list[Dep] = []
    for m in re.finditer(r'["\']([^"\']+)["\']', path.read_text()):
        if d := _parse(m.group(1)):
            deps.append(d)
    return deps


def _read(path: Path) -> list[Dep]:
    return _read_pyproject(path) if path.name == "pyproject.toml" else _read_requirements(path)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _dep_files(root: Path) -> list[Path]:
    """Return requirements-local.txt + per-service requirements.txt / pyproject.toml."""
    found: list[Path] = []
    rl = root / "requirements-local.txt"
    if rl.exists():
        found.append(rl)
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        for name in ("requirements.txt", "pyproject.toml"):
            p = child / name
            if p.exists():
                found.append(p)
    return found


# ---------------------------------------------------------------------------
# Version comparison (stdlib only — no `packaging` required)
# ---------------------------------------------------------------------------


def _vtuple(v: str) -> tuple[int, ...]:
    # Strip local version suffix (+something) then extract all numeric segments.
    return tuple(int(x) for x in re.findall(r"\d+", v.split("+")[0]))


def _vlt(a: str, b: str) -> bool:
    return _vtuple(a) < _vtuple(b)


def _veq(a: str, b: str) -> bool:
    return _vtuple(a) == _vtuple(b)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _verify(root: Path) -> int:
    """Check all dep files against constraints.txt. Returns exit code."""
    if not CONSTRAINTS.exists():
        print(f"ERROR: {CONSTRAINTS} not found")
        return 1

    floors = {d.pkg: d for d in _read_requirements(CONSTRAINTS) if d.op == ">="}

    issues: list[tuple[str, Path, Dep, Dep]] = []
    for path in _dep_files(root):
        by_pkg = {d.pkg: d for d in _read(path)}
        for pkg, floor in floors.items():
            dep = by_pkg.get(pkg)
            if dep is None:
                continue
            if dep.op == "==":
                kind = "CONFLICT" if _vlt(dep.version, floor.version) else "EXACT"
            elif dep.op == ">=" and _vlt(dep.version, floor.version):
                kind = "CONFLICT"
            elif dep.op == ">=" and not _veq(dep.version, floor.version):
                kind = "DRIFT"
            else:
                continue
            issues.append((kind, path, dep, floor))

    if not issues:
        print("OK — all declared deps meet the floors in constraints.txt")
        return 0

    labels = {
        "CONFLICT": "CONFLICT  (below floor — pip may install an incompatible version)",
        "EXACT": "EXACT-PIN (== pins silently break on the next bump; use >= instead)",
        "DRIFT": "DRIFT     (floor is above constraints.txt — informational only)",
    }
    for kind in ("CONFLICT", "EXACT", "DRIFT"):
        group = [(f, d, fl) for k, f, d, fl in issues if k == kind]
        if not group:
            continue
        print(f"\n{labels[kind]}")
        for fpath, dep, floor in group:
            loc = f":{dep.lineno}" if dep.lineno else ""
            print(
                f"  {fpath.relative_to(root)}{loc}"
                f"  {dep.raw}{dep.extras}{dep.op}{dep.version}"
                f"  (floor: {floor.raw}{floor.op}{floor.version})"
            )

    n = {k: sum(1 for x in issues if x[0] == k) for k in ("CONFLICT", "EXACT", "DRIFT")}
    print(
        f"\nSummary: {n['CONFLICT']} conflict(s), {n['EXACT']} exact-pin(s),"
        f" {n['DRIFT']} drift(s)"
    )
    return 1 if (n["CONFLICT"] or n["EXACT"]) else 0


# ---------------------------------------------------------------------------
# bump — in-place text replacement
# ---------------------------------------------------------------------------


def _bump_req(path: Path, canon: str, new_ver: str) -> bool:
    """Update the floor for *canon* in a requirements*.txt file. Returns True if changed."""
    lines = path.read_text().splitlines(keepends=True)
    pat = re.compile(
        r"^(\s*)(" + _name_pattern(canon) + r")(\[[^\]]*\])?\s*(>=|==)\s*([\d][^\s;#,\\]*)",
        re.IGNORECASE,
    )
    out: list[str] = []
    changed = False
    for line in lines:
        m = pat.match(line)
        if m:
            indent, raw, extras, op, ver = m.group(1, 2, 3, 4, 5)
            extras = extras or ""
            tail = line[m.end() :]  # trailing comment, env-marker, or newline
            # Update when: == pin at or below new_ver, or >= floor below new_ver
            need = (op == "==" and not _vlt(new_ver, ver)) or (op == ">=" and _vlt(ver, new_ver))
            if need:
                out.append(f"{indent}{raw}{extras}>={new_ver}{tail}")
                changed = True
                continue
        out.append(line)
    if changed:
        path.write_text("".join(out))
    return changed


def _bump_toml(path: Path, canon: str, new_ver: str) -> bool:
    """Update the floor for *canon* inside TOML string literals. Returns True if changed."""
    text = path.read_text()
    pat = re.compile(
        r'(["\'])(' + _name_pattern(canon) + r')(\[[^\]]*\])?\s*(>=|==)\s*([\d][^\s"\'\\;,]*)\1',
        re.IGNORECASE,
    )
    changed = False

    def _replace(m: re.Match) -> str:
        nonlocal changed
        quote, raw, extras, op, ver = m.group(1, 2, 3, 4, 5)
        extras = extras or ""
        need = (op == "==" and not _vlt(new_ver, ver)) or (op == ">=" and _vlt(ver, new_ver))
        if need:
            changed = True
            return f"{quote}{raw}{extras}>={new_ver}{quote}"
        return m.group(0)

    new_text = pat.sub(_replace, text)
    if changed:
        path.write_text(new_text)
    return changed


def _bump(root: Path, package: str, new_ver: str) -> int:
    if not re.match(r"^\d+(\.\d+)+$", new_ver):
        print(f"ERROR: '{new_ver}' is not a valid version number (expected X.Y.Z)")
        return 1

    if not CONSTRAINTS.exists():
        print(f"ERROR: {CONSTRAINTS} not found")
        return 1

    canon = canonicalize(package)
    floors = {d.pkg: d for d in _read_requirements(CONSTRAINTS) if d.op == ">="}

    if canon not in floors:
        print(f"ERROR: '{package}' is not tracked in {CONSTRAINTS.relative_to(root)}")
        print(f"  Tracked packages: {', '.join(sorted(floors))}")
        return 1

    floor = floors[canon]
    if _vlt(new_ver, floor.version):
        print(
            f"ERROR: {new_ver} is older than the current floor {floor.version}."
            f" Use verify to inspect the current state."
        )
        return 1
    if _veq(new_ver, floor.version):
        print(f"'{package}' is already at >={new_ver} in constraints.txt — nothing to do")
        return 0

    print(f"Bumping {package}: >={floor.version} → >={new_ver}\n")
    for path in [CONSTRAINTS, *_dep_files(root)]:
        fn = _bump_toml if path.name == "pyproject.toml" else _bump_req
        if fn(path, canon, new_ver):
            print(f"  UPDATED  {path.relative_to(root)}")

    print("\nRun `python scripts/constraints_manager.py verify` to confirm.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        prog="constraints_manager.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="Check all dep files against constraints.txt")

    bp = sub.add_parser("bump", help="Raise a package floor across all dep files")
    bp.add_argument("package", help="Package name (e.g. pydantic)")
    bp.add_argument("version", help="New minimum version (e.g. 2.7.0)")

    args = p.parse_args()
    sys.exit(
        _verify(REPO_ROOT) if args.cmd == "verify" else _bump(REPO_ROOT, args.package, args.version)
    )


if __name__ == "__main__":
    main()
