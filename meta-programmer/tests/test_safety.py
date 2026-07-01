"""Tests for the Meta-Programmer deploy-time safety guards (Phase 1.4 + 1.5)."""

import importlib.util
import os
import sys
import tempfile

# Load safety.py directly (pure stdlib) so the test doesn't import the
# meta_programmer package __init__, which pulls in the `docker` SDK.
_SAFETY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "meta_programmer", "safety.py"
)
_spec = importlib.util.spec_from_file_location("mp_safety", _SAFETY_PATH)
_safety = importlib.util.module_from_spec(_spec)
sys.modules["mp_safety"] = _safety  # needed so @dataclass can resolve annotations
_spec.loader.exec_module(_safety)
scan_source = _safety.scan_source
is_dangerous = _safety.is_dangerous
safe_deploy_path = _safety.safe_deploy_path
deploy_atomically = _safety.deploy_atomically


# ── full-source AST taint scan (1.5) ─────────────────────────────────────────

def test_clean_code_passes():
    assert scan_source("def add(a, b):\n    return a + b\n") == []
    assert is_dangerous("x = 1 + 2\ny = [i for i in range(3)]") is False


def test_eval_flagged():
    assert is_dangerous("result = eval(user_supplied)")


def test_exec_flagged():
    assert is_dangerous("exec('import os; os.system(\\'id\\')')")


def test_os_system_flagged():
    assert is_dangerous("import os\nos.system('rm -rf /')")


def test_subprocess_call_flagged():
    assert is_dangerous("import subprocess\nsubprocess.run(['ls'])")


def test_from_import_sink_call_flagged():
    # `from os import system; system(...)` is exactly as dangerous as
    # `os.system(...)` and must not bypass the attribute-form call check.
    assert is_dangerous("from os import system\nsystem('rm -rf /')")
    assert is_dangerous("from subprocess import run\nrun(['ls'])")
    assert is_dangerous("from os import popen\npopen('id')")


def test_from_import_sink_aliased_call_flagged():
    assert is_dangerous("from os import system as s\ns('rm -rf /')")
    assert is_dangerous("from subprocess import Popen as P\nP(['ls'])")


def test_star_import_of_dangerous_module_flagged():
    assert is_dangerous("from os import *\nsystem('id')")
    assert is_dangerous("from subprocess import *")


def test_from_import_benign_name_not_flagged_until_called():
    # `from os import getcwd` is a benign sink-free name; importing it (medium at
    # most) must not be high-severity on its own.
    assert is_dangerous("from os import getcwd\np = getcwd()") is False


def test_socket_import_flagged():
    assert is_dangerous("import socket\ns = socket.socket()")


def test_dunder_globals_access_flagged():
    assert is_dangerous("def f():\n    pass\ng = f.__globals__")


def test_getattr_dunder_flagged():
    assert is_dangerous("o = getattr(obj, '__subclasses__')")


def test_self_referential_import_flagged():
    assert is_dangerous("from kernel.evaluator import KernelEvaluator")
    assert is_dangerous("import meta_programmer.service")


def test_syntax_error_is_high_severity():
    assert is_dangerous("def broken(:\n    pass")


def test_benign_os_path_is_not_high_severity():
    # `import os` is only medium; os.path.join is not a dangerous sink.
    assert is_dangerous("import os\np = os.path.join('a', 'b')") is False


# ── `from <module> import <sink>` bypass (#118) ───────────────────────────────
# The attribute form (`import os` + `os.system`) was caught, but a sink pulled in
# via `from os import system` is called as a bare Name and used to sail through.

def test_from_import_sink_call_flagged():
    assert is_dangerous("from os import system\nsystem('rm -rf /')")


def test_from_import_subprocess_call_flagged():
    assert is_dangerous("from subprocess import run\nrun(['ls'])")


def test_from_import_aliased_sink_call_flagged():
    assert is_dangerous("from os import popen as p\np('id')")


def test_star_import_from_dangerous_module_flagged():
    # A star import pulls every sink in wholesale and defeats name tracking.
    assert is_dangerous("from os import *\nsystem('id')")
    assert is_dangerous("from subprocess import *")


def test_from_import_dangerous_call_is_high_and_named():
    # Symmetric with the attribute-form finding: high severity, "mod.attr()" detail.
    findings = scan_source("from os import system\nsystem('id')")
    high = [f for f in findings if f.severity == "high"]
    assert high
    assert any(f.rule == "dangerous_call" and f.detail == "os.system()" for f in high)


def test_benign_from_import_not_high_severity():
    # Names outside the curated os sink set must NOT be tainted, so the resolver
    # stays precise and doesn't over-block legitimate generated code.
    assert is_dangerous("from os import getcwd\nprint(getcwd())") is False
    assert is_dangerous("from os.path import join\np = join('a', 'b')") is False


def test_aliased_local_binding_of_sink_flagged():
    # `from os import system as s; runner = s; runner(...)` — taint must propagate
    # through a simple local rebinding or the call slips through (fail-open).
    assert is_dangerous("from os import system as s\nrunner = s\nrunner('id')")


def test_alias_chain_of_sink_flagged():
    assert is_dangerous("from os import system\na = system\nb = a\nb('id')")


def test_aliased_local_binding_of_builtin_flagged():
    # Builtin sinks must also survive aliasing: `e = eval; e(...)`.
    assert is_dangerous("e = eval\ne('1 + 1')")


def test_benign_local_alias_not_flagged():
    # A rebinding of a harmless name must not become a high finding.
    assert is_dangerous("runner = print\nrunner('hello')") is False


# ── deploy-path allowlist + traversal/symlink protection (1.4) ────────────────

def test_path_inside_allowlist_accepted():
    with tempfile.TemporaryDirectory() as d:
        allowed = os.path.join(d, "plugins")
        os.makedirs(allowed)
        ok, resolved = safe_deploy_path(os.path.join(allowed, "driver.py"), allowlist=[allowed])
        assert ok is True
        assert resolved.endswith("driver.py")


def test_path_outside_allowlist_rejected():
    with tempfile.TemporaryDirectory() as d:
        allowed = os.path.join(d, "plugins")
        os.makedirs(allowed)
        ok, _ = safe_deploy_path(os.path.join(d, "etc", "passwd"), allowlist=[allowed])
        assert ok is False


def test_dotdot_traversal_rejected():
    with tempfile.TemporaryDirectory() as d:
        allowed = os.path.join(d, "plugins")
        os.makedirs(allowed)
        escaping = os.path.join(allowed, "..", "evil.py")  # resolves outside `allowed`
        ok, _ = safe_deploy_path(escaping, allowlist=[allowed])
        assert ok is False


def test_empty_path_rejected():
    ok, _ = safe_deploy_path("", allowlist=["/data/plugins"])
    assert ok is False


# ── atomic deploy with rollback (1.9) ─────────────────────────────────────────

def test_deploy_new_valid_file():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "plugin.py")
        ok, detail = deploy_atomically(target, "x = 1\n")
        assert ok is True
        with open(target) as f:
            assert f.read() == "x = 1\n"


def test_deploy_syntax_error_to_new_path_leaves_nothing():
    # A broken artifact must not survive: the newly-created file is removed.
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "broken.py")
        ok, detail = deploy_atomically(target, "def broken(:\n    pass")
        assert ok is False
        assert "syntax" in detail.lower()
        assert not os.path.exists(target)


def test_deploy_syntax_error_over_existing_restores_original():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "live.py")
        with open(target, "w") as f:
            f.write("GOOD = True\n")
        ok, _ = deploy_atomically(target, "def broken(:\n    pass")
        assert ok is False
        # Original content must be restored intact.
        with open(target) as f:
            assert f.read() == "GOOD = True\n"


def test_deploy_valid_over_existing_replaces():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "live.py")
        with open(target, "w") as f:
            f.write("OLD = 1\n")
        ok, _ = deploy_atomically(target, "NEW = 2\n")
        assert ok is True
        with open(target) as f:
            assert f.read() == "NEW = 2\n"
