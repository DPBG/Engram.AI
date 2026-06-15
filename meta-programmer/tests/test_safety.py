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
