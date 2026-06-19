"""Tests for the launcher service registry.

Loads registry.py directly via importlib so the test doesn't import the
`launcher` package (which pulls in the supervisor / NATS bootstrap). The
registry is pure stdlib (dataclasses + pathlib).
"""

import importlib.util
import os
import sys

_REG_PATH = os.path.join(os.path.dirname(__file__), "..", "registry.py")
_spec = importlib.util.spec_from_file_location("launcher_registry", _REG_PATH)
registry = importlib.util.module_from_spec(_spec)
sys.modules["launcher_registry"] = registry
_spec.loader.exec_module(registry)

SERVICES = registry.SERVICES
PROFILES = registry.PROFILES
services_for_profile = registry.services_for_profile
get_service = registry.get_service


# ── profile selection ─────────────────────────────────────────────────────────

def test_core_profile_only_returns_core_services():
    core = services_for_profile("core")
    assert core, "core profile should not be empty"
    assert all(s.profile == "core" for s in core)


def test_full_profile_is_a_superset_of_core():
    core = set(s.name for s in services_for_profile("core"))
    full = set(s.name for s in services_for_profile("full"))
    assert core
    assert core <= full
    # full adds at least one non-core service
    assert full - core


def test_all_profile_includes_extra_services():
    full = set(s.name for s in services_for_profile("full"))
    every = set(s.name for s in services_for_profile("all"))
    assert full <= every
    assert any(s.profile == "extra" for s in services_for_profile("all"))


def test_unknown_profile_raises_value_error_listing_choices():
    try:
        services_for_profile("does-not-exist")
    except ValueError as exc:
        msg = str(exc)
        assert "does-not-exist" in msg
        for name in PROFILES:
            assert name in msg
    else:
        raise AssertionError("expected ValueError for unknown profile")


def test_profiles_map_has_expected_keys():
    assert set(PROFILES) == {"core", "full", "all"}


# ── service lookup ────────────────────────────────────────────────────────────

def test_get_service_returns_matching_service():
    svc = get_service("kernel")
    assert svc is not None
    assert svc.module == "kernel.service"
    assert svc.profile == "core"


def test_get_service_unknown_returns_none():
    assert get_service("not-a-real-service") is None


# ── registry invariants ───────────────────────────────────────────────────────

def test_service_names_are_unique():
    names = [s.name for s in SERVICES]
    assert len(names) == len(set(names))


def test_every_service_has_a_known_profile():
    assert all(s.profile in {"core", "full", "extra"} for s in SERVICES)


def test_governance_services_are_ordered_first():
    names = [s.name for s in SERVICES]
    # kernel (the gate) must come before the producers it governs
    assert names.index("kernel") < names.index("neuromorphic")
    assert names.index("kernel") < names.index("dashboard")


# ── Service path helpers ──────────────────────────────────────────────────────

def test_src_path_is_rooted_under_repo_root():
    svc = get_service("planner")
    assert svc.src_path == registry.ROOT / "planner/src"


def test_pythonpath_includes_service_src_and_sdk_src():
    svc = get_service("planner")
    pp = svc.pythonpath()
    parts = pp.split(os.pathsep)
    assert str(registry.ROOT / "planner/src") in parts
    assert str(registry.SDK_SRC) in parts


def test_service_is_frozen_dataclass():
    svc = get_service("kernel")
    try:
        svc.name = "mutated"
    except Exception as exc:
        assert "frozen" in str(exc).lower() or exc.__class__.__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("Service should be immutable (frozen dataclass)")
