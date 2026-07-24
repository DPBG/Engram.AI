from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "profile_hotpath.py"
_SPEC = importlib.util.spec_from_file_location("profile_hotpath_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
profile_hotpath = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = profile_hotpath
_SPEC.loader.exec_module(profile_hotpath)


class FakeResource:
    RUSAGE_SELF = 0

    def __init__(self, value: float | None = None, error: Exception | None = None):
        self.value = value
        self.error = error

    def getrusage(self, who: int):
        assert who == self.RUSAGE_SELF
        if self.error is not None:
            raise self.error
        return SimpleNamespace(ru_maxrss=self.value)


@pytest.mark.parametrize(
    ("platform_name", "raw_value", "expected_mb"),
    [
        ("linux", 2048, 2.0),
        ("darwin", 2 * 1024 * 1024, 2.0),
    ],
)
def test_get_peak_rss_mb_converts_platform_units(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    raw_value: float,
    expected_mb: float,
):
    monkeypatch.setattr(profile_hotpath, "resource", FakeResource(raw_value))
    monkeypatch.setattr(profile_hotpath.sys, "platform", platform_name)

    assert profile_hotpath._get_peak_rss_mb() == expected_mb


def test_get_peak_rss_mb_returns_none_without_resource(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(profile_hotpath, "resource", None)

    assert profile_hotpath._get_peak_rss_mb() is None


def test_get_peak_rss_mb_returns_none_when_getrusage_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(profile_hotpath, "resource", FakeResource(error=OSError("unavailable")))

    assert profile_hotpath._get_peak_rss_mb() is None


def test_format_memory_summary_reports_snapshots_and_non_negative_deltas():
    snapshots = profile_hotpath.MemorySnapshots(
        before_build_mb=42.3,
        after_build_mb=615.8,
        after_warmup_mb=640.1,
        after_profile_mb=646.0,
    )

    output = profile_hotpath._format_memory_summary(snapshots)

    assert "=== MEMORY HIGH-WATER MARKS ===" in output
    assert "Before network:" in output and "42.3 MiB" in output
    assert "After network:" in output and "+573.5 MiB" in output
    assert "After warmup:" in output and "+24.3 MiB" in output
    assert "After profile:" in output and "+5.9 MiB" in output
    assert "Total observed:" in output and "+603.7 MiB" in output


def test_format_memory_summary_clamps_decreasing_synthetic_values():
    snapshots = profile_hotpath.MemorySnapshots(
        before_build_mb=100.0,
        after_build_mb=90.0,
        after_warmup_mb=80.0,
        after_profile_mb=70.0,
    )

    output = profile_hotpath._format_memory_summary(snapshots)

    assert output.count("+0.0 MiB") == 4
    assert "+-" not in output


def test_format_memory_summary_falls_back_when_any_snapshot_is_unavailable():
    snapshots = profile_hotpath.MemorySnapshots(
        before_build_mb=10.0,
        after_build_mb=20.0,
        after_warmup_mb=None,
        after_profile_mb=30.0,
    )

    assert profile_hotpath._format_memory_summary(snapshots) == (
        "=== MEMORY HIGH-WATER MARKS ===\n"
        "Peak RSS unavailable on this platform."
    )
