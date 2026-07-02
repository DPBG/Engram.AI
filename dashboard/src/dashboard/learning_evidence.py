"""Load neuromorphic benchmark JSON and extract learning-evidence time series.

Reads ``neuromorphic/benchmarks/*.json`` (and optional ``/data/benchmarks``)
and normalizes metrics for the dashboard Learning Evidence panel.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_benchmark_dirs() -> list[Path]:
    """Return benchmark directories to scan, in priority order."""
    dirs: list[Path] = []
    env_dir = os.environ.get("ENGRAM_BENCHMARK_DIR", "").strip()
    if env_dir:
        dirs.append(Path(env_dir))
    # Repo-relative path (local dev: dashboard/src/dashboard -> repo root)
    repo_root = Path(__file__).resolve().parents[3]
    dirs.append(repo_root / "neuromorphic" / "benchmarks")
    dirs.append(Path("/data/benchmarks"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        resolved = d.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(d)
    return unique


def _parse_timestamp(raw: dict[str, Any], filename: str) -> str:
    ts = raw.get("timestamp")
    if isinstance(ts, str) and ts:
        return ts
    # benchmark_20260304_041527.json or benchmarks_20260304_041527.json
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        d, t = m.group(1), m.group(2)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_learning_metrics(raw: dict[str, Any]) -> dict[str, float | int | None]:
    """Pull normalized learning-evidence scalars from one benchmark JSON blob."""
    assoc = raw.get("association_strength") or {}
    binding_acc = raw.get("cross_modal_binding_accuracy") or {}
    cm_recall = raw.get("cross_modal_recall") or {}
    learning = raw.get("learning") or {}

    concept_sep = assoc.get("concept_count")
    if concept_sep is None:
        concept_sep = learning.get("concept_count")

    binding_f1 = binding_acc.get("f1")
    if binding_f1 is None:
        v2a = _safe_float(cm_recall.get("visual_to_auditory_recall"))
        a2v = _safe_float(cm_recall.get("auditory_to_visual_recall"))
        if v2a is not None and a2v is not None:
            binding_f1 = (v2a + a2v) / 2.0
        elif v2a is not None:
            binding_f1 = v2a
        elif a2v is not None:
            binding_f1 = a2v
        else:
            binding_f1 = _safe_float(cm_recall.get("binding_strength_after"))
            if binding_f1 is None:
                binding_f1 = _safe_float(cm_recall.get("binding_strength_delta"))

    return {
        "concept_separability": _safe_float(concept_sep),
        "binding_accuracy": _safe_float(binding_f1),
        "binding_precision": _safe_float(binding_acc.get("precision")),
        "binding_recall": _safe_float(binding_acc.get("recall")),
        "step_count": raw.get("step_count") or raw.get("final_state", {}).get("step_count"),
        "total_neurons": raw.get("total_neurons") or raw.get("config", {}).get("total_neurons"),
    }


def load_benchmark_files(
    limit: int = 50,
    benchmark_dirs: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Load benchmark JSON files from disk.

    Returns (entries, error). Each entry has timestamp, filename, source_dir, metrics.
    """
    dirs = benchmark_dirs or resolve_benchmark_dirs()
    found: list[tuple[str, Path, dict[str, Any]]] = []
    scanned_dirs: list[str] = []

    for directory in dirs:
        if not directory.is_dir():
            continue
        scanned_dirs.append(str(directory))
        for path in sorted(directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            found.append((path.name, directory, raw))

    if not found:
        if not scanned_dirs:
            return [], "No benchmark directories found"
        return [], f"No benchmark JSON files in: {', '.join(scanned_dirs)}"

    found.sort(key=lambda item: _parse_timestamp(item[2], item[0]))
    entries: list[dict[str, Any]] = []
    for filename, directory, raw in found[-limit:]:
        entries.append({
            "timestamp": _parse_timestamp(raw, filename),
            "filename": filename,
            "source_dir": str(directory),
            "metrics": extract_learning_metrics(raw),
            "raw_keys": sorted(raw.keys()),
        })
    return entries, None


def build_learning_evidence_series(
    limit: int = 50,
    benchmark_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Build API payload with trend series for the dashboard panel."""
    entries, error = load_benchmark_files(limit=limit, benchmark_dirs=benchmark_dirs)
    series = {
        "concept_separability": [],
        "binding_accuracy": [],
    }
    for entry in entries:
        m = entry["metrics"]
        point_base = {"timestamp": entry["timestamp"], "filename": entry["filename"]}
        if m.get("concept_separability") is not None:
            series["concept_separability"].append({
                **point_base,
                "value": m["concept_separability"],
            })
        if m.get("binding_accuracy") is not None:
            series["binding_accuracy"].append({
                **point_base,
                "value": m["binding_accuracy"],
            })

    latest = entries[-1] if entries else None
    return {
        "count": len(entries),
        "entries": entries,
        "series": series,
        "latest": latest,
        "benchmark_dirs": [str(d) for d in (benchmark_dirs or resolve_benchmark_dirs())],
        "error": error,
    }
