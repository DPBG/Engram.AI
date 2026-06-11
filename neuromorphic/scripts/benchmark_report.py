"""
Benchmark comparison report — compares two benchmark result files.

Usage:
    cd neuromorphic && uv run python scripts/benchmark_report.py benchmarks/before.json benchmarks/after.json
    cd neuromorphic && uv run python scripts/benchmark_report.py benchmarks/  # compare latest two
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def pct_change(before: float, after: float) -> str:
    if before == 0:
        return "N/A"
    change = (after - before) / abs(before) * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}%"


def compare(before: dict, after: dict) -> None:
    print("=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"Before: {before.get('timestamp', '?')}")
    print(f"After:  {after.get('timestamp', '?')}")
    print()

    # Config
    bc = before.get("config", {})
    ac = after.get("config", {})
    print("--- Configuration ---")
    print(f"  Neurons:   {bc.get('total_neurons', '?'):>12,}  ->  {ac.get('total_neurons', '?'):>12,}")
    print(f"  Synapses:  {bc.get('total_synapses', '?'):>12,}  ->  {ac.get('total_synapses', '?'):>12,}")
    print()

    # Speed
    bs = before.get("speed", {})
    as_ = after.get("speed", {})
    print("--- Speed ---")
    bsps = bs.get("steps_per_sec", 0)
    asps = as_.get("steps_per_sec", 0)
    print(f"  Steps/sec: {bsps:>10.1f}  ->  {asps:>10.1f}  ({pct_change(bsps, asps)})")
    print()

    # Learning
    bl = before.get("learning", {})
    al = after.get("learning", {})
    print("--- Learning ---")
    blsps = bl.get("steps_per_sec", 0)
    alsps = al.get("steps_per_sec", 0)
    print(f"  Learn steps/sec: {blsps:>8.1f}  ->  {alsps:>8.1f}  ({pct_change(blsps, alsps)})")

    bcc = bl.get("concept_count", 0)
    acc = al.get("concept_count", 0)
    print(f"  Concept count:   {bcc:>8}  ->  {acc:>8}")

    # Weight changes comparison
    bwc = bl.get("weight_changes", {})
    awc = al.get("weight_changes", {})
    all_groups = set(list(bwc.keys()) + list(awc.keys()))
    if all_groups:
        print("  Weight deltas:")
        for name in sorted(all_groups):
            bd = bwc.get(name, {}).get("delta", 0)
            ad = awc.get(name, {}).get("delta", 0)
            print(f"    {name:30s}: {bd:>+.6f}  ->  {ad:>+.6f}")
    print()

    # Memory
    bm = before.get("memory", {})
    am = after.get("memory", {})
    print("--- Memory ---")
    brss = bm.get("peak_rss_mb", 0)
    arss = am.get("peak_rss_mb", 0)
    print(f"  Peak RSS (MB): {brss:>8.0f}  ->  {arss:>8.0f}  ({pct_change(brss, arss)})")

    bsyn = bm.get("total_synapses", 0)
    asyn = am.get("total_synapses", 0)
    print(f"  Synapses:      {bsyn:>8,}  ->  {asyn:>8,}  ({pct_change(bsyn, asyn)})")
    print()

    # Phase
    bf = before.get("final_state", {})
    af = after.get("final_state", {})
    print("--- Phase ---")
    print(f"  Phase:  {bf.get('phase', '?'):>12}  ->  {af.get('phase', '?'):>12}")
    print(f"  Steps:  {bf.get('step_count', 0):>12,}  ->  {af.get('step_count', 0):>12,}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("paths", nargs="+", help="Two JSON files, or a directory (uses latest two)")
    args = parser.parse_args()

    if len(args.paths) == 1 and Path(args.paths[0]).is_dir():
        # Find latest two results in directory
        files = sorted(Path(args.paths[0]).glob("benchmark_*.json"))
        if len(files) < 2:
            print("Need at least 2 benchmark results to compare.")
            sys.exit(1)
        before_path, after_path = files[-2], files[-1]
    elif len(args.paths) >= 2:
        before_path = Path(args.paths[0])
        after_path = Path(args.paths[1])
    else:
        print("Provide two JSON files or a directory with benchmark results.")
        sys.exit(1)

    print(f"Comparing: {before_path.name} vs {after_path.name}")
    before = load_result(before_path)
    after = load_result(after_path)
    compare(before, after)


if __name__ == "__main__":
    main()
