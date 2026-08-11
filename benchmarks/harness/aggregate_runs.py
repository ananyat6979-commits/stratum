"""
Aggregates every committed run of a given scenario into one summary
table, so a claim like docs/SCOPE.md's "19 clean runs, p50 range
46-203ms" is independently checkable by running this script against
benchmarks/results/, not just trusted from prose.

This exists because of a real mistake: a prior conclusion in this
project's history (see docs/SCOPE.md's correction to the
semantic_vs_round_robin resolution) was drawn from 6 of what turned
out to be 22 committed runs of the same scenario, because there was
no single place to see all of them at once, only a conversational
running tally that undercounted. This script is the fix for that
class of mistake: read the actual committed files, every time.

USAGE
=====
    uv run python aggregate_runs.py --scenario semantic_vs_round_robin \
        --results-dir ../../benchmarks/results/
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import sys
from pathlib import Path

# benchmarks/harness/statistics.py (this project's own module, holding
# compute_percentiles and compare_latencies for compare_interleaved.py)
# shadows the Python standard library's statistics module, since a
# script's own directory takes import precedence. The only reliable fix
# within this directory is removing this directory from sys.path for the
# one import that needs the stdlib module, then restoring it immediately
# after.
_harness_dir = sys.path[0]
sys.path.remove(_harness_dir)
try:
    _stdlib_statistics = importlib.import_module("statistics")
finally:
    sys.path.insert(0, _harness_dir)
stdlib_median = _stdlib_statistics.median


def load_runs(scenario: str, results_dir: Path) -> list[dict]:
    pattern = str(results_dir / f"{scenario}_*_meta.json")
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate all committed runs of one scenario")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--min-success",
        type=int,
        default=None,
        help="Exclude a run from the clean-run summary if either arm's "
        "n_success falls below this. Excluded runs are still listed, "
        "just marked and left out of the aggregate stats, never silently "
        "dropped.",
    )
    args = parser.parse_args()

    runs = load_runs(args.scenario, args.results_dir)
    if not runs:
        print(f"No runs found for scenario '{args.scenario}' in {args.results_dir}")
        return

    print(f"\n{len(runs)} total committed runs for '{args.scenario}'\n")
    print(f"{'run_id':<10} {'timestamp':<20} {'arm':<14} {'n_req':>6} {'n_succ':>7} {'p50_ms':>10}  clean?")
    print("-" * 78)

    clean_p50s: dict[str, list[float]] = {}
    excluded_count = 0

    for run in runs:
        run_id = run.get("run_id", "?")
        ts = run.get("timestamp", "?")[:19]
        arms = run.get("arms", [])
        arm_success = {a["label"]: a.get("n_success", 0) for a in arms}
        is_clean = args.min_success is None or all(
            n >= args.min_success for n in arm_success.values()
        )
        if not is_clean:
            excluded_count += 1
        for arm in arms:
            label = arm.get("label", "?")
            p50 = arm.get("percentiles_ms", {}).get("p50", {}).get("estimate")
            n_req = arm.get("n_requests_sent")
            n_succ = arm.get("n_success")
            marker = "yes" if is_clean else "NO (excluded below)"
            p50_str = f"{p50:.2f}" if p50 is not None else "n/a"
            print(f"{run_id:<10} {ts:<20} {label:<14} {n_req:>6} {n_succ:>7} {p50_str:>10}  {marker}")
            if is_clean and p50 is not None:
                clean_p50s.setdefault(label, []).append(p50)

    print()
    if args.min_success is not None:
        print(f"{excluded_count} run(s) excluded from aggregate stats (min_success={args.min_success})")
    print(f"\nAggregate p50 (ms), clean runs only:")
    for label, values in clean_p50s.items():
        values_sorted = sorted(values)
        print(
            f"  {label:<14} n={len(values_sorted):>3}  "
            f"min={values_sorted[0]:>7.2f}  median={stdlib_median(values_sorted):>7.2f}  "
            f"max={values_sorted[-1]:>7.2f}"
        )
        print(f"    sorted: {values_sorted}")


if __name__ == "__main__":
    main()