"""
STRATUM benchmark runner.

Executes a parameterized benchmark scenario against a running
stratum-gateway process and writes results to a Parquet file with
full metadata for reproducibility.

RESULT SCHEMA
=============
Every Parquet file produced by this runner includes:
  - One row per CO-corrected latency measurement
  - Metadata columns: scenario, run_id, git_sha, timestamp, hardware_info
  - A separate metadata sidecar JSON with the full LoadConfig

This means any benchmark result committed to benchmarks/results/ can
be reproduced on the same hardware by checking out the same git SHA
and running this script with the committed config.

USAGE
=====
    python benchmarks/harness/runner.py \
        --config benchmarks/scenarios/gateway_baseline.yaml \
        --output benchmarks/results/

COMPARISON
==========
To compare two result files:
    python benchmarks/harness/runner.py \
        --compare benchmarks/results/A.parquet benchmarks/results/B.parquet
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import numpy as np

# Harness modules (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from statistics import compute_percentiles, compare_latencies
from coordinated_omission import LoadConfig, run_load


def get_git_sha() -> str:
    """Returns the current git SHA, or 'unknown' if git is not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_scenario(path: Path) -> dict:
    """Load a benchmark scenario YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_load_config(scenario: dict) -> LoadConfig:
    """Build a LoadConfig from a scenario dict."""
    return LoadConfig(
        target_url=scenario["target_url"],
        request_body=json.dumps(scenario["request_body"]),
        arrival_rate_rps=float(scenario["arrival_rate_rps"]),
        duration_seconds=int(scenario["duration_seconds"]),
        warmup_seconds=int(scenario["warmup_seconds"]),
        auth_header=scenario.get("auth_header"),
    )


def run_scenario(scenario_path: Path, output_dir: Path) -> Path:
    """Run a benchmark scenario and write results to a Parquet file.

    Returns the path to the written Parquet file.
    """
    scenario = load_scenario(scenario_path)
    config = build_load_config(scenario)

    run_id = str(uuid.uuid4())[:8]
    git_sha = get_git_sha()
    timestamp = datetime.now(timezone.utc).isoformat()
    scenario_name = scenario.get("name", scenario_path.stem)

    print(f"\nSTRATUM Benchmark Runner")
    print(f"{'='*60}")
    print(f"Scenario:       {scenario_name}")
    print(f"Target:         {config.target_url}")
    print(f"Arrival rate:   {config.arrival_rate_rps} RPS (Poisson)")
    print(f"Duration:       {config.duration_seconds}s (+ {config.warmup_seconds}s warmup)")
    print(f"Git SHA:        {git_sha}")
    print(f"Run ID:         {run_id}")
    print(f"Host:           {socket.gethostname()}")
    print(f"Platform:       {platform.platform()}")
    print(f"CO correction:  ENABLED")
    print(f"{'='*60}")
    print(f"\nRunning warmup ({config.warmup_seconds}s)...")

    result = run_load(config)

    if not result.co_corrected_latencies_ms:
        print("ERROR: No successful measurements collected. Is the gateway running?")
        print(f"  Requests sent: {result.n_requests_sent}")
        print(f"  Errors: {result.n_requests_sent - result.n_success}")
        sys.exit(1)

    latencies = result.co_corrected_latencies_ms
    percentiles = compute_percentiles(latencies)

    print(f"\nResults ({result.n_requests_sent} requests, "
          f"{result.n_success} successful, "
          f"{result.error_rate:.1%} error rate):")
    print(f"  Actual RPS:     {result.actual_rps:.1f}")
    print(f"  CO-corrected measurements: {len(latencies)} "
          f"({len(latencies) - result.n_success} phantom)")
    print(f"\n  Latency (ms) with {int(percentiles['p50'].confidence_level * 100)}% CI:")
    print(f"  {'Metric':<8} {'Estimate':>10} {'CI Lower':>10} {'CI Upper':>10}")
    print(f"  {'-'*42}")
    for name, ci in percentiles.items():
        print(f"  {name:<8} {ci.estimate:>10.2f} {ci.lower:>10.2f} {ci.upper:>10.2f}")

    # Build Parquet table
    # Each row is one CO-corrected latency measurement
    table = pa.table(
        {
            "latency_ms": pa.array(latencies, type=pa.float64()),
            "scenario": pa.array([scenario_name] * len(latencies), type=pa.string()),
            "run_id": pa.array([run_id] * len(latencies), type=pa.string()),
            "git_sha": pa.array([git_sha] * len(latencies), type=pa.string()),
            "timestamp": pa.array([timestamp] * len(latencies), type=pa.string()),
            "hostname": pa.array([config.hostname] * len(latencies), type=pa.string()),
            "os_platform": pa.array([config.os_platform] * len(latencies), type=pa.string()),
            "arrival_rate_rps": pa.array(
                [config.arrival_rate_rps] * len(latencies), type=pa.float64()
            ),
            "duration_seconds": pa.array(
                [config.duration_seconds] * len(latencies), type=pa.int32()
            ),
            "warmup_seconds": pa.array(
                [config.warmup_seconds] * len(latencies), type=pa.int32()
            ),
            "co_correction_enabled": pa.array(
                [True] * len(latencies), type=pa.bool_()
            ),
        }
    )

    # Metadata sidecar (human-readable summary committed alongside Parquet)
    metadata = {
        "scenario": scenario_name,
        "run_id": run_id,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "hostname": config.hostname,
        "os_platform": config.os_platform,
        "config": {
            "target_url": config.target_url,
            "arrival_rate_rps": config.arrival_rate_rps,
            "duration_seconds": config.duration_seconds,
            "warmup_seconds": config.warmup_seconds,
            "auth_header": "REDACTED" if config.auth_header else None,
        },
        "summary": {
            "n_requests_sent": result.n_requests_sent,
            "n_success": result.n_success,
            "error_rate": result.error_rate,
            "actual_rps": result.actual_rps,
            "n_co_corrected_measurements": len(latencies),
            "percentiles_ms": {
                name: {
                    "estimate": ci.estimate,
                    "ci_lower": ci.lower,
                    "ci_upper": ci.upper,
                    "confidence_level": ci.confidence_level,
                }
                for name, ci in percentiles.items()
            },
        },
    }

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = scenario_name.replace(" ", "_").replace("/", "_")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parquet_path = output_dir / f"{safe_name}_{date_str}_{run_id}.parquet"
    meta_path = output_dir / f"{safe_name}_{date_str}_{run_id}_meta.json"

    pq.write_table(table, parquet_path, compression="snappy")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults written:")
    print(f"  {parquet_path}")
    print(f"  {meta_path}")

    return parquet_path


def compare_runs(path_a: Path, path_b: Path) -> None:
    """Statistical comparison of two benchmark result Parquet files.

    Treats the first file as 'treatment' and the second as 'control'.
    Reports: Welch's t-test, Cohen's d, bootstrap CI on the difference.
    """
    table_a = pq.read_table(path_a)
    table_b = pq.read_table(path_b)

    latencies_a = table_a["latency_ms"].to_pylist()
    latencies_b = table_b["latency_ms"].to_pylist()

    scenario_a = table_a["scenario"][0].as_py()
    scenario_b = table_b["scenario"][0].as_py()

    print(f"\nComparison: {path_a.name} vs {path_b.name}")
    print(f"  Treatment ({scenario_a}): {len(latencies_a)} measurements")
    print(f"  Control   ({scenario_b}): {len(latencies_b)} measurements")

    result = compare_latencies(latencies_a, latencies_b)
    print(f"\n{result}")

    # Per-percentile comparison
    percs_a = compute_percentiles(latencies_a)
    percs_b = compute_percentiles(latencies_b)
    print(f"\nPer-percentile (treatment vs control, ms):")
    print(f"  {'Metric':<8} {'Treatment':>12} {'Control':>12} {'Delta':>10}")
    print(f"  {'-'*46}")
    for name in ["p50", "p95", "p99", "p999"]:
        delta = percs_a[name].estimate - percs_b[name].estimate
        print(
            f"  {name:<8} {percs_a[name].estimate:>12.2f} "
            f"{percs_b[name].estimate:>12.2f} "
            f"{delta:>+10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STRATUM benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a benchmark scenario")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results"),
    )

    cmp_parser = subparsers.add_parser("compare", help="Compare two result files")
    cmp_parser.add_argument("treatment", type=Path)
    cmp_parser.add_argument("control", type=Path)

    args = parser.parse_args()

    if args.command == "run":
        run_scenario(args.config, args.output)
    elif args.command == "compare":
        compare_runs(args.treatment, args.control)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
