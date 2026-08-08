"""
Interleaved comparative load runner: SemanticRouter vs RoundRobinRouter.

WHY INTERLEAVED, NOT TWO SEPARATE RUNS
========================================
This machine has documented, severe latency variance from background
load/thermal/scheduling noise (2.8s-337s observed for the identical
prompt against real inference, see benchmarks/README.md and skills.md).
Two sequential runs: benchmark A, then benchmark B, would each be
exposed to whatever the machine happened to be doing during that run's
window, and a real difference between strategies could easily be
swamped by, or mistaken for, a difference in background load between
the two windows.

Running both load generators CONCURRENTLY against two gateway
instances (see stratum-gateway's STRATUM_ROUTING_STRATEGY /
STRATUM_GATEWAY_PORT env vars) means both strategies experience
approximately the same noise floor at approximately the same moments.
The DIFFERENCE between them remains a meaningful signal even when the
absolute numbers are noisy, this is the same logic as a paired
statistical test versus two independent-sample tests: pairing removes
a shared source of variance from the comparison.

HOW INTERLEAVING IS ACHIEVED
==============================
Not by alternating targets within a single sequential loop, that
would just time-shift one strategy's requests relative to the other,
not truly interleave them. Instead, two independent
coordinated_omission.run_load() calls run CONCURRENTLY via
asyncio.gather(), each maintaining its own Poisson arrival schedule
against its own target. Python's asyncio event loop genuinely
interleaves the awaited I/O between them. This required zero changes
to coordinated_omission.py, run_load() and its CO correction are
reused exactly as they are, proven correct by the existing baseline
benchmark's retrospective (see gateway_dispatch_round_robin_baseline.yaml's
documented scheduling-drift incident, which this script's use of the
same run_load() automatically inherits protection against).

WHAT THIS MEASURES (SCOPE, READ BEFORE INTERPRETING RESULTS)
================================================================
Both gateway instances point at an unreachable worker
(http://127.0.0.1:0, expected_status_codes=[502]), same as
gateway_dispatch_round_robin_baseline.yaml. This is deliberate: it
isolates ROUTING OVERHEAD (SemanticRouter's embedding computation +
cache-hit-index lookup + scoring, versus RoundRobinRouter's atomic
increment) from real Ollama inference latency, which is a separate,
much larger, and much noisier measurement this script does NOT
attempt. See docs/SCOPE.md for that distinction.

USAGE
=====
Prerequisite: two stratum-gateway instances already running,

    # Terminal 1
    $env:STRATUM_ROUTING_STRATEGY="round_robin"; $env:STRATUM_GATEWAY_PORT="8080"
    $env:STRATUM_EVENT_LOG_PATH="gw_rr.redb"; $env:STRATUM_WORKER_0_URL="http://127.0.0.1:0"
    cargo run

    # Terminal 2
    $env:STRATUM_ROUTING_STRATEGY="semantic"; $env:STRATUM_GATEWAY_PORT="8081"
    $env:STRATUM_EVENT_LOG_PATH="gw_sem.redb"; $env:STRATUM_WORKER_0_URL="http://127.0.0.1:0"
    cargo run

Then:
    python benchmarks/harness/compare_interleaved.py \
        --config benchmarks/scenarios/semantic_vs_round_robin.yaml \
        --output benchmarks/results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from statistics import compute_percentiles, compare_latencies
from coordinated_omission import LoadConfig, _run_load_async, LoadResult
from runner import get_git_sha


def build_load_config(scenario: dict, variant: dict) -> LoadConfig:
    """Builds a LoadConfig for one arm (variant) of the comparison.

    Shared fields (arrival_rate_rps, duration_seconds, warmup_seconds,
    request_body, expected_status_codes) come from the top-level
    scenario, so both arms run under IDENTICAL load parameters,
    only target_url differs. This is what makes the comparison fair:
    neither arm gets an easier workload than the other.
    """
    expected_codes = scenario.get("expected_status_codes")
    return LoadConfig(
        target_url=variant["target_url"],
        request_body=json.dumps(scenario["request_body"]),
        arrival_rate_rps=float(scenario["arrival_rate_rps"]),
        duration_seconds=int(scenario["duration_seconds"]),
        warmup_seconds=int(scenario["warmup_seconds"]),
        auth_header=scenario.get("auth_header"),
        expected_status_codes=set(expected_codes) if expected_codes else None,
    )


async def _run_both_concurrently(
    config_a: LoadConfig, config_b: LoadConfig
) -> tuple[LoadResult, LoadResult]:
    """Runs both load generators concurrently, returns (result_a, result_b).

    Calls coordinated_omission's internal _run_load_async() directly,
    NOT the public run_load() wrapper, run_load() calls asyncio.run()
    itself, which cannot be nested inside an event loop that's already
    running (this function is itself a coroutine, awaited from
    run_comparison() via asyncio.run()). asyncio.gather runs both
    coroutines on the SAME already-running event loop; each maintains
    its own independent Poisson schedule against its own target, and
    the awaited I/O genuinely interleaves between them. This is the
    mechanism the module docstring above describes.
    """
    return await asyncio.gather(_run_load_async(config_a), _run_load_async(config_b))


def write_result(
    result: LoadResult,
    label: str,
    scenario_name: str,
    run_id: str,
    git_sha: str,
    timestamp: str,
    output_dir: Path,
) -> tuple[Path, dict]:
    """Writes one arm's result to Parquet, same schema as runner.py's
    run_scenario(), so existing tooling (runner.py, compare) can read
    these files too, not just this script's own comparison.
    """
    latencies = result.co_corrected_latencies_ms
    percentiles = compute_percentiles(latencies) if latencies else {}

    table = pa.table(
        {
            "latency_ms": pa.array(latencies, type=pa.float64()),
            "scenario": pa.array([f"{scenario_name}_{label}"] * len(latencies), type=pa.string()),
            "run_id": pa.array([run_id] * len(latencies), type=pa.string()),
            "git_sha": pa.array([git_sha] * len(latencies), type=pa.string()),
            "timestamp": pa.array([timestamp] * len(latencies), type=pa.string()),
            "hostname": pa.array([result.config.hostname] * len(latencies), type=pa.string()),
            "os_platform": pa.array([result.config.os_platform] * len(latencies), type=pa.string()),
            "arrival_rate_rps": pa.array(
                [result.config.arrival_rate_rps] * len(latencies), type=pa.float64()
            ),
            "duration_seconds": pa.array(
                [result.config.duration_seconds] * len(latencies), type=pa.int32()
            ),
            "warmup_seconds": pa.array(
                [result.config.warmup_seconds] * len(latencies), type=pa.int32()
            ),
            "co_correction_enabled": pa.array([True] * len(latencies), type=pa.bool_()),
        }
    )

    safe_name = f"{scenario_name}_{label}".replace(" ", "_").replace("/", "_")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parquet_path = output_dir / f"{safe_name}_{date_str}_{run_id}.parquet"
    pq.write_table(table, parquet_path, compression="snappy")

    summary = {
        "label": label,
        "target_url": result.config.target_url,
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
            }
            for name, ci in percentiles.items()
        },
        "parquet_path": str(parquet_path),
    }
    return parquet_path, summary


def run_comparison(scenario_path: Path, output_dir: Path) -> None:
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    scenario_name = scenario.get("name", scenario_path.stem)
    variants = scenario["variants"]
    if len(variants) != 2:
        print(
            f"ERROR: expected exactly 2 variants (treatment, control), "
            f"got {len(variants)}. This script compares exactly two arms."
        )
        sys.exit(1)

    config_a = build_load_config(scenario, variants[0])
    config_b = build_load_config(scenario, variants[1])

    run_id = str(uuid.uuid4())[:8]
    git_sha = get_git_sha()
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"\nSTRATUM Interleaved Comparison")
    print(f"{'='*60}")
    print(f"Scenario:       {scenario_name}")
    print(f"Arm A ({variants[0]['label']}): {config_a.target_url}")
    print(f"Arm B ({variants[1]['label']}): {config_b.target_url}")
    print(f"Arrival rate:   {config_a.arrival_rate_rps} RPS per arm (Poisson, independent schedules)")
    print(f"Duration:       {config_a.duration_seconds}s (+ {config_a.warmup_seconds}s warmup)")
    print(f"Git SHA:        {git_sha}")
    print(f"Run ID:         {run_id}")
    print(f"{'='*60}")
    print(
        f"\nRunning both arms CONCURRENTLY (not sequentially) so both "
        f"experience the same background-load noise floor...\n"
    )

    result_a, result_b = asyncio.run(_run_both_concurrently(config_a, config_b))

    for label, result in [(variants[0]["label"], result_a), (variants[1]["label"], result_b)]:
        if not result.co_corrected_latencies_ms:
            print(
                f"ERROR: arm '{label}' collected zero successful measurements. "
                f"Is the gateway running at {result.config.target_url}?"
            )
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    path_a, summary_a = write_result(
        result_a, variants[0]["label"], scenario_name, run_id, git_sha, timestamp, output_dir
    )
    path_b, summary_b = write_result(
        result_b, variants[1]["label"], scenario_name, run_id, git_sha, timestamp, output_dir
    )

    print(f"\nResults, arm '{variants[0]['label']}':")
    print(f"  requests={summary_a['n_requests_sent']} success={summary_a['n_success']} "
          f"error_rate={summary_a['error_rate']:.1%} actual_rps={summary_a['actual_rps']:.2f}")
    print(f"\nResults, arm '{variants[1]['label']}':")
    print(f"  requests={summary_b['n_requests_sent']} success={summary_b['n_success']} "
          f"error_rate={summary_b['error_rate']:.1%} actual_rps={summary_b['actual_rps']:.2f}")

    print(f"\nPer-percentile ({variants[0]['label']} vs {variants[1]['label']}, ms):")
    print(f"  {'Metric':<8} {variants[0]['label']:>14} {variants[1]['label']:>14} {'Delta':>10}")
    print(f"  {'-'*52}")
    percs_a = compute_percentiles(result_a.co_corrected_latencies_ms)
    percs_b = compute_percentiles(result_b.co_corrected_latencies_ms)
    for name in ["p50", "p95", "p99", "p999"]:
        delta = percs_a[name].estimate - percs_b[name].estimate
        print(
            f"  {name:<8} {percs_a[name].estimate:>14.2f} "
            f"{percs_b[name].estimate:>14.2f} {delta:>+10.2f}"
        )

    stat_result = compare_latencies(
        result_a.co_corrected_latencies_ms, result_b.co_corrected_latencies_ms
    )
    print(f"\nStatistical comparison:\n{stat_result}")

    meta_path = output_dir / f"{scenario_name}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_{run_id}_meta.json"
    metadata = {
        "scenario": scenario_name,
        "run_id": run_id,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "interleaved": True,
        "arms": [summary_a, summary_b],
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="STRATUM interleaved comparison runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    run_comparison(args.config, args.output)


if __name__ == "__main__":
    main()