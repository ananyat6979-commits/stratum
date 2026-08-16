"""
Checkpointed, resumable Phase 2 runner: a live mSPRT sequential test
comparing SemanticRouter against RoundRobinRouter under real Ollama
inference, interleaved (both arms queried concurrently per pair, not
sequentially), stopping exactly when mSPRT's stopping rule fires
rather than at a pre-committed fixed sample size.

WHY THIS EXISTS, NOT compare_interleaved.py
================================================
compare_interleaved.py (Phase 1) ran a fixed duration and computed
percentiles afterward, correct for that question (routing overhead
against an unreachable worker, fast, bounded). Phase 2 asks a
different question (does real end-to-end latency differ) under severe,
already-documented variance, where a fixed sample size would have to
be guessed (see phase2_power_check.py for why that was rejected) and
could run for an unknown, possibly very long time before the real
effect (if any) separates from noise. mSPRT is the right tool
specifically because it adapts to the real data rather than requiring
the answer to "how many observations" to be known in advance.

CRASH SAFETY
============
Every completed pair is followed by an ATOMIC checkpoint write (temp
file + os.replace, which is atomic on both Windows and POSIX for
same-volume renames). A crash, kill, or unexpected exit at any point
leaves the checkpoint file as either the previous complete state or
the new complete state, never a partial write. Resuming re-reads this
checkpoint and continues from exactly n_completed_pairs, replaying the
deterministic prompt stream (see phase2_prompt_pool.py) to that exact
position rather than restarting it.

WHAT COUNTS AS ONE OBSERVATION
===================================
For prompt P (drawn from the deterministic stream), both arms
(semantic on port 8081, round_robin on port 8080, per the established
STRATUM_ROUTING_STRATEGY convention) are queried CONCURRENTLY with the
SAME prompt via asyncio.gather, each wall-clock-timed independently
from just before the request to just after the response. If EITHER
request fails or exceeds PER_REQUEST_TIMEOUT_SECONDS, the entire pair
is discarded (not recorded as zero, not recorded as a failure value,
simply not fed to mSPRT at all) and logged, so mSPRT's input stream
stays composed only of genuinely matched, valid observations. This is
a real, stated limitation: some fraction of attempted pairs will be
discarded, meaning the true number of REQUESTS sent will exceed the
number of OBSERVATIONS mSPRT sees, by an amount this script reports
but cannot know in advance.

USAGE
=====
Start (or resume, if a checkpoint already exists at --checkpoint):
    uv run --project ../../services/experiment-engine python phase2_runner.py \
        --checkpoint phase2_state.json \
        --round-robin-url http://127.0.0.1:8080/v1/chat/completions \
        --semantic-url http://127.0.0.1:8081/v1/chat/completions \
        --sigma 15.0 --tau 1.5 --alpha 0.05 --max-observations 2000

Ctrl+C at any point is safe, the current pair in flight may be lost,
nothing already checkpointed is. Re-run the identical command to
resume; a config mismatch against the existing checkpoint is a hard
error, not a warning, since silently continuing a sequential test
under a changed config invalidates mSPRT's Type-I error guarantee.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "services" / "experiment-engine" / "src"))

from phase2_prompt_pool import infinite_stream
from stratum_experiment.experiment import Experiment
from stratum_experiment.msprt import MSPRTConfig

PER_REQUEST_TIMEOUT_SECONDS = 180.0
STREAM_SEED = 20260814


@dataclass
class RunConfig:
    alpha: float
    sigma_squared: float
    tau_squared: float
    max_observations: int
    round_robin_url: str
    semantic_url: str
    stream_seed: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Checkpoint:
    config: dict
    n_completed_pairs: int
    n_skipped_pairs: int
    sum_d: float
    started_at: str
    last_updated_at: str
    should_reject_null: bool
    e_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_checkpoint(path: Path) -> Checkpoint | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return Checkpoint(**data)


def write_checkpoint_atomic(path: Path, checkpoint: Checkpoint) -> None:
    """Atomic write: write to a temp file in the same directory, then
    os.replace over the real path. os.replace is atomic on both
    Windows and POSIX when source and destination are on the same
    filesystem, which they are here since the temp file is created
    alongside the real checkpoint path.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)
    os.replace(tmp_path, path)


def config_matches(run_config: RunConfig, checkpoint_config: dict) -> tuple[bool, str]:
    """Returns (matches, mismatch_description). Checked field by field
    so a mismatch is reported precisely, not just as "configs differ."
    """
    current = run_config.to_dict()
    mismatches = []
    for key in ("alpha", "sigma_squared", "tau_squared", "round_robin_url", "semantic_url", "stream_seed"):
        if current[key] != checkpoint_config.get(key):
            mismatches.append(f"{key}: checkpoint={checkpoint_config.get(key)!r} vs requested={current[key]!r}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, ""


async def _timed_request(client: httpx.AsyncClient, url: str, body: dict) -> float | None:
    """Sends one request, returns wall-clock seconds to completion, or
    None if it failed or exceeded PER_REQUEST_TIMEOUT_SECONDS. Timeout
    and connection errors are both treated as None, a real, honest
    failure to get a matched observation, not silently retried.
    """
    start = time.monotonic()
    try:
        response = await client.post(url, json=body, timeout=PER_REQUEST_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - start
        if response.status_code != 200:
            print(f"    non-200 ({response.status_code}) from {url}, discarding pair")
            return None
        return elapsed
    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
        elapsed = time.monotonic() - start
        print(f"    request to {url} failed after {elapsed:.1f}s ({type(e).__name__}), discarding pair")
        return None


async def run(run_config: RunConfig, checkpoint_path: Path) -> None:
    existing = load_checkpoint(checkpoint_path)

    if existing is not None:
        matches, mismatch_desc = config_matches(run_config, existing.config)
        if not matches:
            print("FATAL: checkpoint config does not match requested config, refusing to resume.")
            print(f"  {mismatch_desc}")
            print("  Resuming a sequential test under a changed config invalidates its")
            print("  Type-I error guarantee. Either match the original config exactly,")
            print("  or start a new run with a different --checkpoint path.")
            sys.exit(1)
        print(f"Resuming from checkpoint: {existing.n_completed_pairs} pairs completed, "
              f"{existing.n_skipped_pairs} skipped, e_value={existing.e_value:.6f}")
    else:
        print("No existing checkpoint, starting fresh.")

    msprt_config = MSPRTConfig(
        alpha=run_config.alpha,
        sigma_squared=run_config.sigma_squared,
        tau_squared=run_config.tau_squared,
    )
    experiment = Experiment(name="phase2_routing_quality", config=msprt_config)

    n_completed = 0
    n_skipped = 0
    if existing is not None:
        # Reconstruct MSPRTState's sufficient statistics directly.
        # Feeding synthetic (sum_d/n, 0) pairs would NOT reproduce the
        # same e_value trajectory correctly for a formula this
        # nonlinear in n and sum_d together, set the fields directly
        # instead, which is exact, not an approximation.
        experiment.state.n = existing.n_completed_pairs
        experiment.state.sum_d = existing.sum_d
        n_completed = existing.n_completed_pairs
        n_skipped = existing.n_skipped_pairs

    started_at = existing.started_at if existing else _now_iso()

    stream = infinite_stream(seed=run_config.stream_seed)
    # Advance the deterministic stream to exactly where we left off.
    for _ in range(n_completed + n_skipped):
        next(stream)

    print(f"Starting from pair {n_completed + n_skipped + 1}, "
          f"current e_value={experiment.result().e_value:.6f}, "
          f"threshold={msprt_config.rejection_threshold}")

    async with httpx.AsyncClient() as client:
        while True:
            if experiment.result().should_reject_null:
                print(f"\nmSPRT REJECTED THE NULL after {n_completed} observations "
                      f"({n_skipped} pairs skipped). e_value={experiment.result().e_value:.4f} "
                      f">= threshold {msprt_config.rejection_threshold}.")
                print(f"Mean difference (semantic - round_robin, seconds): "
                      f"{experiment.result().mean_difference:.4f}")
                break
            if n_completed >= run_config.max_observations:
                print(f"\nReached max_observations={run_config.max_observations} without "
                      f"rejecting the null. Honest null result: no detected effect at this "
                      f"sample size under alpha={run_config.alpha}.")
                break

            entry = next(stream)
            body = {
                "model": "phi3:mini",
                "messages": [{"role": "user", "content": entry.prompt}],
                "max_tokens": 50,
            }

            round_robin_task = _timed_request(client, run_config.round_robin_url, body)
            semantic_task = _timed_request(client, run_config.semantic_url, body)
            round_robin_latency, semantic_latency = await asyncio.gather(round_robin_task, semantic_task)

            if round_robin_latency is None or semantic_latency is None:
                n_skipped += 1
                print(f"  pair skipped (prompt_id={entry.prompt_id}), "
                      f"n_skipped now {n_skipped}")
            else:
                experiment.record_observation(
                    treatment_value=semantic_latency, control_value=round_robin_latency
                )
                n_completed += 1
                result = experiment.result()
                print(f"  pair {n_completed}: rr={round_robin_latency:.2f}s "
                      f"sem={semantic_latency:.2f}s diff={semantic_latency - round_robin_latency:+.2f}s "
                      f"e_value={result.e_value:.4f}")

            checkpoint = Checkpoint(
                config=run_config.to_dict(),
                n_completed_pairs=n_completed,
                n_skipped_pairs=n_skipped,
                sum_d=experiment.state.sum_d,
                started_at=started_at,
                last_updated_at=_now_iso(),
                should_reject_null=experiment.result().should_reject_null,
                e_value=experiment.result().e_value,
            )
            write_checkpoint_atomic(checkpoint_path, checkpoint)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 checkpointed mSPRT routing-quality runner")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--round-robin-url", required=True)
    parser.add_argument("--semantic-url", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, required=True, help="assumed per-arm stddev, seconds")
    parser.add_argument("--tau", type=float, required=True, help="prior effect-size scale, seconds")
    parser.add_argument("--max-observations", type=int, default=2000)
    args = parser.parse_args()

    run_config = RunConfig(
        alpha=args.alpha,
        sigma_squared=2 * args.sigma**2,
        tau_squared=args.tau**2,
        max_observations=args.max_observations,
        round_robin_url=args.round_robin_url,
        semantic_url=args.semantic_url,
        stream_seed=STREAM_SEED,
    )

    asyncio.run(run(run_config, args.checkpoint))


if __name__ == "__main__":
    main()