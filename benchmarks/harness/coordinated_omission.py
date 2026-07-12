"""
Coordinated omission corrected HTTP load generator.

WHAT IS COORDINATED OMISSION
=============================
Standard load generators (wrk, locust in default mode, ab) exhibit
coordinated omission: when the server is slow, the generator slows down
its request rate to match, which means slow responses are undersampled
in the latency distribution. The measured P99 appears lower than the
actual P99 under sustained load because the measurement is biased toward
only sampling latency when the server is fast.

Example: server stalls for 1 second. Standard generator: stops sending
requests during the stall, measures 1 response at 1000ms, reports 1000ms
as an outlier. CO-corrected generator: records that it *would have* sent
10 requests during the stall (at 10 RPS), each of which would have
experienced the full 1000ms wait, synthesizes 10 measurements of 1000ms.

Gil Tene named and formalized this in 2015. HdrHistogram implements the
correction. wrk2 implements it. This module implements it in pure Python
against our gateway endpoint.

REFERENCE
=========
Tene, G. (2015). "How NOT to measure latency."
https://www.infoq.com/presentations/latency-pitfalls
"""

from __future__ import annotations

import asyncio
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class RawMeasurement:
    """A single HTTP request measurement with CO correction metadata.

    intended_issue_time_s: when the request SHOULD have been issued
        (based on Poisson schedule), used for CO correction.
    actual_response_time_s: when the response actually completed.
    status_code: HTTP response status.
    error: if set, the request failed with this message (no status_code).
    expected_status_codes: if set, overrides the default is_success check
        so a benchmark can deliberately measure a non-2xx outcome (e.g.
        dispatch-failure latency, where every response is a legitimate,
        expected 502).
    """

    intended_issue_time_s: float
    actual_response_time_s: float
    status_code: Optional[int]
    error: Optional[str] = None
    expected_status_codes: Optional[set[int]] = None

    @property
    def observed_latency_ms(self) -> float:
        """Latency as seen by the client (may be artificially low under CO)."""
        return (self.actual_response_time_s - self.intended_issue_time_s) * 1000.0

    @property
    def is_success(self) -> bool:
        """
        True if this measurement should be included in latency statistics.

        Default: any status code < 500 (standard "not a server error").
        Overridden per-scenario via LoadConfig.expected_status_codes when
        a benchmark is deliberately measuring a non-2xx outcome (e.g.
        dispatch-failure latency, where every response is a legitimate,
        expected 502, see gateway_dispatch_round_robin_baseline.yaml).
        """
        if self.status_code is None:
            return False
        if self.expected_status_codes is not None:
            return self.status_code in self.expected_status_codes
        return self.status_code < 500


@dataclass
class LoadResult:
    """All measurements from a single load generation run.

    co_corrected_latencies_ms contains the corrected latency set
    used for all statistical analysis. raw_measurements is retained
    for debugging and auditing.
    """

    config: "LoadConfig"
    raw_measurements: list[RawMeasurement] = field(default_factory=list)
    co_corrected_latencies_ms: list[float] = field(default_factory=list)
    start_wall_time: float = 0.0
    end_wall_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_wall_time - self.start_wall_time

    @property
    def n_requests_sent(self) -> int:
        return len(self.raw_measurements)

    @property
    def n_success(self) -> int:
        return sum(1 for m in self.raw_measurements if m.is_success)

    @property
    def error_rate(self) -> float:
        if not self.raw_measurements:
            return 0.0
        return 1.0 - self.n_success / self.n_requests_sent

    @property
    def actual_rps(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.n_requests_sent / self.duration_s


@dataclass
class LoadConfig:
    """Fully specifies a single load generation run.

    Every field that affects results is captured here so the config
    can be committed alongside benchmark results for reproduction.
    """

    target_url: str
    request_body: str
    arrival_rate_rps: float
    duration_seconds: int
    warmup_seconds: int
    auth_header: Optional[str] = None
    content_type: str = "application/json"
    timeout_s: float = 10.0
    expected_status_codes: Optional[set[int]] = None
    # Derived/populated automatically at run time
    hostname: str = field(default_factory=socket.gethostname)
    os_platform: str = field(default_factory=platform.platform)
    python_version: str = field(default_factory=platform.python_version)


def _apply_co_correction(
    measurements: list[RawMeasurement],
    inter_arrival_time_s: float,
    max_phantom_latency_s: float = 5.0,
) -> list[float]:
    corrected: list[float] = []

    for m in measurements:
        if not m.is_success:
            continue

        # Always include the actual observed latency
        actual_latency_ms = m.observed_latency_ms
        corrected.append(actual_latency_ms)

        # CO correction: synthesize phantom measurements for requests
        # that would have been issued during the slow period
        phantom_issue_time = m.intended_issue_time_s + inter_arrival_time_s
        while (phantom_issue_time < m.actual_response_time_s and
               m.actual_response_time_s - phantom_issue_time < max_phantom_latency_s):  # ADD THIS GUARD
            phantom_latency_ms = (m.actual_response_time_s - phantom_issue_time) * 1000.0
            corrected.append(phantom_latency_ms)
            phantom_issue_time += inter_arrival_time_s
    return corrected


async def _run_load_async(config: LoadConfig) -> LoadResult:
    """Core async load generation loop."""
    result = LoadResult(config=config)
    inter_arrival_time_s = 1.0 / config.arrival_rate_rps

    headers = {"content-type": config.content_type}
    if config.auth_header:
        headers["authorization"] = config.auth_header

    result.start_wall_time = time.monotonic()
    run_start = result.start_wall_time
    warmup_end = run_start + config.warmup_seconds
    run_end = run_start + config.warmup_seconds + config.duration_seconds

    # Schedule counter: how many requests should have been issued by now
    next_issue_time = run_start
    warmup_measurements: list[RawMeasurement] = []

    async with httpx.AsyncClient(timeout=config.timeout_s) as client:
        while True:
            now = time.monotonic()

            if now >= run_end:
                break

            if now < next_issue_time:
                # Sleep until the next scheduled issue time
                await asyncio.sleep(next_issue_time - now)
                now = time.monotonic()

            intended = next_issue_time
            next_issue_time += inter_arrival_time_s

            try:
                response = await client.post(
                    config.target_url,
                    content=config.request_body,
                    headers=headers,
                )
                measurement = RawMeasurement(
                    intended_issue_time_s=intended,
                    actual_response_time_s=time.monotonic(),
                    status_code=response.status_code,
                    expected_status_codes=config.expected_status_codes,
                )
            except Exception as e:
                measurement = RawMeasurement(
                    intended_issue_time_s=intended,
                    actual_response_time_s=time.monotonic(),
                    status_code=None,
                    error=str(e),
                    expected_status_codes=config.expected_status_codes,
                )

            if now < warmup_end:
                # Warmup period: collect but don't include in final results
                warmup_measurements.append(measurement)
            else:
                result.raw_measurements.append(measurement)

    result.end_wall_time = time.monotonic()
    result.co_corrected_latencies_ms = _apply_co_correction(
        result.raw_measurements,
        inter_arrival_time_s,
    )

    n_warmup = len(warmup_measurements)
    n_warmup_success = sum(1 for m in warmup_measurements if m.is_success)
    print(
        f"  Warmup: {n_warmup} requests, {n_warmup_success} succeeded "
        f"({config.warmup_seconds}s excluded from results)"
    )

    return result


def run_load(config: LoadConfig) -> LoadResult:
    """Run a load generation session and return CO-corrected measurements.

    This is the main entry point. Runs the async loop in a new event loop.
    """
    return asyncio.run(_run_load_async(config))