"""
Prometheus metrics collector for vLLM/Ollama KV block table utilization.

Scrapes the worker's /metrics endpoint and extracts KV cache utilization.
Feeds the extracted values into KvPressurePredictor instances.

METRIC DISCOVERY
================
vLLM exposes:
  vllm:gpu_cache_usage_perc: fraction of KV cache blocks in use
  vllm:num_preemptions_total: cumulative preemptions (eviction indicator)

Ollama (our Phase 2/3 backend) does not expose KV cache metrics natively.
For Ollama workers, we use a synthetic utilization estimate based on
response latency relative to baseline (higher latency ≈ higher cache pressure).
This is a proxy, not a direct measurement. Documented as a known limitation.

SCRAPE INTERVAL
===============
Default: 15 seconds (matching Prometheus default).
The predictor's horizon calibration assumes this interval.
If you change SCRAPE_INTERVAL_S, update predictor.py accordingly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from .predictor import KvPressurePredictor

logger = logging.getLogger(__name__)

SCRAPE_INTERVAL_S = 15
SCRAPE_TIMEOUT_S = 5.0
METRIC_NAME_VLLM = "vllm:gpu_cache_usage_perc"



@dataclass
class WorkerMetrics:
    """Latest scraped metrics for a single worker."""
    worker_id: str
    address: str
    kv_utilization: float          # [0.0, 1.0]
    predicted_utilization: float   # [0.0, 1.0]: 100ms ahead
    last_scrape_time: float        # Unix timestamp
    scrape_success: bool
    backend_type: str              # "vllm" | "ollama"


class MetricsCollector:
    """
    Async metrics collector. One instance manages all workers.

    Usage:
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://127.0.0.1:11434", "ollama")
        await collector.start()   # begins background scrape loop
        metrics = collector.get_metrics("worker-0")
    """

    def __init__(self, scrape_interval_s: float = SCRAPE_INTERVAL_S):
        self._scrape_interval_s = scrape_interval_s
        self._predictors: dict[str, KvPressurePredictor] = {}
        self._metrics: dict[str, WorkerMetrics] = {}
        self._worker_addresses: dict[str, tuple[str, str]] = {}  # id -> (address, backend_type)
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def register_worker(
        self,
        worker_id: str,
        address: str,
        backend_type: str = "ollama",
    ) -> None:
        """Register a worker for metric collection."""
        self._worker_addresses[worker_id] = (address, backend_type)
        self._predictors[worker_id] = KvPressurePredictor()
        logger.info("Registered worker %s (%s) at %s", worker_id, backend_type, address)

    def deregister_worker(self, worker_id: str) -> None:
        self._worker_addresses.pop(worker_id, None)
        self._predictors.pop(worker_id, None)
        self._metrics.pop(worker_id, None)

    async def start(self) -> None:
        """Start the background scrape loop."""
        self._running = True
        self._task = asyncio.create_task(self._scrape_loop())
        logger.info(
            "MetricsCollector started, scrape interval=%ds",
            self._scrape_interval_s,
        )

    async def stop(self) -> None:
        """Stop the background scrape loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_metrics(self, worker_id: str) -> Optional[WorkerMetrics]:
        """Get the latest metrics for a worker. Returns None if not yet scraped."""
        return self._metrics.get(worker_id)

    def get_predicted_pressure(self, worker_id: str) -> float:
        """
        Get the predicted KV pressure for a worker.

        Returns 0.0 (no pressure assumed) if the worker hasn't been
        scraped yet or if the predictor isn't warmed up.
        This is the safe default: it prevents over-steering away from
        workers before we have data.
        """
        predictor = self._predictors.get(worker_id)
        if predictor is None or not predictor.is_warmed_up:
            return 0.0
        return predictor.predict()

    async def _scrape_loop(self) -> None:
        """Background loop: scrape all workers every scrape_interval_s seconds."""
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_S) as client:
            while self._running:
                await self._scrape_all(client)
                await asyncio.sleep(self._scrape_interval_s)

    async def _scrape_all(self, client: httpx.AsyncClient) -> None:
        tasks = [
            self._scrape_worker(client, worker_id, address, backend_type)
            for worker_id, (address, backend_type) in self._worker_addresses.items()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _scrape_worker(
        self,
        client: httpx.AsyncClient,
        worker_id: str,
        address: str,
        backend_type: str,
    ) -> None:
        # Conservative fallback value used when a scrape genuinely
        # fails (connection refused, timeout, malformed response).
        # This must be high, not 0.0: 0.0 reads as "empty, healthy
        # cache" to KvPressurePredictor and the router, which is the
        # opposite of what an unreachable worker should signal.
        HIGH_PRESSURE_ON_SCRAPE_FAILURE = 1.0

        try:
            utilization = await self._fetch_utilization(client, address, backend_type)

            if utilization is None:
                # The scrape failed. This is the fix for the real bug:
                # previously, a failure inside _fetch_ollama_utilization
                # was silently converted to a clean-looking 0.0 before
                # it ever reached this function, so this method's own
                # except block below (with the "record as high
                # pressure" comment) never actually ran for that
                # failure mode. Handling None explicitly here closes
                # that gap without relying on an exception to propagate.
                logger.warning(
                    "Worker %s: scrape returned no data, recording as "
                    "high pressure (%.1f) instead of a clean reading",
                    worker_id, HIGH_PRESSURE_ON_SCRAPE_FAILURE,
                )
                predictor = self._predictors[worker_id]
                predictor.update(HIGH_PRESSURE_ON_SCRAPE_FAILURE)
                predicted = predictor.predict()
                self._metrics[worker_id] = WorkerMetrics(
                    worker_id=worker_id,
                    address=address,
                    kv_utilization=HIGH_PRESSURE_ON_SCRAPE_FAILURE,
                    predicted_utilization=predicted,
                    last_scrape_time=time.time(),
                    scrape_success=False,
                    backend_type=backend_type,
                )
                return

            predictor = self._predictors[worker_id]
            predictor.update(utilization)
            predicted = predictor.predict()

            self._metrics[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                address=address,
                kv_utilization=utilization,
                predicted_utilization=predicted,
                last_scrape_time=time.time(),
                scrape_success=True,
                backend_type=backend_type,
            )

            logger.debug(
                "Worker %s: utilization=%.3f predicted=%.3f",
                worker_id, utilization, predicted,
            )

        except Exception as e:
            # Genuinely unexpected failures not already converted to
            # None by the fetch layer (e.g. a bug in the predictor
            # itself) still land here, with the same conservative
            # fallback: keep the last known-good utilization, mark
            # scrape_success=False.
            logger.warning("Failed to scrape worker %s: %s", worker_id, e)
            if worker_id in self._metrics:
                existing = self._metrics[worker_id]
                self._metrics[worker_id] = WorkerMetrics(
                    worker_id=existing.worker_id,
                    address=existing.address,
                    kv_utilization=existing.kv_utilization,
                    predicted_utilization=existing.predicted_utilization,
                    last_scrape_time=time.time(),
                    scrape_success=False,
                    backend_type=existing.backend_type,
                )

    async def _fetch_utilization(
        self,
        client: httpx.AsyncClient,
        address: str,
        backend_type: str,
    ) -> Optional[float]:
        """
        Returns None when the underlying scrape failed. The caller,
        _scrape_worker, must treat None as a failed scrape and record
        conservative high pressure, never substitute it with a clean
        successful reading.
        """
        if backend_type == "vllm":
            return await self._fetch_vllm_utilization(client, address)
        else:
            return await self._fetch_ollama_utilization(client, address)

    async def _fetch_vllm_utilization(
        self,
        client: httpx.AsyncClient,
        address: str,
    ) -> float:
        """Parse vLLM's Prometheus /metrics endpoint."""
        response = await client.get(f"{address}/metrics")
        response.raise_for_status()

        for line in response.text.splitlines():
            if line.startswith(METRIC_NAME_VLLM) and not line.startswith("#"):
                value_str = line.split()[-1]
                return float(value_str)

        # Metric not found: return 0 (no cache data yet)
        return 0.0

    async def _fetch_ollama_utilization(
        self,
        client: httpx.AsyncClient,
        address: str,
    ) -> Optional[float]:
        """
        Estimate KV utilization from Ollama's /api/ps endpoint.

        Ollama does not expose KV cache metrics directly. We use the
        ratio of in-use model memory to total model memory as a proxy.
        This correlates with KV cache pressure but is not equivalent.

        Returns 0.0 only when Ollama responded successfully and reports
        zero loaded models. This is a real, legitimate zero: the worker
        is idle, not broken.

        Returns None when the scrape itself failed: connection refused,
        timeout, non-2xx status, or a malformed response. None is not
        a utilization value and must never be substituted with 0.0 by
        the caller. A 0.0 here is indistinguishable, downstream, from
        "this worker is empty and perfectly healthy", exactly the
        wrong signal for a worker that is actually unreachable.
        """
        try:
            response = await client.get(f"{address}/api/ps")
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            if not models:
                return 0.0
            # Sum size_vram across every currently loaded model, not
            # just one. Multiple models can be resident in Ollama at
            # once, and pressure should reflect total VRAM committed,
            # not an arbitrary single model's share of it.
            max_vram_bytes = 8 * 1024 * 1024 * 1024
            total_vram = sum(m.get("size_vram", 0) for m in models)
            return min(1.0, total_vram / max_vram_bytes)
        except Exception as e:
            logger.warning("Ollama scrape failed for %s: %s", address, e)
            return None