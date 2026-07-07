"""
HTTP API for stratum-cache-oracle.

Exposes the MetricsCollector's oracle signals over HTTP so the Rust
router (stratum-router) can consume them via a polling adapter
implementing WorkerSignalsProvider.

SCOPE (honest, per ADR-007)
============================
Only kv_pressure has a real producer (KvPressurePredictor via
MetricsCollector). predicted_latency_ms and sla_affinity have no
implementation anywhere in this service, this endpoint returns
neutral placeholder values for them, explicitly labeled as such in
the response, not silently defaulted.

cache_hit_prob is PERMANENTLY 0.0 / cache_hit_prob_is_real: False on
this endpoint, by design, not pending future work. See ADR-009:
cache_hit_prob is a (request, worker) pair signal, not worker-state,
and cannot be honestly answered by a fixed-interval snapshot poll with
no knowledge of the next request's content. It is computed locally
and synchronously in stratum-router (crates/stratum-router/src/
cache_hit_index.rs) instead. The Python CacheHitIndex/embedding.py in
this service are a validated reference prototype (see their own
module docstrings), not dead code awaiting integration, do not wire
them into this endpoint.

WHY POLLING, NOT REQUEST-RESPONSE PER ROUTING DECISION
=========================================================
stratum-router's RouterStrategy::route() has a documented contract:
must never block indefinitely, must be deterministic given the same
inputs and internal state, and is called on the request hot path.
A live HTTP call per routing decision would violate all three,
network jitter breaks determinism, a slow/down oracle would block
routing, and per-request HTTP round-trips add latency to every
request regardless of whether cache-oracle is even needed for that
decision.

Instead: this endpoint returns a full snapshot of all known workers'
signals. The Rust-side HttpSignalsProvider polls this endpoint on a
fixed interval (default 2s) in a background task and caches the
result. signals_for_workers() reads synchronously from that cache,
satisfying the trait's non-blocking/deterministic contract. Staleness
is handled by falling back to neutral() signals if the last successful
poll exceeds a max-age threshold, see stratum-router's
HttpSignalsProvider implementation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .collector import MetricsCollector

logger = logging.getLogger(__name__)

# Module-level collector instance. In production this is populated by
# register_worker() calls at startup (from a config file or service
# discovery) and by the background scrape loop. For now, workers must
# be registered via the /workers/register endpoint below, or by calling
# collector.register_worker() directly if embedding this app.
collector = MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await collector.start()
    logger.info("cache-oracle API started, scrape loop running")
    yield
    await collector.stop()


app = FastAPI(title="stratum-cache-oracle", version="0.1.0", lifespan=lifespan)


class WorkerSignalsResponse(BaseModel):
    """
    Response shape for a single worker's oracle signals.

    Field-level honesty: `cache_hit_prob_is_real` and
    `latency_sla_signals_are_real` tell the consumer explicitly which
    fields carry real signal vs. neutral placeholders, so the Rust
    side's telemetry can report this accurately rather than silently
    treating placeholder data as if it were measured.
    """

    worker_id: str
    kv_pressure: float
    cache_hit_prob: float
    predicted_latency_ms: float
    sla_affinity: float
    n_observations: int
    cache_hit_prob_is_real: bool = False
    latency_sla_signals_are_real: bool = False


class SignalsSnapshotResponse(BaseModel):
    workers: list[WorkerSignalsResponse]
    scrape_interval_s: float


class RegisterWorkerRequest(BaseModel):
    worker_id: str
    address: str
    backend_type: str = "ollama"


@app.post("/workers/register")
async def register_worker(req: RegisterWorkerRequest) -> dict:
    collector.register_worker(req.worker_id, req.address, req.backend_type)
    return {"status": "registered", "worker_id": req.worker_id}


@app.delete("/workers/{worker_id}")
async def deregister_worker(worker_id: str) -> dict:
    collector.deregister_worker(worker_id)
    return {"status": "deregistered", "worker_id": worker_id}


@app.get("/signals", response_model=SignalsSnapshotResponse)
async def get_signals() -> SignalsSnapshotResponse:
    """
    Return the current oracle signals snapshot for all registered workers.

    Polled by stratum-router's HttpSignalsProvider on a fixed interval,
    not called per routing decision.
    """
    workers_out = []
    for worker_id in collector._worker_addresses.keys():
        metrics = collector.get_metrics(worker_id)
        if metrics is None:
            # Not yet scraped, return neutral, n_observations=0 so the
            # Rust side's MIN_ORACLE_PULLS check correctly treats this
            # as unwarmed rather than trusting a zero-pressure reading.
            workers_out.append(
                WorkerSignalsResponse(
                    worker_id=worker_id,
                    kv_pressure=0.0,
                    cache_hit_prob=0.0,
                    predicted_latency_ms=100.0,
                    sla_affinity=0.5,
                    n_observations=0,
                )
            )
            continue

        predictor = collector._predictors.get(worker_id)
        n_obs = predictor.n_observations if predictor else 0

        workers_out.append(
            WorkerSignalsResponse(
                worker_id=worker_id,
                kv_pressure=metrics.predicted_utilization,
                cache_hit_prob=0.0,  # unimplemented, see module docstring
                predicted_latency_ms=100.0,  # unimplemented, see module docstring
                sla_affinity=0.5,  # unimplemented, see module docstring
                n_observations=n_obs,
                cache_hit_prob_is_real=False,
                latency_sla_signals_are_real=False,
            )
        )

    return SignalsSnapshotResponse(
        workers=workers_out,
        scrape_interval_s=collector._scrape_interval_s,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "n_workers": len(collector._worker_addresses)}