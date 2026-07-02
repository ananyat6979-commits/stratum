# ADR-007: KV Pressure Prediction Horizon vs. Scrape Interval Mismatch

**Status**: Proposed- problem identified, solution deferred
**Date**: 2026-07-02
**Deciders**: Project owner

## Context

`KvPressurePredictor` (services/cache-oracle/src/stratum_oracle/predictor.py)
was designed to forecast KV cache block table utilization 100ms ahead
(plus a 200ms vLLM metric-reporting-lag adjustment, for an effective
300ms horizon), so the router can steer away from workers approaching
saturation before an eviction cascade begins.

The predictor's Holt-Winters h-step forecast is `level + h * trend`,
where `h = effective_horizon_ms / scrape_interval_ms`. With the current
Prometheus scrape interval of 15,000ms and a 300ms effective horizon,
`h = 0.02`. This means the trend term contributes at most 2% of one
step's worth of trend to the forecast — `predict()`'s output is, by
construction, dominated by the current level and behaves close to a
last-value predictor.

This is not an implementation bug. It is a genuine information-theoretic
limit: a 15-second sampling interval cannot support meaningful
sub-second forecasting, regardless of the smoothing algorithm used.
The module's original design notes describe Holt-Winters' potential
accuracy improvement over last-value prediction (RMSE 0.043 vs 0.071)
as a property of the algorithm under adequate sampling density, that
comparison has not been separately validated against this deployment's
actual 15s/300ms ratio, and given the math above, it would not be
expected to hold at this ratio.

## Problem Statement

The router needs KV pressure signal at request-granularity (sub-second,
ideally per-request) to make routing decisions that actually avoid
cache thrashing. Prometheus's periodic scrape model, at any interval
compatible with reasonable scrape load, cannot deliver that granularity.

## Options Under Consideration (not yet evaluated in depth)

### Option A: Reduce scrape interval
Simplest change. Risk: increases load on every worker's /metrics
endpoint proportionally; may hit rate limits or add latency to the
serving path if the metrics endpoint isn't cheap to compute. Even at
1-second scraping, `h` only improves to 0.3, still level-dominated.
Would need sub-100ms scraping to meaningfully change `h`, which is
almost certainly impractical for Prometheus-style pull metrics.

### Option B: Event-driven pressure reporting
Replace (or supplement) Prometheus scraping with the router or gateway
reporting observed KV pressure directly after each request/response
cycle, e.g., the InferenceResponseEvent already defined in
causal.proto (RFC-001) carries `kv_blocks_used`, which could feed the
predictor directly instead of via a scrape. This gives true
request-granularity signal, decoupled from any scrape interval.
Requires: wiring event-log consumption into the cache-oracle service,
or a separate lightweight ingestion path. Larger design surface than
Option A but addresses the actual limitation rather than working
around it.

### Option C: Accept level-dominated prediction, reframe the claim
If sub-second signal isn't achievable in this project's scope, the
honest move is to stop claiming 100ms-ahead forecasting and instead
present `predict()` as "smoothed current-state estimate with light
trend adjustment", still useful for routing (smoothing reduces noise
vs. raw scrape values) but without the eviction-cascade-prediction
framing the current docstring implies.

## Decision

**Not yet made.** This ADR exists to track the problem as a real,
acknowledged design gap rather than let it sit silently inside
`predictor.py`'s optimistic docstring. Filed alongside a test
(`test_short_horizon_prediction_is_dominated_by_current_level`) that
locks in the current, honest behavior so any future change here is
deliberate and reviewed, not accidental.

## Revisit Trigger

Before the Phase 3 benchmark (SemanticRouter vs RoundRobinRouter) is
used to make any claim involving KV-pressure-aware routing quality,
this ADR should be resolved, the benchmark's interpretation depends
on which option (or explicit non-fix) is chosen.