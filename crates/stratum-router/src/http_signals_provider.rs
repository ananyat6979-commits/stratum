//! HTTP-polling implementation of WorkerSignalsProvider.
//!
//! Queries cache-oracle's GET /signals endpoint on a fixed interval in
//! a background task, caching the result in an RwLock. signals_for_workers()
//! reads synchronously from that cache, never making a live network call --
//! this satisfies RouterStrategy::route()'s documented contract (must never
//! block indefinitely, must be deterministic given the same internal state).
//!
//! # Staleness handling
//! If the last successful poll is older than `max_staleness`, all signals
//! are reported as unwarmed (n_observations=0), which causes SemanticRouter
//! to fall back to neutral()/round-robin behavior rather than route based
//! on data that might no longer reflect reality. This is a fail-safe, not
//! a fail-open: a dead cache-oracle degrades routing quality, it does not
//! cause routing errors or block requests.
//!
//! # Honesty about placeholder signals (ADR-007, api.py docstring)
//! Only kv_pressure has a real producer today. cache_hit_prob,
//! predicted_latency_ms, and sla_affinity are cache-oracle-side
//! placeholders. This provider passes through whatever cache-oracle
//! reports without pretending otherwise -- see RoutingSignals fields
//! populated directly from the HTTP response, no synthetic enrichment.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::scoring::RoutingSignals;
use crate::semantic_router::{WorkerOracleSignals, WorkerSignalsProvider};

/// Wire format matching cache-oracle's WorkerSignalsResponse (api.py).
#[derive(Debug, Clone, Deserialize)]
struct WorkerSignalsWire {
    worker_id: String,
    kv_pressure: f64,
    cache_hit_prob: f64,
    predicted_latency_ms: f64,
    sla_affinity: f64,
    n_observations: u64,
    #[allow(dead_code)] // reserved for telemetry, not yet consumed
    cache_hit_prob_is_real: bool,
    #[allow(dead_code)]
    latency_sla_signals_are_real: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct SignalsSnapshotWire {
    workers: Vec<WorkerSignalsWire>,
    #[allow(dead_code)]
    scrape_interval_s: f64,
}

/// Cached snapshot with the time it was last successfully refreshed.
struct CachedSnapshot {
    by_worker_id: HashMap<String, WorkerOracleSignals>,
    last_refreshed: Instant,
}

impl CachedSnapshot {
    fn empty() -> Self {
        Self {
            by_worker_id: HashMap::new(),
            // Instant::now() at construction means a freshly-created
            // provider starts "stale" only after max_staleness elapses
            // from startup, not immediately stale before the first poll.
            last_refreshed: Instant::now(),
        }
    }
}

/// Polls cache-oracle's /signals endpoint and serves cached results
/// synchronously to SemanticRouter.
pub struct HttpSignalsProvider {
    cache: Arc<RwLock<CachedSnapshot>>,
    max_staleness: Duration,
}

impl HttpSignalsProvider {
    /// Construct a provider and spawn its background polling task.
    ///
    /// # Arguments
    /// * `base_url` - cache-oracle's base URL, e.g. "http://127.0.0.1:8001"
    /// * `poll_interval` - how often to fetch /signals
    /// * `max_staleness` - if the cache is older than this, signals_for_workers
    ///   reports all workers as unwarmed (n_observations=0) rather than
    ///   serving data that may no longer be accurate
    ///
    /// Must be called from within a Tokio runtime (spawns a background task).
    pub fn new(
        base_url: impl Into<String>,
        poll_interval: Duration,
        max_staleness: Duration,
    ) -> Self {
        let cache = Arc::new(RwLock::new(CachedSnapshot::empty()));
        let base_url = base_url.into();

        let poll_cache = Arc::clone(&cache);
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            let signals_url = format!("{base_url}/signals");

            loop {
                match Self::fetch_once(&client, &signals_url).await {
                    Ok(snapshot) => {
                        let by_worker_id = snapshot
                            .workers
                            .into_iter()
                            .map(|w| {
                                (
                                    w.worker_id.clone(),
                                    WorkerOracleSignals {
                                        worker_id: w.worker_id,
                                        signals: RoutingSignals {
                                            cache_hit_prob: w.cache_hit_prob,
                                            predicted_latency_ms: w.predicted_latency_ms,
                                            sla_affinity: w.sla_affinity,
                                            kv_pressure: w.kv_pressure,
                                        },
                                        n_observations: w.n_observations,
                                    },
                                )
                            })
                            .collect();

                        let mut guard = poll_cache.write().unwrap();
                        guard.by_worker_id = by_worker_id;
                        guard.last_refreshed = Instant::now();
                    }
                    Err(e) => {
                        // Log and continue -- do NOT clear the cache on a
                        // single failed poll. Staleness handling in
                        // signals_for_workers() covers sustained outages;
                        // a transient blip should not discard good data.
                        tracing::warn!(
                            error = %e,
                            url = %signals_url,
                            "cache-oracle poll failed, retaining last-known signals"
                        );
                    }
                }

                tokio::time::sleep(poll_interval).await;
            }
        });

        Self {
            cache,
            max_staleness,
        }
    }

    async fn fetch_once(
        client: &reqwest::Client,
        url: &str,
    ) -> Result<SignalsSnapshotWire, reqwest::Error> {
        client
            .get(url)
            .send()
            .await?
            .json::<SignalsSnapshotWire>()
            .await
    }
}

impl WorkerSignalsProvider for HttpSignalsProvider {
    fn signals_for_workers(&self, worker_ids: &[&str]) -> Vec<WorkerOracleSignals> {
        let guard = self.cache.read().unwrap();
        let is_stale = guard.last_refreshed.elapsed() > self.max_staleness;

        worker_ids
            .iter()
            .map(|id| {
                if is_stale {
                    return WorkerOracleSignals {
                        worker_id: id.to_string(),
                        signals: RoutingSignals::neutral(),
                        n_observations: 0,
                    };
                }

                guard
                    .by_worker_id
                    .get(*id)
                    .cloned()
                    .unwrap_or_else(|| WorkerOracleSignals {
                        worker_id: id.to_string(),
                        signals: RoutingSignals::neutral(),
                        n_observations: 0,
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: these tests exercise CachedSnapshot and staleness logic
    // directly, without a running HTTP server or Tokio runtime spawn --
    // consistent with the project's preference for testing logic without
    // network dependencies (mirrors the WorkerSignalsProvider mock pattern
    // in semantic_router.rs). Full HTTP round-trip is exercised manually /
    // in a future integration test once cache-oracle is running locally.

    fn provider_with_cache(
        by_worker_id: HashMap<String, WorkerOracleSignals>,
        last_refreshed: Instant,
        max_staleness: Duration,
    ) -> HttpSignalsProvider {
        HttpSignalsProvider {
            cache: Arc::new(RwLock::new(CachedSnapshot {
                by_worker_id,
                last_refreshed,
            })),
            max_staleness,
        }
    }

    #[test]
    fn fresh_cache_returns_cached_signals() {
        let mut map = HashMap::new();
        map.insert(
            "worker-0".to_string(),
            WorkerOracleSignals {
                worker_id: "worker-0".to_string(),
                signals: RoutingSignals {
                    cache_hit_prob: 0.7,
                    predicted_latency_ms: 42.0,
                    sla_affinity: 0.5,
                    kv_pressure: 0.3,
                },
                n_observations: 10,
            },
        );

        let provider = provider_with_cache(map, Instant::now(), Duration::from_secs(10));
        let result = provider.signals_for_workers(&["worker-0"]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].n_observations, 10);
        assert_eq!(result[0].signals.kv_pressure, 0.3);
    }

    #[test]
    fn stale_cache_returns_neutral_unwarmed_signals() {
        let mut map = HashMap::new();
        map.insert(
            "worker-0".to_string(),
            WorkerOracleSignals {
                worker_id: "worker-0".to_string(),
                signals: RoutingSignals {
                    cache_hit_prob: 0.9,
                    predicted_latency_ms: 10.0,
                    sla_affinity: 0.9,
                    kv_pressure: 0.1,
                },
                n_observations: 50,
            },
        );

        // last_refreshed far enough in the past to exceed max_staleness
        let stale_time = Instant::now() - Duration::from_secs(60);
        let provider = provider_with_cache(map, stale_time, Duration::from_secs(5));
        let result = provider.signals_for_workers(&["worker-0"]);

        assert_eq!(
            result[0].n_observations, 0,
            "stale data must report as unwarmed"
        );
        assert_eq!(
            result[0].signals.kv_pressure,
            RoutingSignals::neutral().kv_pressure
        );
    }

    #[test]
    fn unknown_worker_returns_neutral() {
        let provider = provider_with_cache(HashMap::new(), Instant::now(), Duration::from_secs(10));
        let result = provider.signals_for_workers(&["unknown-worker"]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].worker_id, "unknown-worker");
        assert_eq!(result[0].n_observations, 0);
    }

    #[test]
    fn queries_multiple_workers_independently() {
        let mut map = HashMap::new();
        map.insert(
            "worker-a".to_string(),
            WorkerOracleSignals {
                worker_id: "worker-a".to_string(),
                signals: RoutingSignals::neutral(),
                n_observations: 20,
            },
        );

        let provider = provider_with_cache(map, Instant::now(), Duration::from_secs(10));
        let result = provider.signals_for_workers(&["worker-a", "worker-b"]);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].n_observations, 20); // known worker
        assert_eq!(result[1].n_observations, 0); // unknown worker
    }
}
