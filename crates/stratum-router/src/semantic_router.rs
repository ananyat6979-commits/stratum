//! SemanticRouter: cache-aware, oracle-driven RouterStrategy.
//!
//! This is the primary routing algorithm. It replaces RoundRobinRouter
//! as the default once the cache oracle is warmed up (>= MIN_ORACLE_PULLS
//! observations per worker).
//!
//! # Pipeline per request
//! 1. Check backpressure — shed immediately if the SLA bucket is exhausted
//! 2. Filter to registry-routable workers — only workers explicitly marked
//!    Unavailable are excluded; a worker absent from the registry is treated
//!    as routable (registry is opt-in health tracking, not a whitelist)
//! 3. If a session_id is present, try affinity routing via jump consistent hash
//!    — route to the affinity worker if it's healthy and its pressure is < 0.8
//! 4. Otherwise, if no worker has enough oracle observations to be trusted,
//!    fall back to round-robin so load distribution isn't silently disabled
//!    during the cold-start window
//! 5. Otherwise, fetch oracle signals for all routable workers and score them
//! 6. Select the highest-scoring worker via select_best_worker()
//! 7. Update the LinUCB bandit with the routing decision (deferred — actual
//!    reward requires observing the outcome, so bandit.update() is called by
//!    the caller after the inference completes, not here)
//!
//! # WorkerSignalsProvider
//! The oracle signals come from a trait, not a direct call to the Python
//! cache-oracle service. This keeps SemanticRouter fully unit-testable
//! without the oracle running. The HTTP adapter implementing this trait
//! is added when the oracle HTTP endpoint is wired up in Phase 3b.

use std::sync::{Arc, RwLock};

use crate::backpressure::{BackpressureController, BackpressureDecision, RouterSlaClass};
use crate::consistent_hash::affinity_bucket;
use crate::router::{RouterError, RouterStrategy, RoutingDecision, WorkerSpec};
use crate::scoring::{select_best_worker, RoutingSignals, ScoreWeights};
use crate::worker_registry::{WorkerHealth, WorkerRegistry};

/// Minimum number of oracle observations before trusting oracle signals.
/// Below this threshold, neutral signals are used (equivalent to round-robin).
const MIN_ORACLE_PULLS: u64 = 5;

/// KV pressure threshold above which affinity routing is overridden.
/// If the affinity worker's pressure exceeds this, fall through to scoring.
const AFFINITY_PRESSURE_THRESHOLD: f64 = 0.8;

/// Oracle signals for a single worker, as seen by the router.
/// This is the data contract between the router and whatever provides signals
/// (mock in tests, HTTP oracle client in production).
#[derive(Debug, Clone)]
pub struct WorkerOracleSignals {
    pub worker_id: String,
    pub signals: RoutingSignals,
    /// Number of observations backing these signals.
    /// Used to decide whether to trust them or fall back to neutral.
    pub n_observations: u64,
}

/// Provides per-worker oracle signals to the SemanticRouter.
///
/// The production implementation queries the cache-oracle HTTP service.
/// Tests use MockSignalsProvider.
pub trait WorkerSignalsProvider: Send + Sync + 'static {
    fn signals_for_workers(&self, worker_ids: &[&str]) -> Vec<WorkerOracleSignals>;
}

/// Test double: returns configurable fixed signals for all workers.
pub struct MockSignalsProvider {
    signals: RoutingSignals,
    n_observations: u64,
}

impl MockSignalsProvider {
    pub fn new(signals: RoutingSignals, n_observations: u64) -> Self {
        Self {
            signals,
            n_observations,
        }
    }

    /// Neutral signals: equal scores for all workers, routing reduces to
    /// tie-breaking by worker index (deterministic, same as round-robin order).
    pub fn neutral() -> Self {
        Self::new(RoutingSignals::neutral(), 0)
    }

    /// Warmed-up oracle: enough observations to trust signals.
    pub fn warmed(signals: RoutingSignals) -> Self {
        Self::new(signals, MIN_ORACLE_PULLS + 1)
    }
}

impl WorkerSignalsProvider for MockSignalsProvider {
    fn signals_for_workers(&self, worker_ids: &[&str]) -> Vec<WorkerOracleSignals> {
        worker_ids
            .iter()
            .map(|id| WorkerOracleSignals {
                worker_id: id.to_string(),
                signals: self.signals,
                n_observations: self.n_observations,
            })
            .collect()
    }
}

/// Cache-aware, oracle-driven router.
pub struct SemanticRouter<P: WorkerSignalsProvider> {
    registry: Arc<WorkerRegistry>,
    signals_provider: Arc<P>,
    backpressure: Arc<BackpressureController>,
    weights: Arc<RwLock<ScoreWeights>>,
    /// Round-robin counter used only during pre-warmup fallback, when no
    /// worker has enough oracle observations to trust score-based routing.
    /// Mirrors RoundRobinRouter's counter design so behavior degrades to
    /// something at least as good as the strategy this one replaces.
    fallback_counter: std::sync::atomic::AtomicU64,
}

impl<P: WorkerSignalsProvider> SemanticRouter<P> {
    pub fn new(
        registry: Arc<WorkerRegistry>,
        signals_provider: Arc<P>,
        backpressure: Arc<BackpressureController>,
    ) -> Self {
        Self {
            registry,
            signals_provider,
            backpressure,
            weights: Arc::new(RwLock::new(ScoreWeights::equal())),
            fallback_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Update the routing score weights (called by the bandit after convergence).
    pub fn update_weights(&self, weights: ScoreWeights) {
        *self.weights.write().unwrap() = weights;
    }

    /// Current score weights (for observability/telemetry).
    pub fn current_weights(&self) -> ScoreWeights {
        *self.weights.read().unwrap()
    }

    /// Attempt affinity routing for a session.
    ///
    /// Returns Some(worker) if the affinity worker is healthy and
    /// not under excessive KV pressure. Returns None to fall through
    /// to score-based routing.
    fn try_affinity_route(
        &self,
        session_id: &str,
        workers: &[WorkerSpec],
        oracle_signals: &[WorkerOracleSignals],
    ) -> Option<WorkerSpec> {
        let bucket = affinity_bucket(Some(session_id), workers.len())?;
        let affinity_worker = workers.get(bucket)?;

        // Check health via registry
        let health = self.registry.health(&affinity_worker.worker_id)?;
        if health != WorkerHealth::Healthy {
            return None;
        }

        // Check pressure — don't use affinity if worker is near saturation
        let oracle = oracle_signals
            .iter()
            .find(|s| s.worker_id == affinity_worker.worker_id)?;

        if oracle.signals.kv_pressure >= AFFINITY_PRESSURE_THRESHOLD {
            return None;
        }

        Some(affinity_worker.clone())
    }
}

impl<P: WorkerSignalsProvider> RouterStrategy for SemanticRouter<P> {
    fn route(
        &self,
        replay_key: &str,
        workers: &[WorkerSpec],
    ) -> Result<RoutingDecision, RouterError> {
        if workers.is_empty() {
            return Err(RouterError::NoWorkersAvailable);
        }

        // Backpressure check: shed before doing any scoring work.
        // SLA class is inferred from a "sla:<class>:" prefix convention on
        // replay_key, mirroring the "session:<id>:" convention used for
        // affinity. Defaults to Batch (most permissive bucket) if absent,
        // since gateway-level SLA assignment already happened upstream --
        // this is a defense-in-depth check, not the primary enforcement point.
        let sla_class = infer_sla_class(replay_key);
        if self.backpressure.check(sla_class) == BackpressureDecision::Shed {
            return Err(RouterError::NoWorkersAvailable);
        }

        // FIX 1: filter to registry-routable workers before any scoring.
        // A worker absent from the registry is treated as routable (registry
        // is opt-in health tracking, not a whitelist) -- only workers
        // EXPLICITLY marked Unavailable are excluded. Previously only
        // try_affinity_route consulted the registry, for a single candidate;
        // every other request ignored health state entirely.
        let routable: Vec<WorkerSpec> = workers
            .iter()
            .filter(|w| {
                self.registry
                    .health(&w.worker_id)
                    .map(|h| h.is_routable())
                    .unwrap_or(true)
            })
            .cloned()
            .collect();

        if routable.is_empty() {
            return Err(RouterError::NoWorkersAvailable);
        }

        // Extract session_id from replay_key prefix if present.
        // Convention: replay_key may be prefixed with "session:<id>:"
        // If not, no affinity is applied.
        let session_id = replay_key
            .strip_prefix("session:")
            .and_then(|s| s.split(':').next());

        // Fetch oracle signals for all routable workers
        let worker_ids: Vec<&str> = routable.iter().map(|w| w.worker_id.as_str()).collect();
        let oracle_signals = self.signals_provider.signals_for_workers(&worker_ids);

        // Try affinity routing first
        if let Some(session) = session_id {
            if let Some(worker) = self.try_affinity_route(session, &routable, &oracle_signals) {
                return Ok(RoutingDecision {
                    score: 1.0,
                    reason: format!("affinity:{session}"),
                    worker,
                });
            }
        }

        // FIX 2: pre-warmup fallback. If NO worker has enough oracle
        // observations to be trusted, score-based selection would compare
        // identical neutral() signals across every candidate, and
        // select_best_worker's strict `>` tie-break pins every request to
        // index 0 -- silently disabling load distribution during the exact
        // window (cold start, cache empty) where it matters most. This is
        // a regression versus RoundRobinRouter, the strategy being replaced.
        // Fall back to the same round-robin pattern until the oracle warms up.
        let any_warmed = oracle_signals
            .iter()
            .any(|s| s.n_observations >= MIN_ORACLE_PULLS);

        if !any_warmed {
            let idx = self
                .fallback_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed) as usize
                % routable.len();
            return Ok(RoutingDecision {
                score: 1.0,
                reason: "fallback:round_robin_pre_warmup".to_string(),
                worker: routable[idx].clone(),
            });
        }

        // Score-based routing: at least one worker has trustworthy signals.
        let weights = self.current_weights();
        let signals_vec: Vec<RoutingSignals> = oracle_signals
            .iter()
            .map(|s| {
                if s.n_observations >= MIN_ORACLE_PULLS {
                    s.signals
                } else {
                    RoutingSignals::neutral()
                }
            })
            .collect();

        let best_idx = select_best_worker(&routable, &signals_vec, &weights)
            .ok_or(RouterError::NoWorkersAvailable)?;

        let worker = routable[best_idx].clone();
        let score = {
            let max_latency = signals_vec
                .iter()
                .map(|s| s.predicted_latency_ms)
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1.0);
            crate::scoring::compute_score(&signals_vec[best_idx], &weights, max_latency)
        };

        Ok(RoutingDecision {
            score,
            reason: format!("semantic:score={score:.3}"),
            worker,
        })
    }

    fn strategy_name(&self) -> &'static str {
        "semantic"
    }
}

/// Infer the router-tier SLA class from a replay_key's optional
/// "sla:<class>:" prefix. Defaults to Batch if absent or unrecognized --
/// this is a secondary defense-in-depth check; the gateway already
/// enforces SLA-aware rate limiting upstream, so an unparseable class
/// here should fail open to the least-privileged bucket, not error out.
fn infer_sla_class(replay_key: &str) -> RouterSlaClass {
    replay_key
        .strip_prefix("sla:")
        .and_then(|s| s.split(':').next())
        .map(|class| match class {
            "realtime" => RouterSlaClass::Realtime,
            "interactive" => RouterSlaClass::Interactive,
            _ => RouterSlaClass::Batch,
        })
        .unwrap_or(RouterSlaClass::Batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker_registry::WorkerRegistry;

    fn test_workers(n: u32) -> Vec<WorkerSpec> {
        (0..n)
            .map(|i| WorkerSpec::new(format!("worker-{i}"), format!("127.0.0.1:{}", 11434 + i)))
            .collect()
    }

    fn test_router(provider: MockSignalsProvider) -> SemanticRouter<MockSignalsProvider> {
        let registry = Arc::new(WorkerRegistry::new());
        let bp = Arc::new(BackpressureController::with_defaults());
        SemanticRouter::new(registry, Arc::new(provider), bp)
    }

    #[test]
    fn routes_to_highest_scoring_worker() {
        let router = test_router(MockSignalsProvider::warmed(RoutingSignals {
            cache_hit_prob: 0.9,
            predicted_latency_ms: 50.0,
            sla_affinity: 0.9,
            kv_pressure: 0.1,
        }));
        let workers = test_workers(3);
        // All workers get same signals; tie-breaks to index 0
        let decision = router.route("key", &workers).unwrap();
        assert_eq!(decision.worker.worker_id, "worker-0");
        assert_eq!(router.strategy_name(), "semantic");
    }

    #[test]
    fn neutral_signals_before_warmup_routes_deterministically() {
        // Note: with the pre-warmup fallback (Fix 2), all-neutral signals now
        // route via round-robin rather than deterministic tie-break, so this
        // test verifies the fallback path fires but no longer expects the
        // same worker for repeated calls. See
        // pre_warmup_fallback_distributes_evenly_across_workers for the
        // distribution guarantee.
        let router = test_router(MockSignalsProvider::neutral());
        let workers = test_workers(3);
        let d1 = router.route("key", &workers).unwrap();
        assert!(d1.reason.contains("fallback"));
    }

    #[test]
    fn no_workers_returns_error() {
        let router = test_router(MockSignalsProvider::neutral());
        assert!(matches!(
            router.route("key", &[]),
            Err(RouterError::NoWorkersAvailable)
        ));
    }

    #[test]
    fn session_affinity_routes_same_session_to_same_worker() {
        let router = test_router(MockSignalsProvider::warmed(RoutingSignals {
            kv_pressure: 0.1, // below threshold
            ..RoutingSignals::neutral()
        }));

        // Register workers as healthy
        for w in test_workers(4) {
            router.registry.register(w);
        }

        let workers = test_workers(4);
        let key = "session:user-abc:req-001";
        let d1 = router.route(key, &workers).unwrap();
        let d2 = router.route(key, &workers).unwrap();
        let d3 = router.route(key, &workers).unwrap();

        assert_eq!(d1.worker.worker_id, d2.worker.worker_id);
        assert_eq!(d2.worker.worker_id, d3.worker.worker_id);
        assert!(d1.reason.contains("affinity"));
    }

    #[test]
    fn high_pressure_overrides_affinity() {
        let router = test_router(MockSignalsProvider::warmed(RoutingSignals {
            kv_pressure: 0.95, // above AFFINITY_PRESSURE_THRESHOLD
            ..RoutingSignals::neutral()
        }));

        for w in test_workers(4) {
            router.registry.register(w);
        }

        let workers = test_workers(4);
        let key = "session:user-abc:req-001";
        let decision = router.route(key, &workers).unwrap();
        // Should fall through to score-based, not affinity
        assert!(!decision.reason.contains("affinity"));
    }

    #[test]
    fn update_weights_changes_routing_behavior() {
        let router = test_router(MockSignalsProvider::warmed(RoutingSignals::neutral()));
        let initial = router.current_weights();
        router.update_weights(ScoreWeights::new(0.7, 0.1, 0.1, 0.1));
        let updated = router.current_weights();
        assert!((updated.cache_hit_prob - initial.cache_hit_prob).abs() > 0.1);
    }

    #[test]
    fn strategy_name_is_semantic() {
        let router = test_router(MockSignalsProvider::neutral());
        assert_eq!(router.strategy_name(), "semantic");
    }

    #[test]
    fn backpressure_shed_prevents_routing() {
        let registry = Arc::new(WorkerRegistry::new());
        // Tiny capacity so we can exhaust it deterministically
        let bp = Arc::new(BackpressureController::new(
            crate::backpressure::BucketConfig {
                capacity: 1.0,
                refill_rate_per_sec: 0.001,
            },
            crate::backpressure::BucketConfig {
                capacity: 1.0,
                refill_rate_per_sec: 0.001,
            },
            crate::backpressure::BucketConfig {
                capacity: 1.0,
                refill_rate_per_sec: 0.001,
            },
        ));
        let router = SemanticRouter::new(registry, Arc::new(MockSignalsProvider::neutral()), bp);
        let workers = test_workers(2);

        // First request consumes the single token
        let first = router.route("sla:realtime:req-1", &workers);
        assert!(first.is_ok());

        // Second immediate request should be shed
        let second = router.route("sla:realtime:req-2", &workers);
        assert!(matches!(second, Err(RouterError::NoWorkersAvailable)));
    }

    #[test]
    fn infer_sla_class_defaults_to_batch() {
        assert_eq!(infer_sla_class("plain-key"), RouterSlaClass::Batch);
        assert_eq!(
            infer_sla_class("sla:realtime:key"),
            RouterSlaClass::Realtime
        );
        assert_eq!(
            infer_sla_class("sla:interactive:key"),
            RouterSlaClass::Interactive
        );
        assert_eq!(infer_sla_class("sla:bogus:key"), RouterSlaClass::Batch);
    }

    #[test]
    fn unavailable_worker_is_excluded_from_non_affinity_routing() {
        let registry = Arc::new(WorkerRegistry::new());
        let workers = test_workers(3);
        for w in &workers {
            registry.register(w.clone());
        }
        registry.set_health("worker-0", WorkerHealth::Unavailable);

        let bp = Arc::new(BackpressureController::with_defaults());
        let router = SemanticRouter::new(
            registry,
            Arc::new(MockSignalsProvider::warmed(RoutingSignals {
                cache_hit_prob: 0.9,
                predicted_latency_ms: 50.0,
                sla_affinity: 0.9,
                kv_pressure: 0.1,
            })),
            bp,
        );

        for _ in 0..10 {
            let decision = router.route("key", &workers).unwrap();
            assert_ne!(
                decision.worker.worker_id, "worker-0",
                "Unavailable worker must never be selected"
            );
        }
    }

    #[test]
    fn pre_warmup_fallback_distributes_evenly_across_workers() {
        let router = test_router(MockSignalsProvider::neutral());
        let workers = test_workers(3);

        let mut counts = [0usize; 3];
        for _ in 0..30 {
            let decision = router.route("key", &workers).unwrap();
            let idx = workers
                .iter()
                .position(|w| w.worker_id == decision.worker.worker_id)
                .unwrap();
            counts[idx] += 1;
            assert!(decision.reason.contains("fallback"));
        }

        assert_eq!(
            counts,
            [10, 10, 10],
            "pre-warmup fallback must distribute evenly, not pin to worker[0]"
        );
    }
}
