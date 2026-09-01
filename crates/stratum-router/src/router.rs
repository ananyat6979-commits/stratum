//! Router trait and built-in routing strategies.
//!
//! # Design
//! `RouterStrategy` is the core abstraction. Every routing algorithm,
//! round-robin, semantic cache-aware, bandit-weighted, implements this
//! one trait. The gateway calls `route()` and gets back a worker ID and
//! a routing score. It does not know which strategy is active.
//!
//! This separation means:
//! - Phase 2 ships `RoundRobinRouter` (this file) as the baseline
//! - Phase 3 adds `SemanticRouter` without touching any call site
//! - The experiment engine can A/B test strategies by swapping the
//!   `Arc<dyn RouterStrategy>` at runtime without restarting the gateway
//!
//! # Event Log Integration
//! `route_and_log()` wraps `route()` and writes a `RoutingDecisionEvent`
//! to the event log. This is the integration point between the router
//! and the replay system. All production routing goes through
//! `route_and_log()`, never through `route()` directly.

use std::sync::atomic::{AtomicU64, Ordering};

use stratum_replay::event_log::{AppendOnlyEventLog, EventLogError, ReplayEvent};

/// A worker available to receive inference requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerSpec {
    pub worker_id: String,
    pub address: String,
}

impl WorkerSpec {
    pub fn new(worker_id: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            address: address.into(),
        }
    }

    /// Convenience constructor for tests.
    pub fn test_worker(index: u32) -> Self {
        Self::new(
            format!("worker-{index}"),
            format!("127.0.0.1:{}", 11434 + index),
        )
    }
}

/// Snapshot of the oracle signals actually used to select a worker,
/// captured inside route() at the moment of selection.
///
/// # Why this is captured here, not recomputed after route() returns
/// SemanticRouter's per-call signal computation (see semantic_router.rs's
/// route(), specifically the local, synchronous overwrite of
/// cache_hit_prob) is transient: nothing about it is stored on `self`
/// after route() returns. A caller trying to reconstruct "what signals
/// led to this decision" after the fact would either have to re-fetch
/// oracle state (which may have changed) or guess, both wrong. This
/// struct exists specifically so the real, exact values used ARE the
/// values recorded, captured at the only point they're actually in
/// scope.
///
/// # Why this is Option<T> on RoutingDecision, not a required field
/// RoundRobinRouter has no oracle signals at all, None here is an
/// honest absence, not a fabricated neutral() value standing in for
/// "no data." SemanticRouter's own affinity-routing and pre-warmup
/// fallback paths (see semantic_router.rs's route()) also correctly
/// return None: both explicitly bypass score-based oracle selection,
/// so attaching a snapshot to either would misrepresent what actually
/// drove the decision. Only the real, scored path populates this.
#[derive(Debug, Clone)]
pub struct OracleSnapshot {
    /// See scoring::RoutingSignals::cache_hit_prob. This is the
    /// LOCALLY-COMPUTED value SemanticRouter actually used, not the
    /// (permanently placeholder, always-0.0) wire value from the
    /// cache-oracle HTTP response, see semantic_router.rs's route()
    /// for the explicit overwrite this reflects.
    pub cache_hit_prob: f64,
    /// See scoring::RoutingSignals::predicted_latency_ms.
    pub predicted_latency_ms: f64,
    /// See scoring::RoutingSignals::sla_affinity.
    pub sla_affinity: f64,
    /// See scoring::RoutingSignals::kv_pressure.
    pub kv_pressure: f64,
    /// See WorkerOracleSignals::n_observations. How many real
    /// observations backed the signals above, distinct from the
    /// four scoring fields, this lives one level up on
    /// WorkerOracleSignals, not on RoutingSignals itself.
    pub n_observations: u64,
}

/// The result of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The selected worker.
    pub worker: WorkerSpec,
    /// Routing score for the selected worker in [0.0, 1.0].
    /// For round-robin: always 1.0 (all workers equivalent).
    /// For semantic router: weighted combination of oracle signals.
    pub score: f64,
    /// Human-readable description of why this worker was chosen.
    /// Used in telemetry and replay debugging.
    pub reason: String,
    /// The oracle signals actually used to reach this decision, if
    /// any. See OracleSnapshot's own doc comment for exactly when
    /// this is Some vs None.
    pub oracle_snapshot: Option<OracleSnapshot>,
}

/// Errors that can occur during routing.
#[derive(Debug)]
pub enum RouterError {
    /// No workers are registered and available.
    NoWorkersAvailable,
    /// The event log write failed.
    EventLog(EventLogError),
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoWorkersAvailable => write!(f, "no workers available for routing"),
            Self::EventLog(e) => write!(f, "event log error: {e}"),
        }
    }
}

/// Core routing interface. All routing algorithms implement this trait.
///
/// # Contract
/// - `route()` must be deterministic given the same inputs and the same
///   internal state. This is required for replay correctness.
/// - `route()` must never block indefinitely. It is called on the
///   request hot path.
/// - `route()` must be cancel-safe: if the future is dropped after
///   `route()` returns, no partial state should be left inconsistent.
///   (Round-robin: trivially cancel-safe. Semantic router: must ensure
///   FAISS queries and bandit updates are atomic with respect to routing.)
///
/// # Implementors
/// - `RoundRobinRouter`: baseline, no oracle signals
/// - `SemanticRouter` (Phase 3): FAISS cache-hit prediction + LinUCB
pub trait RouterStrategy: Send + Sync + 'static {
    /// Select a worker for the given replay_key.
    ///
    /// `replay_key` identifies the request for event log correlation.
    /// `prompt` is the request's raw text content, used by strategies
    /// that need to reason about request similarity (e.g. SemanticRouter's
    /// cache-hit prediction). RoundRobinRouter ignores it entirely,
    /// accepting an unused parameter here is the honest cost of a trait
    /// method serving both a strategy that needs content-awareness and
    /// one that doesn't, rather than parsing prompt text out of
    /// replay_key via string convention (fragile: prompt text can
    /// contain any character, including the delimiters other
    /// replay_key conventions already use).
    /// `workers` is the current set of available workers.
    fn route(
        &self,
        replay_key: &str,
        prompt: &str,
        workers: &[WorkerSpec],
    ) -> Result<RoutingDecision, RouterError>;

    /// Human-readable name for this strategy, used in telemetry labels.
    fn strategy_name(&self) -> &'static str;

    /// Record that a request was actually dispatched to `worker_id` after
    /// a `route()` call selected it, so strategies with per-worker state
    /// (e.g. SemanticRouter's cache-hit index) can update.
    ///
    /// Default no-op: most strategies (RoundRobinRouter) have no such
    /// state. This exists on the trait, rather than requiring a caller to
    /// downcast `dyn RouterStrategy` to a concrete type, specifically so
    /// `AppState` can hold `Arc<dyn RouterStrategy>` and call this
    /// uniformly regardless of which strategy is active, see
    /// `stratum-gateway`'s `handle_chat_completions`, which calls this
    /// exactly once, after dispatch succeeds, never from inside route().
    fn record_outcome(&self, _worker_id: &str, _prompt: &str) {}
}

/// Round-robin router: selects workers cyclically, no oracle signals.
///
/// This is the Phase 2 baseline. Every request goes to the next worker
/// in sequence, wrapping around. Workers receive equal load regardless
/// of their current state (KV cache pressure, response time, etc.).
///
/// # Correctness
/// The counter uses `fetch_add` with `Relaxed` ordering. This is correct
/// here: we don't need the counter to be ordered relative to any other
/// memory operation, we only need atomicity (no two calls get the same
/// counter value). `SeqCst` would be unnecessary overhead.
///
/// # Cancellation safety
/// `route()` only does an atomic fetch_add and an array index. Cancel-safe.
pub struct RoundRobinRouter {
    counter: AtomicU64,
}

impl RoundRobinRouter {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }
}

impl Default for RoundRobinRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl RouterStrategy for RoundRobinRouter {
    fn route(
        &self,
        _replay_key: &str,
        _prompt: &str,
        workers: &[WorkerSpec],
    ) -> Result<RoutingDecision, RouterError> {
        if workers.is_empty() {
            return Err(RouterError::NoWorkersAvailable);
        }

        let index = self.counter.fetch_add(1, Ordering::Relaxed) as usize % workers.len();
        let worker = workers[index].clone();

        Ok(RoutingDecision {
            score: 1.0,
            reason: format!("round-robin index {index}"),
            worker,
            oracle_snapshot: None, // RoundRobinRouter has no oracle signals
        })
    }

    fn strategy_name(&self) -> &'static str {
        "round_robin"
    }
}

/// Route a request and write the decision to the event log.
///
/// This is the production entry point. The `replay_key` and
/// `ingress_event_id` link this routing decision to the ingress event
/// in the causal DAG.
///
/// The event payload is a bincode-serialized `RoutingDecisionPayload`.
/// The event log layer treats it as opaque bytes, it does not interpret
/// the payload schema.
pub fn route_and_log(
    strategy: &dyn RouterStrategy,
    replay_key: &str,
    prompt: &str,
    ingress_event_id: u128,
    workers: &[WorkerSpec],
    log: &AppendOnlyEventLog,
) -> Result<(RoutingDecision, ReplayEvent), RouterError> {
    let decision = strategy.route(replay_key, prompt, workers)?;

    // Serialize the routing decision as the event payload.
    // This is intentionally a simple struct, not the full causal.proto
    // RoutingDecisionEvent, the proto layer is added in Phase 3 when
    // the oracle state snapshot becomes meaningful.
    let payload = RoutingDecisionPayload {
        replay_key: replay_key.to_string(),
        selected_worker_id: decision.worker.worker_id.clone(),
        routing_score: decision.score,
        strategy_name: strategy.strategy_name().to_string(),
        reason: decision.reason.clone(),
        oracle_cache_hit_prob: decision.oracle_snapshot.as_ref().map(|s| s.cache_hit_prob),
        oracle_predicted_latency_ms: decision.oracle_snapshot.as_ref().map(|s| s.predicted_latency_ms),
        oracle_sla_affinity: decision.oracle_snapshot.as_ref().map(|s| s.sla_affinity),
        oracle_kv_pressure: decision.oracle_snapshot.as_ref().map(|s| s.kv_pressure),
        oracle_n_observations: decision.oracle_snapshot.as_ref().map(|s| s.n_observations),
    };

    let payload_bytes =
        bincode::serialize(&payload).expect("RoutingDecisionPayload serialization must not fail");

    let event_id = uuid::Uuid::new_v4().as_u128();
    let event = log
        .append(event_id, vec![ingress_event_id], payload_bytes)
        .map_err(RouterError::EventLog)?;

    Ok((decision, event))
}

/// The payload written to the event log for each routing decision.
///
/// The five `oracle_*` fields were added when SemanticRouter's real
/// oracle signals (see OracleSnapshot) became worth recording, all
/// five are None together for RoundRobinRouter decisions and
/// SemanticRouter's affinity/pre-warmup-fallback paths (see
/// RoutingDecision::oracle_snapshot's doc comment), Some together only
/// for the real, scored SemanticRouter path. This closes part of the
/// gap to RFC-001's original CausalDecisionEvent design (see
/// docs/SCOPE.md's "A related gap" note), still smaller than that
/// design's full OracleStateSnapshot (this records only the winning
/// worker's signals, not every candidate's), a deliberate, stated
/// scope choice, not an oversight.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RoutingDecisionPayload {
    pub replay_key: String,
    pub selected_worker_id: String,
    pub routing_score: f64,
    pub strategy_name: String,
    pub reason: String,
    pub oracle_cache_hit_prob: Option<f64>,
    pub oracle_predicted_latency_ms: Option<f64>,
    pub oracle_sla_affinity: Option<f64>,
    pub oracle_kv_pressure: Option<f64>,
    pub oracle_n_observations: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_workers(n: u32) -> Vec<WorkerSpec> {
        (0..n).map(WorkerSpec::test_worker).collect()
    }

    #[test]
    fn round_robin_distributes_evenly() {
        let router = RoundRobinRouter::new();
        let workers = test_workers(3);
        let mut counts = vec![0usize; 3];

        for _ in 0..30 {
            let decision = router.route("key", "test prompt", &workers).unwrap();
            let idx = workers
                .iter()
                .position(|w| w.worker_id == decision.worker.worker_id)
                .unwrap();
            counts[idx] += 1;
        }

        // 30 requests / 3 workers = 10 each, exactly
        assert_eq!(counts, vec![10, 10, 10]);
    }

    #[test]
    fn round_robin_wraps_around_correctly() {
        let router = RoundRobinRouter::new();
        let workers = test_workers(2);

        let d0 = router.route("k", "test prompt", &workers).unwrap();
        let d1 = router.route("k", "test prompt", &workers).unwrap();
        let d2 = router.route("k", "test prompt", &workers).unwrap();

        assert_eq!(d0.worker.worker_id, "worker-0");
        assert_eq!(d1.worker.worker_id, "worker-1");
        assert_eq!(d2.worker.worker_id, "worker-0"); // wraps
    }

    #[test]
    fn no_workers_returns_error() {
        let router = RoundRobinRouter::new();
        let result = router.route("key", "test prompt", &[]);
        assert!(matches!(result, Err(RouterError::NoWorkersAvailable)));
    }

    #[test]
    fn round_robin_score_is_always_one() {
        let router = RoundRobinRouter::new();
        let workers = test_workers(2);
        let decision = router.route("key", "test prompt", &workers).unwrap();
        assert_eq!(decision.score, 1.0);
    }

    #[test]
    fn strategy_name_is_correct() {
        let router = RoundRobinRouter::new();
        assert_eq!(router.strategy_name(), "round_robin");
    }

    #[test]
    fn route_and_log_writes_event_with_correct_dependency() {
        let dir =
            std::env::temp_dir().join(format!("stratum-router-test-{}", uuid::Uuid::new_v4()));
        let log_path = dir.with_extension("redb");

        let log = AppendOnlyEventLog::open(&log_path, "node-0").unwrap();
        let router = RoundRobinRouter::new();
        let workers = test_workers(2);

        let ingress_event_id: u128 = 42;
        let (_decision, event) = route_and_log(
            &router,
            "replay-key-abc",
            "test prompt",
            ingress_event_id,
            &workers,
            &log,
        )
        .unwrap();

        // The routing event must list the ingress event as a dependency
        assert_eq!(event.dependency_ids, vec![ingress_event_id]);

        // The event must be retrievable from the log
        let all_events = log.load_all().unwrap();
        assert_eq!(all_events.len(), 1);
        assert_eq!(all_events[0].event_id, event.event_id);
    }

    #[test]
    fn route_and_log_payload_is_deserializable() {
        let log_path = std::env::temp_dir().join(format!(
            "stratum-router-payload-test-{}.redb",
            uuid::Uuid::new_v4()
        ));

        let log = AppendOnlyEventLog::open(&log_path, "node-0").unwrap();
        let router = RoundRobinRouter::new();
        let workers = test_workers(2);

        let (_decision, event) =
            route_and_log(&router, "test-key", "test prompt", 0u128, &workers, &log).unwrap();

        // The payload must deserialize back to a RoutingDecisionPayload
        let payload: RoutingDecisionPayload = bincode::deserialize(&event.payload).unwrap();

        assert_eq!(payload.replay_key, "test-key");
        assert_eq!(payload.strategy_name, "round_robin");
        assert!(payload.routing_score > 0.0);
    }

        #[test]
    fn round_robin_decisions_have_no_oracle_snapshot() {
        let log_path = std::env::temp_dir().join(format!(
            "stratum-router-oracle-test-{}.redb",
            uuid::Uuid::new_v4()
        ));

        let log = AppendOnlyEventLog::open(&log_path, "node-0").unwrap();
        let router = RoundRobinRouter::new();
        let workers = test_workers(2);

        let (decision, event) =
            route_and_log(&router, "test-key", "test prompt", 0u128, &workers, &log).unwrap();

        assert!(decision.oracle_snapshot.is_none());

        let payload: RoutingDecisionPayload = bincode::deserialize(&event.payload).unwrap();
        assert!(payload.oracle_cache_hit_prob.is_none());
        assert!(payload.oracle_predicted_latency_ms.is_none());
        assert!(payload.oracle_sla_affinity.is_none());
        assert!(payload.oracle_kv_pressure.is_none());
        assert!(payload.oracle_n_observations.is_none());
    }
}
