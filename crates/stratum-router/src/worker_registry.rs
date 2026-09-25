//! Worker registry: tracks available inference workers and their health.
//!
//! The registry is the router's view of the serving cluster. It maintains:
//! - Which workers are currently registered and reachable
//! - Per-worker health state (healthy, degraded, unavailable)
//! - Per-worker metadata (address, model, capacity)
//!
//! # Health Model
//! Workers transition through three states:
//!   Healthy → Degraded → Unavailable → Healthy
//!
//! Degraded workers still receive traffic but at reduced weight.
//! Unavailable workers are excluded from routing until they recover.
//!
//! State transitions are driven by:
//! - Explicit health check results (HTTP /health endpoint polling)
//! - Implicit signals from routing outcomes (consecutive timeouts → Degraded)
//!
//! Phase 3 implements explicit health checking. For now, all registered
//! workers are assumed Healthy. the router trusts the operator to
//! deregister workers that are actually down.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::router::WorkerSpec;

/// Health state of a single worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealth {
    /// Worker is responding normally. Full routing weight.
    Healthy,
    /// Worker is responding but degraded (high latency, high error rate).
    /// Still receives traffic at reduced weight.
    Degraded,
    /// Worker is not responding. Excluded from routing.
    Unavailable,
}

impl WorkerHealth {
    /// Routing weight multiplier. Used to deprioritize degraded workers.
    pub fn weight_multiplier(&self) -> f64 {
        match self {
            Self::Healthy => 1.0,
            Self::Degraded => 0.3,
            Self::Unavailable => 0.0,
        }
    }

    pub fn is_routable(&self) -> bool {
        *self != Self::Unavailable
    }
}

/// Metadata and health state for a single registered worker.
#[derive(Debug, Clone)]
pub struct WorkerEntry {
    pub spec: WorkerSpec,
    pub health: WorkerHealth,
    /// When this worker was registered.
    pub registered_at: Instant,
    /// When the health state was last updated.
    pub last_health_update: Instant,
    /// Consecutive routing failures since last success.
    /// Used for implicit health degradation.
    pub consecutive_failures: u32,
}

impl WorkerEntry {
    pub fn new(spec: WorkerSpec) -> Self {
        let now = Instant::now();
        Self {
            spec,
            health: WorkerHealth::Healthy,
            registered_at: now,
            last_health_update: now,
            consecutive_failures: 0,
        }
    }

    /// Record a successful routing outcome. Resets failure counter and
    /// unconditionally refreshes last_health_update. Transitions
    /// Degraded → Healthy after a success.
    ///
    /// # A real bug this fixed
    /// Previously only refreshed last_health_update inside the
    /// `if self.health == WorkerHealth::Degraded` branch. A worker
    /// that stayed Healthy continuously, receiving a steady stream of
    /// real successes and never degrading, never had its timestamp
    /// touched past construction time. mark_stale_unavailable's own
    /// doc comment says it marks workers that have "not had a
    /// successful routing outcome within timeout". that was not
    /// what the code actually checked before this fix, since a
    /// continuously healthy worker's last_health_update never
    /// advanced regardless of how many real successes it recorded.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.last_health_update = Instant::now();
        if self.health == WorkerHealth::Degraded {
            self.health = WorkerHealth::Healthy;
        }
    }

    /// Record a routing failure. Escalates health state after thresholds.
    ///
    /// Thresholds (configurable in production; hardcoded for Phase 3):
    /// - 3 consecutive failures → Degraded
    /// - 10 consecutive failures → Unavailable
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        let new_health = if self.consecutive_failures >= 10 {
            WorkerHealth::Unavailable
        } else if self.consecutive_failures >= 3 {
            WorkerHealth::Degraded
        } else {
            self.health
        };
        if new_health != self.health {
            self.health = new_health;
            self.last_health_update = Instant::now();
        }
    }
}

/// Thread-safe registry of available inference workers.
///
/// `Arc<WorkerRegistry>` can be cloned cheaply and shared across
/// the router, health checker, and chaos injector without copying
/// the registry state.
#[derive(Debug, Default)]
pub struct WorkerRegistry {
    workers: RwLock<HashMap<String, WorkerEntry>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new worker. If the worker is already registered,
    /// its entry is updated with the new spec (but health state is preserved).
    pub fn register(&self, spec: WorkerSpec) {
        let mut workers = self.workers.write().unwrap();
        workers
            .entry(spec.worker_id.clone())
            .or_insert_with(|| WorkerEntry::new(spec));
    }

    /// Deregister a worker by ID. Removes it from routing immediately.
    pub fn deregister(&self, worker_id: &str) {
        let mut workers = self.workers.write().unwrap();
        workers.remove(worker_id);
    }

    /// Return all routable workers (Healthy or Degraded) as `WorkerSpec`s.
    /// Unavailable workers are excluded.
    pub fn routable_workers(&self) -> Vec<WorkerSpec> {
        let workers = self.workers.read().unwrap();
        workers
            .values()
            .filter(|e| e.health.is_routable())
            .map(|e| e.spec.clone())
            .collect()
    }

    /// Return all registered workers regardless of health state.
    /// Used by the chaos system and health dashboard.
    pub fn all_workers(&self) -> Vec<WorkerEntry> {
        let workers = self.workers.read().unwrap();
        workers.values().cloned().collect()
    }

    /// Record a successful routing outcome for the given worker.
    pub fn record_success(&self, worker_id: &str) {
        let mut workers = self.workers.write().unwrap();
        if let Some(entry) = workers.get_mut(worker_id) {
            entry.record_success();
        }
    }

    /// Record a routing failure for the given worker.
    pub fn record_failure(&self, worker_id: &str) {
        let mut workers = self.workers.write().unwrap();
        if let Some(entry) = workers.get_mut(worker_id) {
            entry.record_failure();
        }
    }

    /// Forcibly set a worker's health state.
    /// Used by the chaos injector to simulate failures.
    pub fn set_health(&self, worker_id: &str, health: WorkerHealth) {
        let mut workers = self.workers.write().unwrap();
        if let Some(entry) = workers.get_mut(worker_id) {
            entry.health = health;
            entry.last_health_update = Instant::now();
        }
    }

    /// Return the number of registered workers.
    pub fn len(&self) -> usize {
        self.workers.read().unwrap().len()
    }

    /// Return true if no workers are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the health state of a specific worker, if registered.
    pub fn health(&self, worker_id: &str) -> Option<WorkerHealth> {
        self.workers
            .read()
            .unwrap()
            .get(worker_id)
            .map(|e| e.health)
    }

    /// Mark all currently-Healthy workers that have not had a
    /// successful routing outcome (via record_success) within
    /// `timeout` as Unavailable. Used by a periodic health sweep.
    ///
    /// This comment now accurately describes the check: prior to the
    /// fix on record_success (see its own doc comment), a
    /// continuously healthy worker's last_health_update never
    /// advanced past construction time, so this doc comment's stated
    /// intent and the code's actual behavior disagreed for exactly
    /// that case.
    pub fn mark_stale_unavailable(&self, timeout: Duration) {
        let mut workers = self.workers.write().unwrap();
        let now = Instant::now();
        for entry in workers.values_mut() {
            if entry.health == WorkerHealth::Healthy
                && now.duration_since(entry.last_health_update) > timeout
            {
                entry.health = WorkerHealth::Unavailable;
                entry.last_health_update = now;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str) -> WorkerSpec {
        WorkerSpec::new(id, "127.0.0.1:11434".to_string())
    }

    #[test]
    fn register_and_retrieve() {
        let registry = WorkerRegistry::new();
        registry.register(spec("worker-0"));
        registry.register(spec("worker-1"));

        assert_eq!(registry.len(), 2);
        let routable = registry.routable_workers();
        assert_eq!(routable.len(), 2);
    }

    #[test]
    fn deregister_removes_worker() {
        let registry = WorkerRegistry::new();
        registry.register(spec("worker-0"));
        registry.register(spec("worker-1"));
        registry.deregister("worker-0");

        assert_eq!(registry.len(), 1);
        let routable = registry.routable_workers();
        assert_eq!(routable.len(), 1);
        assert_eq!(routable[0].worker_id, "worker-1");
    }

    #[test]
    fn three_consecutive_failures_degrade_worker() {
        let registry = WorkerRegistry::new();
        registry.register(spec("worker-0"));

        for _ in 0..3 {
            registry.record_failure("worker-0");
        }

        assert_eq!(registry.health("worker-0"), Some(WorkerHealth::Degraded));
        // Degraded workers are still routable
        assert_eq!(registry.routable_workers().len(), 1);
    }

    #[test]
    fn ten_consecutive_failures_make_unavailable() {
        let registry = WorkerRegistry::new();
        registry.register(spec("worker-0"));
        registry.register(spec("worker-1"));

        for _ in 0..10 {
            registry.record_failure("worker-0");
        }

        assert_eq!(registry.health("worker-0"), Some(WorkerHealth::Unavailable));
        // Unavailable workers are excluded from routing
        let routable = registry.routable_workers();
        assert_eq!(routable.len(), 1);
        assert_eq!(routable[0].worker_id, "worker-1");
    }

    #[test]
    fn success_after_degraded_restores_healthy() {
        let registry = WorkerRegistry::new();
        registry.register(spec("worker-0"));

        for _ in 0..5 {
            registry.record_failure("worker-0");
        }
        assert_eq!(registry.health("worker-0"), Some(WorkerHealth::Degraded));

        registry.record_success("worker-0");
        assert_eq!(registry.health("worker-0"), Some(WorkerHealth::Healthy));
    }

    #[test]
    fn set_health_works_for_chaos_injection() {
        let registry = WorkerRegistry::new();
        registry.register(spec("worker-0"));

        registry.set_health("worker-0", WorkerHealth::Unavailable);
        assert_eq!(registry.routable_workers().len(), 0);

        registry.set_health("worker-0", WorkerHealth::Healthy);
        assert_eq!(registry.routable_workers().len(), 1);
    }

    #[test]
    fn weight_multipliers_are_correct() {
        assert_eq!(WorkerHealth::Healthy.weight_multiplier(), 1.0);
        assert_eq!(WorkerHealth::Degraded.weight_multiplier(), 0.3);
        assert_eq!(WorkerHealth::Unavailable.weight_multiplier(), 0.0);
    }

    /// Regression test for the real bug: WorkerEntry::record_success
    /// used to only refresh last_health_update inside the
    /// Degraded -> Healthy branch, so calling it on an already-Healthy
    /// worker (the common, correct case) left last_health_update
    /// frozen at construction time. This test calls record_success
    /// directly on a WorkerEntry that starts and stays Healthy, and
    /// checks the timestamp actually moves forward.
    #[test]
    fn record_success_refreshes_timestamp_even_when_already_healthy() {
        let mut entry = WorkerEntry::new(spec("worker-0"));
        assert_eq!(entry.health, WorkerHealth::Healthy);
        let initial_ts = entry.last_health_update;

        std::thread::sleep(Duration::from_millis(20));
        entry.record_success();

        assert_eq!(
            entry.health,
            WorkerHealth::Healthy,
            "a success on an already-Healthy worker must not change its health state"
        );
        assert!(
            entry.last_health_update > initial_ts,
            "last_health_update must advance on every record_success call, \
             not just on a Degraded -> Healthy transition"
        );
    }

    /// Registry-level counterpart of the test above: a worker that
    /// keeps receiving real successes, and never degrades, must never
    /// be flagged stale by mark_stale_unavailable, no matter how long
    /// it has been Healthy overall. Before the fix, this worker's
    /// last_health_update was frozen at registration time, so a
    /// staleness sweep run long after registration (but shortly after
    /// the worker's most recent real success) would have incorrectly
    /// marked it Unavailable.
    #[test]
    fn continuously_healthy_worker_with_ongoing_successes_is_never_marked_stale() {
        let registry = WorkerRegistry::new();
        registry.register(spec("worker-0"));

        // Simulate time passing since registration, during which the
        // worker has been correctly, repeatedly succeeding.
        std::thread::sleep(Duration::from_millis(30));
        registry.record_success("worker-0");

        // A staleness sweep with a timeout shorter than the total time
        // since registration, but longer than the time since the last
        // record_success call, must NOT mark this worker stale: the
        // fix's whole point is that last_health_update tracks the last
        // real success, not registration time.
        registry.mark_stale_unavailable(Duration::from_millis(15));

        assert_eq!(
            registry.health("worker-0"),
            Some(WorkerHealth::Healthy),
            "a worker with a recent real success must not be marked stale, \
             even if a long time has passed since it was first registered"
        );
    }

    /// The direct counterpart to the test above: a worker that is
    /// registered but never receives any success at all must still be
    /// correctly marked stale once the timeout elapses. This confirms
    /// the fix does not accidentally disable staleness detection
    /// entirely; it must still fire for a worker that genuinely never
    /// succeeds.
    #[test]
    fn worker_with_no_successes_is_still_correctly_marked_stale() {
        let registry = WorkerRegistry::new();
        registry.register(spec("worker-0"));

        std::thread::sleep(Duration::from_millis(20));
        registry.mark_stale_unavailable(Duration::from_millis(5));

        assert_eq!(
            registry.health("worker-0"),
            Some(WorkerHealth::Unavailable),
            "a worker that never had a successful routing outcome must \
             still be marked stale once the timeout elapses"
        );
    }
}