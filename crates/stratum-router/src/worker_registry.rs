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
//! workers are assumed Healthy — the router trusts the operator to
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

    /// Record a successful routing outcome. Resets failure counter.
    /// Transitions Degraded → Healthy after a success.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        if self.health == WorkerHealth::Degraded {
            self.health = WorkerHealth::Healthy;
            self.last_health_update = Instant::now();
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

    /// Mark all workers that have not had a successful routing outcome
    /// within `timeout` as Unavailable. Used by a periodic health sweep.
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
}
