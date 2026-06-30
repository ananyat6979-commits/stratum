//! Backpressure and flow control for the router.
//!
//! The router applies backpressure when workers are saturated:
//! requests are rejected with a structured error rather than queued
//! indefinitely. This prevents head-of-line blocking and maintains
//! P99 latency under overload.
//!
//! # Design
//! Three token buckets, one per SLA class (REALTIME, INTERACTIVE, BATCH).
//! Each bucket is independent — BATCH saturation never blocks REALTIME.
//! Within each SLA class, requests are admitted on a first-come basis.
//!
//! The router checks the backpressure system BEFORE making a routing
//! decision. A rejected request never reaches the scoring function or
//! the worker — this prevents the oracle from being queried under load.
//!
//! # Token Bucket Parameters
//! Parameters are intentionally conservative defaults. Phase 3's
//! baseline benchmark will establish the correct values for this hardware.
//! See `BucketConfig` for the tuning rationale.
//!
//! # Relationship to stratum-gateway rate limiter
//! The gateway's rate limiter (rate_limit.rs) controls ingress rate —
//! how many requests per second enter STRATUM at all.
//! The router's backpressure controls queueing at the worker tier —
//! how many requests are in-flight to workers simultaneously.
//! These are complementary: the gateway limits arrival rate,
//! the router limits concurrency.

use std::sync::Mutex;
use std::time::Instant;

/// SLA class for routing backpressure decisions.
/// Mirrors the gateway's SlaClass but is defined independently
/// to avoid a hard coupling between stratum-router and stratum-gateway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterSlaClass {
    Realtime,
    Interactive,
    Batch,
}

/// Result of a backpressure check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureDecision {
    /// Request is admitted. Routing proceeds.
    Admit,
    /// Request is shed. Caller should return 429 to the client.
    Shed,
}

/// Configuration for a single token bucket.
#[derive(Debug, Clone, Copy)]
pub struct BucketConfig {
    /// Maximum number of tokens (burst capacity).
    pub capacity: f64,
    /// Token refill rate in tokens per second.
    pub refill_rate_per_sec: f64,
}

/// Lazy-refill token bucket (same design as gateway rate_limit.rs).
struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate_per_sec: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(config: BucketConfig) -> Self {
        Self {
            capacity: config.capacity,
            tokens: config.capacity,
            refill_rate_per_sec: config.refill_rate_per_sec,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate_per_sec).min(self.capacity);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Per-SLA-class backpressure controller.
///
/// Thread-safe via per-bucket mutexes. The three mutexes are independent —
/// a slow BATCH consumer never blocks a REALTIME admission check.
pub struct BackpressureController {
    realtime: Mutex<TokenBucket>,
    interactive: Mutex<TokenBucket>,
    batch: Mutex<TokenBucket>,
}

impl BackpressureController {
    /// Construct with explicit per-class configuration.
    pub fn new(realtime: BucketConfig, interactive: BucketConfig, batch: BucketConfig) -> Self {
        Self {
            realtime: Mutex::new(TokenBucket::new(realtime)),
            interactive: Mutex::new(TokenBucket::new(interactive)),
            batch: Mutex::new(TokenBucket::new(batch)),
        }
    }

    /// Construct with default configuration.
    ///
    /// Defaults are deliberately conservative — tuned for a 2-worker
    /// laptop cluster. Increase for server-class hardware.
    ///
    /// REALTIME: small capacity (10), fast refill (10/sec)
    ///   → admits bursts up to 10, sustained rate 10 RPS
    /// INTERACTIVE: medium capacity (50), medium refill (50/sec)
    /// BATCH: large capacity (200), slower refill (100/sec)
    ///   → tolerates larger bursts, smoothed over time
    pub fn with_defaults() -> Self {
        Self::new(
            BucketConfig {
                capacity: 10.0,
                refill_rate_per_sec: 10.0,
            },
            BucketConfig {
                capacity: 50.0,
                refill_rate_per_sec: 50.0,
            },
            BucketConfig {
                capacity: 200.0,
                refill_rate_per_sec: 100.0,
            },
        )
    }

    /// Check whether a request of the given SLA class should be admitted.
    ///
    /// Consumes one token if admitted. Tokens are not returned if the
    /// subsequent routing call fails — backpressure is conservative
    /// under failures.
    pub fn check(&self, sla_class: RouterSlaClass) -> BackpressureDecision {
        let admitted = match sla_class {
            RouterSlaClass::Realtime => self.realtime.lock().unwrap().try_consume(),
            RouterSlaClass::Interactive => self.interactive.lock().unwrap().try_consume(),
            RouterSlaClass::Batch => self.batch.lock().unwrap().try_consume(),
        };
        if admitted {
            BackpressureDecision::Admit
        } else {
            BackpressureDecision::Shed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_controller() -> BackpressureController {
        BackpressureController::new(
            BucketConfig {
                capacity: 3.0,
                refill_rate_per_sec: 100.0,
            },
            BucketConfig {
                capacity: 3.0,
                refill_rate_per_sec: 100.0,
            },
            BucketConfig {
                capacity: 3.0,
                refill_rate_per_sec: 100.0,
            },
        )
    }

    #[test]
    fn admits_within_capacity() {
        let bp = small_controller();
        assert_eq!(
            bp.check(RouterSlaClass::Realtime),
            BackpressureDecision::Admit
        );
        assert_eq!(
            bp.check(RouterSlaClass::Realtime),
            BackpressureDecision::Admit
        );
        assert_eq!(
            bp.check(RouterSlaClass::Realtime),
            BackpressureDecision::Admit
        );
    }

    #[test]
    fn sheds_when_exhausted() {
        let bp = small_controller();
        bp.check(RouterSlaClass::Realtime);
        bp.check(RouterSlaClass::Realtime);
        bp.check(RouterSlaClass::Realtime);
        assert_eq!(
            bp.check(RouterSlaClass::Realtime),
            BackpressureDecision::Shed
        );
    }

    #[test]
    fn sla_classes_are_independent() {
        let bp = small_controller();
        // Exhaust REALTIME
        for _ in 0..3 {
            bp.check(RouterSlaClass::Realtime);
        }
        assert_eq!(
            bp.check(RouterSlaClass::Realtime),
            BackpressureDecision::Shed
        );
        // INTERACTIVE and BATCH unaffected
        assert_eq!(
            bp.check(RouterSlaClass::Interactive),
            BackpressureDecision::Admit
        );
        assert_eq!(bp.check(RouterSlaClass::Batch), BackpressureDecision::Admit);
    }

    #[test]
    fn refills_over_time() {
        let bp = small_controller();
        // Exhaust
        for _ in 0..3 {
            bp.check(RouterSlaClass::Batch);
        }
        assert_eq!(bp.check(RouterSlaClass::Batch), BackpressureDecision::Shed);
        // Wait for refill (100/sec → 1 token per 10ms)
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(bp.check(RouterSlaClass::Batch), BackpressureDecision::Admit);
    }

    #[test]
    fn with_defaults_constructs_without_panic() {
        let bp = BackpressureController::with_defaults();
        assert_eq!(
            bp.check(RouterSlaClass::Realtime),
            BackpressureDecision::Admit
        );
        assert_eq!(
            bp.check(RouterSlaClass::Interactive),
            BackpressureDecision::Admit
        );
        assert_eq!(bp.check(RouterSlaClass::Batch), BackpressureDecision::Admit);
    }
}
