//! Token bucket rate limiting, per SLA class.
//!
//! Each [`SlaClass`] gets an independent token bucket. This prevents
//! high-volume BATCH traffic from starving REALTIME requests of capacity —
//! the failure mode documented in the blueprint's backpressure design
//! (separate buckets per SLA class with strict priority ordering).
//!
//! # Algorithm
//! Standard token bucket: capacity `C`, refill rate `R` tokens/sec.
//! - Bucket starts full (C tokens)
//! - Tokens refill continuously at rate R (computed lazily on each check,
//!   not via a background timer — avoids spawning a task per bucket)
//! - A request consumes 1 token. If the bucket has < 1 token, the request
//!   is rejected (caller should return HTTP 429)
//!
//! # Why lazy refill, not a background timer
//! A timer-based refill requires either a tokio task per bucket (3 buckets
//! here, but this pattern doesn't scale if buckets become per-tenant) or
//! a single shared timer with bucket iteration (adds coordination complexity).
//! Lazy refill computes "tokens accumulated since last check" on demand —
//! O(1), no background tasks, no drift.

use std::sync::Mutex;
use std::time::Instant;

use crate::sla::SlaClass;

/// A single token bucket. Thread-safe via internal mutex — multiple
/// gateway worker threads check the same bucket concurrently.
struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate_per_sec: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: f64, refill_rate_per_sec: f64) -> Self {
        Self {
            capacity,
            tokens: capacity, // start full
            refill_rate_per_sec,
            last_refill: Instant::now(),
        }
    }

    /// Attempt to consume one token. Returns `true` if a token was
    /// available and consumed, `false` if the bucket is empty.
    ///
    /// Refills the bucket based on elapsed time before checking,
    /// capped at `capacity` (tokens don't accumulate unboundedly
    /// while idle).
    fn try_consume(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();

        let refill_amount = elapsed * self.refill_rate_per_sec;
        self.tokens = (self.tokens + refill_amount).min(self.capacity);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Per-SLA-class rate limiter.
///
/// # Configuration Rationale
/// REALTIME gets the smallest bucket but is never meant to be the
/// highest-volume class — it represents latency-sensitive interactive
/// traffic. BATCH gets the largest bucket and highest refill rate since
/// it represents bulk/background work that can tolerate throttling.
///
/// These defaults are starting points (documented as such), not tuned
/// production values — tuning requires the Phase 1 baseline benchmark
/// to establish realistic per-class request rates.
pub struct RateLimiter {
    realtime: Mutex<TokenBucket>,
    interactive: Mutex<TokenBucket>,
    batch: Mutex<TokenBucket>,
}

/// Configuration for a single SLA class's token bucket.
#[derive(Debug, Clone, Copy)]
pub struct BucketConfig {
    pub capacity: f64,
    pub refill_rate_per_sec: f64,
}

impl RateLimiter {
    /// Construct a rate limiter with explicit per-class configuration.
    pub fn new(realtime: BucketConfig, interactive: BucketConfig, batch: BucketConfig) -> Self {
        Self {
            realtime: Mutex::new(TokenBucket::new(
                realtime.capacity,
                realtime.refill_rate_per_sec,
            )),
            interactive: Mutex::new(TokenBucket::new(
                interactive.capacity,
                interactive.refill_rate_per_sec,
            )),
            batch: Mutex::new(TokenBucket::new(batch.capacity, batch.refill_rate_per_sec)),
        }
    }

    /// Construct a rate limiter with documented default configuration.
    ///
    /// Defaults (requests/sec sustained, burst capacity):
    /// - REALTIME:    10 capacity,  10/sec refill — low burst, steady rate
    /// - INTERACTIVE: 50 capacity,  50/sec refill
    /// - BATCH:       200 capacity, 100/sec refill — large burst allowance,
    ///   refill rate below capacity to encourage smoothing over time
    ///
    /// These are placeholder defaults pending Phase 1 baseline benchmark data.
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

    /// Check whether a request of the given SLA class is allowed.
    ///
    /// Returns `true` if a token was consumed (request allowed),
    /// `false` if the bucket for this class is exhausted (request
    /// should be rejected with HTTP 429).
    ///
    /// # Cancellation safety
    /// Synchronous, mutex-guarded. If the calling future is cancelled
    /// after this returns `true`, the token is still consumed — there
    /// is no way to "return" a consumed token. This is intentional:
    /// rate limiting must be conservative under cancellation.
    pub fn check(&self, class: SlaClass) -> bool {
        let bucket = match class {
            SlaClass::Realtime => &self.realtime,
            SlaClass::Interactive => &self.interactive,
            SlaClass::Batch => &self.batch,
        };

        let mut bucket = bucket.lock().expect("rate limiter mutex poisoned");
        bucket.try_consume()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    /// Helper: a limiter with a small, fast-refilling bucket for one class,
    /// to keep tests fast without sleeping for real-world durations.
    fn small_limiter() -> RateLimiter {
        RateLimiter::new(
            BucketConfig {
                capacity: 2.0,
                refill_rate_per_sec: 100.0,
            }, // realtime
            BucketConfig {
                capacity: 2.0,
                refill_rate_per_sec: 100.0,
            }, // interactive
            BucketConfig {
                capacity: 2.0,
                refill_rate_per_sec: 100.0,
            }, // batch
        )
    }

    #[test]
    fn bucket_starts_full_and_allows_capacity_requests() {
        let limiter = small_limiter();
        // Capacity is 2.0 — first two requests succeed immediately
        assert!(limiter.check(SlaClass::Realtime));
        assert!(limiter.check(SlaClass::Realtime));
    }

    #[test]
    fn bucket_rejects_when_exhausted() {
        let limiter = small_limiter();
        assert!(limiter.check(SlaClass::Realtime));
        assert!(limiter.check(SlaClass::Realtime));
        // Third immediate request exceeds capacity=2.0 with negligible refill
        assert!(!limiter.check(SlaClass::Realtime));
    }

    #[test]
    fn bucket_refills_over_time() {
        let limiter = small_limiter();
        // Drain the bucket
        assert!(limiter.check(SlaClass::Realtime));
        assert!(limiter.check(SlaClass::Realtime));
        assert!(!limiter.check(SlaClass::Realtime));

        // Refill rate is 100/sec → 1 token takes 10ms. Sleep 20ms for margin.
        sleep(Duration::from_millis(20));

        assert!(
            limiter.check(SlaClass::Realtime),
            "bucket should have refilled at least one token after 20ms at 100/sec"
        );
    }

    #[test]
    fn refill_does_not_exceed_capacity() {
        let limiter = small_limiter();
        // Bucket starts full (2.0). Sleep long enough that naive refill
        // would overflow capacity if not capped.
        sleep(Duration::from_millis(100));

        // Should be able to consume exactly 2 tokens (capacity), not more.
        assert!(limiter.check(SlaClass::Realtime));
        assert!(limiter.check(SlaClass::Realtime));
        assert!(
            !limiter.check(SlaClass::Realtime),
            "refill must be capped at capacity, even after long idle period"
        );
    }

    #[test]
    fn sla_classes_have_independent_buckets() {
        let limiter = small_limiter();
        // Exhaust REALTIME bucket
        assert!(limiter.check(SlaClass::Realtime));
        assert!(limiter.check(SlaClass::Realtime));
        assert!(!limiter.check(SlaClass::Realtime));

        // INTERACTIVE and BATCH buckets must be unaffected
        assert!(limiter.check(SlaClass::Interactive));
        assert!(limiter.check(SlaClass::Batch));
    }

    #[test]
    fn with_defaults_constructs_without_panic() {
        let limiter = RateLimiter::with_defaults();
        // Sanity: each class should allow at least one request immediately
        assert!(limiter.check(SlaClass::Realtime));
        assert!(limiter.check(SlaClass::Interactive));
        assert!(limiter.check(SlaClass::Batch));
    }

    #[test]
    fn default_batch_bucket_has_larger_capacity_than_realtime() {
        // Documents the design intent: BATCH tolerates larger bursts.
        // Drain REALTIME's default capacity (10), confirm BATCH still
        // has headroom well beyond that under the same call count.
        let limiter = RateLimiter::with_defaults();

        for _ in 0..10 {
            assert!(limiter.check(SlaClass::Realtime));
        }
        assert!(
            !limiter.check(SlaClass::Realtime),
            "realtime should be exhausted after 10 (capacity=10)"
        );

        for _ in 0..10 {
            assert!(limiter.check(SlaClass::Batch));
        }
        assert!(
            limiter.check(SlaClass::Batch),
            "batch (capacity=200) should still have tokens after only 10 consumed"
        );
    }
}
