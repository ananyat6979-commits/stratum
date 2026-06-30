//! Multi-objective routing score function for the semantic router.
//!
//! The routing decision selects the worker with the highest score:
//!
//!   score(worker, request) =
//!     α · cache_hit_probability
//!   + β · (1 / predicted_latency_normalized)
//!   + γ · sla_affinity
//!   + δ · (1 - kv_pressure)
//!
//! where α + β + γ + δ = 1.0 (weights are normalized).
//!
//! Weights are learned by the LinUCB bandit in `bandit.rs`.
//! Default weights are equal (0.25 each) — the bandit converges
//! to optimal weights after ~200-400 requests per worker.
//!
//! # Signal Definitions
//!
//! `cache_hit_probability`: estimated probability that routing this
//! request to this worker will result in a KV cache hit. Computed by
//! the cache oracle (Phase 3: FAISS IVF-PQ ANN search over request
//! embeddings). Range: [0.0, 1.0].
//!
//! `predicted_latency_ms`: predicted end-to-end latency for this request
//! on this worker, in milliseconds. Normalized to [0.0, 1.0] by dividing
//! by the maximum observed latency across all workers. Lower latency =
//! higher score, hence (1 / normalized_latency).
//!
//! `sla_affinity`: how well this worker's current load profile matches
//! the request's SLA class. REALTIME requests prefer low-pressure workers;
//! BATCH requests are less sensitive. Range: [0.0, 1.0].
//!
//! `kv_pressure`: current KV cache block table utilization on this worker.
//! High pressure = high eviction risk = avoid. Range: [0.0, 1.0].

use crate::router::WorkerSpec;

/// The four input signals for a single worker-request pair.
#[derive(Debug, Clone, Copy)]
pub struct RoutingSignals {
    /// Estimated KV cache hit probability for this request on this worker.
    /// Computed by the cache oracle. 0.0 before the oracle is warmed up.
    pub cache_hit_prob: f64,

    /// Predicted latency for this request on this worker, in milliseconds.
    /// Used to compute the inverse-latency score component.
    /// Must be > 0.0. Use a conservative default (e.g., 1000ms) when unknown.
    pub predicted_latency_ms: f64,

    /// SLA affinity score for this worker given the request's SLA class.
    /// 1.0 = perfect match, 0.0 = worst possible match.
    pub sla_affinity: f64,

    /// Current KV cache pressure on this worker. 0.0 = empty, 1.0 = full.
    /// From the cache oracle's block table utilization metric.
    pub kv_pressure: f64,
}

impl RoutingSignals {
    /// Construct signals with safe defaults for use before the oracle warms up.
    ///
    /// All signals default to neutral values that produce equal scores across
    /// workers, effectively falling back to score-neutral selection (which
    /// the caller breaks via consistent hash or round-robin).
    pub fn neutral() -> Self {
        Self {
            cache_hit_prob: 0.0,
            predicted_latency_ms: 100.0,
            sla_affinity: 0.5,
            kv_pressure: 0.0,
        }
    }
}

/// Routing score weights. Must sum to 1.0 (enforced by constructor).
#[derive(Debug, Clone, Copy)]
pub struct ScoreWeights {
    pub cache_hit_prob: f64,
    pub inverse_latency: f64,
    pub sla_affinity: f64,
    pub pressure_avoidance: f64,
}

impl ScoreWeights {
    /// Construct weights, normalizing so they sum to 1.0.
    ///
    /// # Panics
    /// Panics if all weights are zero (undefined normalization).
    pub fn new(cache_hit: f64, inv_latency: f64, sla: f64, pressure: f64) -> Self {
        let total = cache_hit + inv_latency + sla + pressure;
        assert!(total > 0.0, "at least one weight must be positive");
        Self {
            cache_hit_prob: cache_hit / total,
            inverse_latency: inv_latency / total,
            sla_affinity: sla / total,
            pressure_avoidance: pressure / total,
        }
    }

    /// Equal weights — the initial state before LinUCB has learned anything.
    pub fn equal() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    /// Verify the invariant that weights sum to 1.0 (within floating point tolerance).
    pub fn is_normalized(&self) -> bool {
        let sum = self.cache_hit_prob
            + self.inverse_latency
            + self.sla_affinity
            + self.pressure_avoidance;
        (sum - 1.0).abs() < 1e-6
    }
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self::equal()
    }
}

/// Compute the routing score for a single worker given its signals and weights.
///
/// Returns a value in approximately [0.0, 1.0]. Values outside this range
/// are possible when latency normalization produces extreme values.
///
/// # Arguments
/// * `signals` — the four oracle signals for this worker
/// * `weights` — the current bandit weight vector
/// * `max_latency_ms` — the maximum predicted latency across all workers,
///   used to normalize the latency component. Must be > 0.0.
pub fn compute_score(signals: &RoutingSignals, weights: &ScoreWeights, max_latency_ms: f64) -> f64 {
    debug_assert!(max_latency_ms > 0.0, "max_latency_ms must be > 0");
    debug_assert!(weights.is_normalized(), "weights must sum to 1.0");

    // Normalize latency to [0, 1] and invert: lower latency = higher score
    let normalized_latency = signals.predicted_latency_ms / max_latency_ms;
    let inv_latency_score = 1.0 - normalized_latency.min(1.0);

    // Pressure avoidance: high pressure = low score
    let pressure_score = 1.0 - signals.kv_pressure.clamp(0.0, 1.0);

    weights.cache_hit_prob * signals.cache_hit_prob.clamp(0.0, 1.0)
        + weights.inverse_latency * inv_latency_score
        + weights.sla_affinity * signals.sla_affinity.clamp(0.0, 1.0)
        + weights.pressure_avoidance * pressure_score
}

/// Select the best worker from a set given per-worker signals.
///
/// Returns the index of the selected worker in `workers`.
/// Ties are broken by worker index (lower index wins) — deterministic.
///
/// Returns `None` if `workers` is empty.
pub fn select_best_worker(
    workers: &[WorkerSpec],
    signals: &[RoutingSignals],
    weights: &ScoreWeights,
) -> Option<usize> {
    assert_eq!(
        workers.len(),
        signals.len(),
        "workers and signals must have the same length"
    );

    if workers.is_empty() {
        return None;
    }

    let max_latency = signals
        .iter()
        .map(|s| s.predicted_latency_ms)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0); // guard against zero max_latency

    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (i, signal) in signals.iter().enumerate() {
        let score = compute_score(signal, weights, max_latency);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    Some(best_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_worker(i: u32) -> WorkerSpec {
        WorkerSpec::new(format!("worker-{i}"), format!("127.0.0.1:{}", 11434 + i))
    }

    #[test]
    fn equal_weights_normalized_to_025_each() {
        let w = ScoreWeights::equal();
        assert!((w.cache_hit_prob - 0.25).abs() < 1e-10);
        assert!((w.inverse_latency - 0.25).abs() < 1e-10);
        assert!((w.sla_affinity - 0.25).abs() < 1e-10);
        assert!((w.pressure_avoidance - 0.25).abs() < 1e-10);
        assert!(w.is_normalized());
    }

    #[test]
    fn custom_weights_are_normalized() {
        let w = ScoreWeights::new(2.0, 1.0, 1.0, 0.0);
        assert!(w.is_normalized());
        assert!((w.cache_hit_prob - 0.5).abs() < 1e-10);
    }

    #[test]
    fn high_cache_hit_prob_increases_score() {
        let weights = ScoreWeights::new(1.0, 0.0, 0.0, 0.0); // cache only
        let low = RoutingSignals {
            cache_hit_prob: 0.1,
            ..RoutingSignals::neutral()
        };
        let high = RoutingSignals {
            cache_hit_prob: 0.9,
            ..RoutingSignals::neutral()
        };
        let s_low = compute_score(&low, &weights, 100.0);
        let s_high = compute_score(&high, &weights, 100.0);
        assert!(s_high > s_low);
    }

    #[test]
    fn high_kv_pressure_decreases_score() {
        let weights = ScoreWeights::new(0.0, 0.0, 0.0, 1.0); // pressure only
        let low_pressure = RoutingSignals {
            kv_pressure: 0.1,
            ..RoutingSignals::neutral()
        };
        let high_pressure = RoutingSignals {
            kv_pressure: 0.9,
            ..RoutingSignals::neutral()
        };
        let s_low = compute_score(&low_pressure, &weights, 100.0);
        let s_high = compute_score(&high_pressure, &weights, 100.0);
        assert!(s_low > s_high);
    }

    #[test]
    fn lower_latency_gives_higher_score() {
        let weights = ScoreWeights::new(0.0, 1.0, 0.0, 0.0); // latency only
        let fast = RoutingSignals {
            predicted_latency_ms: 10.0,
            ..RoutingSignals::neutral()
        };
        let slow = RoutingSignals {
            predicted_latency_ms: 900.0,
            ..RoutingSignals::neutral()
        };
        let max_latency = 1000.0;
        let s_fast = compute_score(&fast, &weights, max_latency);
        let s_slow = compute_score(&slow, &weights, max_latency);
        assert!(s_fast > s_slow);
    }

    #[test]
    fn select_best_worker_chooses_highest_score() {
        let workers = vec![test_worker(0), test_worker(1), test_worker(2)];
        let signals = vec![
            RoutingSignals {
                cache_hit_prob: 0.1,
                kv_pressure: 0.8,
                ..RoutingSignals::neutral()
            },
            RoutingSignals {
                cache_hit_prob: 0.9,
                kv_pressure: 0.1,
                ..RoutingSignals::neutral()
            }, // best
            RoutingSignals {
                cache_hit_prob: 0.5,
                kv_pressure: 0.5,
                ..RoutingSignals::neutral()
            },
        ];
        let weights = ScoreWeights::equal();
        let best = select_best_worker(&workers, &signals, &weights).unwrap();
        assert_eq!(
            best, 1,
            "worker-1 should be selected (highest cache hit, lowest pressure)"
        );
    }

    #[test]
    fn select_best_worker_returns_none_for_empty() {
        let result = select_best_worker(&[], &[], &ScoreWeights::equal());
        assert_eq!(result, None);
    }

    #[test]
    fn score_is_in_unit_range_for_valid_inputs() {
        let weights = ScoreWeights::equal();
        let signals = RoutingSignals {
            cache_hit_prob: 0.7,
            predicted_latency_ms: 200.0,
            sla_affinity: 0.8,
            kv_pressure: 0.3,
        };
        let score = compute_score(&signals, &weights, 1000.0);
        assert!((0.0..=1.0).contains(&score), "score {score} out of [0,1]");
    }

    #[test]
    fn tie_breaking_favors_lower_index() {
        let workers = vec![test_worker(0), test_worker(1)];
        // Identical signals -> identical scores -> lower index wins
        let signals = vec![RoutingSignals::neutral(), RoutingSignals::neutral()];
        let best = select_best_worker(&workers, &signals, &ScoreWeights::equal()).unwrap();
        assert_eq!(best, 0);
    }
}
