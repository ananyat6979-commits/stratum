//! LinUCB contextual bandit for adaptive routing weight learning.
//!
//! The bandit learns which routing signal weights (cache_hit_prob,
//! inverse_latency, sla_affinity, pressure_avoidance) produce the
//! best outcomes (lowest latency, highest cache hit rate) over time.
//!
//! # Why LinUCB
//! The routing problem is a contextual bandit: the reward depends on
//! both the action (which worker to route to) and the context (request
//! features, current worker states). LinUCB models the reward as a
//! linear function of the context and provides an exploration bonus
//! (the UCB term) that prevents premature convergence to a local optimum.
//!
//! Regret bound: O(√(d·T·log³(T/δ))) where d=4 (feature dimension).
//! With d=4, convergence is fast — approximately 200-400 requests per
//! arm to reach near-optimal weights.
//!
//! # Reference
//! Li et al. (2010). "A Contextual-Bandit Approach to Personalized News
//! Article Recommendation." WWW 2010. Algorithm 1 (LinUCB with Disjoint
//! Linear Models).
//!
//! # Feature Vector
//! The context vector for each arm (worker) is the 4-dimensional signal
//! vector: [cache_hit_prob, inv_latency_score, sla_affinity, pressure_score].
//! This matches the routing score function in `scoring.rs`.

use crate::scoring::{RoutingSignals, ScoreWeights};

const D: usize = 4; // Feature dimension: matches ScoreWeights field count

/// A single LinUCB arm (one per routing strategy variant or per worker).
///
/// Each arm maintains sufficient statistics for the ridge regression:
///   A = I_d + X^T X  (d×d, positive definite)
///   b = X^T r        (d-vector, reward-weighted)
///
/// The optimal weight vector is: θ̂ = A^{-1} b
#[derive(Debug, Clone)]
pub struct LinUcbArm {
    /// A = I + sum of outer products of context vectors.
    /// Initialized to identity to implement L2 regularization.
    a: [[f64; D]; D],
    /// b = sum of reward-weighted context vectors.
    b: [f64; D],
    /// Number of times this arm has been selected.
    pub n_pulls: u64,
}

impl LinUcbArm {
    pub fn new() -> Self {
        let mut a = [[0.0f64; D]; D];
        #[allow(clippy::needless_range_loop)]
        for i in 0..D {
            a[i][i] = 1.0; // identity matrix
        }
        Self {
            a,
            b: [0.0; D],
            n_pulls: 0,
        }
    }

    /// Update arm parameters after observing a reward for a given context.
    ///
    /// # Arguments
    /// * `context` — the 4-dimensional feature vector
    /// * `reward` — observed reward (e.g., normalized inverse latency)
    pub fn update(&mut self, context: &[f64; D], reward: f64) {
        // A += context * context^T
        for (i, row) in self.a.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += context[i] * context[j];
            }
        }
        // b += reward * context
        for (i, b_i) in self.b.iter_mut().enumerate() {
            *b_i += reward * context[i];
        }
        self.n_pulls += 1;
    }

    /// Compute the UCB score for this arm given a context vector.
    ///
    /// UCB = θ̂^T x + alpha * sqrt(x^T A^{-1} x)
    ///     = expected_reward + exploration_bonus
    ///
    /// The exploration bonus decreases as more data is collected (A grows)
    /// and is larger in directions with high uncertainty (low data density).
    ///
    /// # Arguments
    /// * `context` — the 4-dimensional feature vector
    /// * `alpha` — exploration parameter. Higher = more exploration.
    ///   Theoretically: alpha = 1 + sqrt(ln(2/δ)/2) for δ-correct UCB.
    ///   In practice: tuned empirically. Start with alpha=1.0.
    pub fn ucb_score(&self, context: &[f64; D], alpha: f64) -> f64 {
        // Solve A * theta_hat = b via Gaussian elimination
        // (A is 4x4, so this is O(d^3) = O(64) — negligible overhead)
        let theta_hat = solve_linear_system(&self.a, &self.b);

        // Expected reward: theta_hat^T * context
        let expected = dot(&theta_hat, context);

        // Exploration bonus: alpha * sqrt(context^T * A^{-1} * context)
        let a_inv_context = solve_linear_system(&self.a, context);
        let uncertainty = dot(context, &a_inv_context).max(0.0).sqrt();

        expected + alpha * uncertainty
    }

    /// Extract the current optimal weight vector θ̂ = A^{-1} b.
    ///
    /// This is the estimated best weight combination based on observed rewards.
    /// Used to update the `ScoreWeights` after sufficient data is collected.
    pub fn theta_hat(&self) -> [f64; D] {
        solve_linear_system(&self.a, &self.b)
    }
}

impl Default for LinUcbArm {
    fn default() -> Self {
        Self::new()
    }
}

/// LinUCB bandit for routing weight learning.
///
/// Uses a single arm (disjoint model) over the weight space.
/// The "arm" here is not "which worker to route to" but "which weight
/// vector to use" — the bandit learns the best weights, and the
/// scoring function uses those weights to select the best worker.
pub struct LinUcbBandit {
    arm: LinUcbArm,
    /// Exploration parameter. Currently unused by LinUcbBandit's public
    /// API (current_weights() reads theta_hat directly, not ucb_score()).
    /// Reserved for Phase 4 when the bandit selects among discrete
    /// strategy variants via UCB rather than returning a single learned
    /// weight vector. Kept now rather than removed so the constructor
    /// signature doesn't change when that wiring lands.
    #[allow(dead_code)]
    alpha: f64,
}

impl LinUcbBandit {
    /// Create a new bandit with the given exploration parameter.
    ///
    /// `alpha=1.0` is a reasonable starting point. Consider annealing
    /// alpha as `n_pulls` grows: `alpha = alpha_0 / sqrt(n_pulls + 1)`.
    pub fn new(alpha: f64) -> Self {
        Self {
            arm: LinUcbArm::new(),
            alpha,
        }
    }

    /// Observe a routing outcome and update the bandit.
    ///
    /// # Arguments
    /// * `signals` — the oracle signals at decision time
    /// * `reward` — the observed reward (higher is better).
    ///   Typical choice: `reward = 1.0 - (latency_ms / max_latency_ms)`
    ///   for latency-optimizing routing.
    pub fn update(&mut self, signals: &RoutingSignals, reward: f64) {
        let context = signals_to_context(signals);
        self.arm.update(&context, reward);
    }

    /// Get the current best weight estimate as a `ScoreWeights`.
    ///
    /// Returns `ScoreWeights::equal()` until sufficient data has been
    /// collected (at least `min_pulls` observations). This prevents the
    /// bandit from over-fitting to early noise.
    pub fn current_weights(&self, min_pulls: u64) -> ScoreWeights {
        if self.arm.n_pulls < min_pulls {
            return ScoreWeights::equal();
        }

        let theta = self.arm.theta_hat();

        // Clamp to [0.01, ∞) to prevent negative weights, then normalize.
        // Negative weights could arise from noisy reward signals early on.
        let w0 = theta[0].max(0.01);
        let w1 = theta[1].max(0.01);
        let w2 = theta[2].max(0.01);
        let w3 = theta[3].max(0.01);

        ScoreWeights::new(w0, w1, w2, w3)
    }

    /// Returns the number of observations collected so far.
    pub fn n_pulls(&self) -> u64 {
        self.arm.n_pulls
    }
}

/// Convert a `RoutingSignals` to a 4-dimensional context vector.
///
/// The context vector is [cache_hit_prob, inv_latency_score, sla_affinity, pressure_score]
/// where inv_latency_score and pressure_score are already inverted
/// (higher = better) to keep the linear model's expected reward positive.
fn signals_to_context(signals: &RoutingSignals) -> [f64; D] {
    let inv_latency = (1.0 - (signals.predicted_latency_ms / 10_000.0).min(1.0)).max(0.0);
    let pressure_score = 1.0 - signals.kv_pressure.clamp(0.0, 1.0);
    [
        signals.cache_hit_prob.clamp(0.0, 1.0),
        inv_latency,
        signals.sla_affinity.clamp(0.0, 1.0),
        pressure_score,
    ]
}

/// Dot product of two D-dimensional vectors.
fn dot(a: &[f64; D], b: &[f64; D]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Solve the linear system Ax = b using Gaussian elimination with
/// partial pivoting. A is D×D (4×4 here), b is D-dimensional.
///
/// Returns x = A^{-1} b.
///
/// This is O(D^3) — for D=4, this is 64 operations, negligible on
/// the routing hot path.
fn solve_linear_system(a: &[[f64; D]; D], b: &[f64; D]) -> [f64; D] {
    // Augmented matrix [A | b]
    let mut aug = [[0.0f64; D + 1]; D];
    for i in 0..D {
        aug[i][..D].copy_from_slice(&a[i]);
        aug[i][D] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..D {
        // Find pivot row
        let pivot_row = (col..D)
            .max_by(|&r1, &r2| aug[r1][col].abs().partial_cmp(&aug[r2][col].abs()).unwrap())
            .unwrap();
        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue; // Singular or near-singular — skip (regularization prevents this)
        }

        for row in (col + 1)..D {
            let factor = aug[row][col] / pivot;
            // needless_range_loop: this mutates aug[row] while reading aug[col],
            // which can't be expressed as a single iterator without split_at_mut.
            // D=4 makes this loop's cost negligible; clarity wins over the lint here.
            #[allow(clippy::needless_range_loop)]
            for k in col..=D {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    // Back substitution
    let mut x = [0.0f64; D];
    for i in (0..D).rev() {
        x[i] = aug[i][D];
        for j in (i + 1)..D {
            x[i] -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-12 {
            x[i] /= aug[i][i];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_arm_starts_with_identity_a_and_zero_b() {
        let arm = LinUcbArm::new();
        assert_eq!(arm.n_pulls, 0);
        // A should be identity
        for i in 0..D {
            for j in 0..D {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((arm.a[i][j] - expected).abs() < 1e-10);
            }
        }
        // b should be zero
        for i in 0..D {
            assert_eq!(arm.b[i], 0.0);
        }
    }

    #[test]
    fn update_increments_pull_count() {
        let mut arm = LinUcbArm::new();
        let ctx = [0.5, 0.5, 0.5, 0.5];
        arm.update(&ctx, 1.0);
        assert_eq!(arm.n_pulls, 1);
        arm.update(&ctx, 0.8);
        assert_eq!(arm.n_pulls, 2);
    }

    #[test]
    fn solve_linear_system_identity() {
        // A = I, b = [1,2,3,4] -> x = [1,2,3,4]
        let mut a = [[0.0f64; D]; D];
        #[allow(clippy::needless_range_loop)]
        for i in 0..D {
            a[i][i] = 1.0;
        }
        let b = [1.0, 2.0, 3.0, 4.0];
        let x = solve_linear_system(&a, &b);
        for i in 0..D {
            assert!((x[i] - b[i]).abs() < 1e-9, "x[{i}] = {} != {}", x[i], b[i]);
        }
    }

    #[test]
    fn bandit_returns_equal_weights_before_min_pulls() {
        let bandit = LinUcbBandit::new(1.0);
        let weights = bandit.current_weights(10);
        assert!(weights.is_normalized());
        assert!((weights.cache_hit_prob - 0.25).abs() < 1e-9);
    }

    #[test]
    fn bandit_updates_without_panic() {
        let mut bandit = LinUcbBandit::new(1.0);
        let signals = RoutingSignals {
            cache_hit_prob: 0.8,
            predicted_latency_ms: 50.0,
            sla_affinity: 0.9,
            kv_pressure: 0.1,
        };
        for _ in 0..20 {
            bandit.update(&signals, 0.9);
        }
        let weights = bandit.current_weights(10);
        assert!(weights.is_normalized());
    }

    #[test]
    fn high_reward_for_cache_hits_increases_cache_weight() {
        let mut bandit = LinUcbBandit::new(0.1); // low alpha for faster convergence in test
                                                 // Repeatedly reward high-cache-hit signals
        let good_signals = RoutingSignals {
            cache_hit_prob: 0.95,
            predicted_latency_ms: 100.0,
            sla_affinity: 0.5,
            kv_pressure: 0.1,
        };
        for _ in 0..100 {
            bandit.update(&good_signals, 1.0);
        }
        // After 100 high-cache-hit observations with high reward,
        // cache_hit_prob weight should be elevated above 0.25
        let weights = bandit.current_weights(10);
        assert!(
            weights.cache_hit_prob > 0.25,
            "expected cache weight > 0.25 after high-reward cache-hit training, got {}",
            weights.cache_hit_prob
        );
    }

    #[test]
    fn signals_to_context_inverts_latency_and_pressure() {
        let signals = RoutingSignals {
            cache_hit_prob: 0.7,
            predicted_latency_ms: 0.0, // zero latency -> max inv_latency
            sla_affinity: 0.8,
            kv_pressure: 0.0, // zero pressure -> max pressure_score
        };
        let ctx = signals_to_context(&signals);
        assert!((ctx[0] - 0.7).abs() < 1e-9); // cache_hit_prob unchanged
        assert!((ctx[1] - 1.0).abs() < 1e-9); // zero latency -> 1.0
        assert!((ctx[2] - 0.8).abs() < 1e-9); // sla_affinity unchanged
        assert!((ctx[3] - 1.0).abs() < 1e-9); // zero pressure -> 1.0
    }
}
