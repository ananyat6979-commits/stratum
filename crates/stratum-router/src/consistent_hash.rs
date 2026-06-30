//! Consistent hashing for request affinity routing.
//!
//! Request affinity means routing requests from the same session to the
//! same worker, so KV cache entries built during earlier turns are reused
//! for later turns. Without affinity, each request in a session lands on
//! a random worker and the KV cache is cold on every turn.
//!
//! # Algorithm: Jump Consistent Hash
//! Jump consistent hash (Lamping & Veach, 2014) maps a 64-bit key to
//! a bucket in [0, n) using O(ln n) time and O(1) space. It produces
//! minimal redistribution when the bucket count changes: only 1/n of
//! keys move when a bucket is added or removed.
//!
//! # Small Cluster Caveat
//! Jump consistent hash produces non-uniform distribution for n < 5.
//! For development clusters (n=2), the distribution is approximately
//! uniform in practice but not guaranteed. Documented in ADR-002 draft.
//! If precise uniformity is required for small clusters, switch to
//! virtual node consistent hash (more complex, not needed at Phase 3).
//!
//! Reference: Lamping, J. & Veach, E. (2014). "A Fast, Minimal Memory,
//! Consistent Hash Algorithm." arXiv:1406.2294.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute the jump consistent hash of a key for `num_buckets` buckets.
///
/// Returns a value in `[0, num_buckets)`.
///
/// # Panics
/// Panics if `num_buckets == 0`. Callers must ensure at least one bucket.
pub fn jump_hash(key: u64, num_buckets: usize) -> usize {
    assert!(num_buckets > 0, "num_buckets must be > 0");

    let mut k = key;
    let mut b: i64 = -1;
    let mut j: i64 = 0;

    while j < num_buckets as i64 {
        b = j;
        k = k.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(1);
        j = ((b + 1) as f64 * (((1i64 << 31) as f64) / (((k >> 33) + 1) as f64))) as i64;
    }

    b as usize
}

/// Hash an arbitrary key type to a u64 for use with `jump_hash`.
pub fn hash_key<K: Hash>(key: &K) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Select a bucket index for the given session_id using jump consistent hash.
///
/// If `session_id` is `None`, returns `None` — the caller should fall back
/// to round-robin or the scoring-based router for requests without sessions.
pub fn affinity_bucket(session_id: Option<&str>, num_buckets: usize) -> Option<usize> {
    let session_id = session_id?;
    if num_buckets == 0 {
        return None;
    }
    let key = hash_key(&session_id);
    Some(jump_hash(key, num_buckets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jump_hash_returns_valid_bucket() {
        for n in 1..=10 {
            for key in 0..100u64 {
                let bucket = jump_hash(key, n);
                assert!(bucket < n, "bucket {bucket} out of range for n={n}");
            }
        }
    }

    #[test]
    fn jump_hash_is_deterministic() {
        // Same key + same n must always produce the same bucket
        for key in 0..50u64 {
            let b1 = jump_hash(key, 4);
            let b2 = jump_hash(key, 4);
            assert_eq!(b1, b2);
        }
    }

    #[test]
    fn jump_hash_distributes_across_buckets() {
        // With 1000 keys and 4 buckets, each bucket should get roughly 250.
        // Allow generous tolerance (±15%) — exact uniformity is not guaranteed.
        let n = 4;
        let mut counts = vec![0usize; n];
        for key in 0..1000u64 {
            counts[jump_hash(key, n)] += 1;
        }
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count > 200 && count < 300,
                "bucket {i} got {count} keys (expected ~250)"
            );
        }
    }

    #[test]
    fn jump_hash_minimal_redistribution_on_growth() {
        // When n grows from 3 to 4, only ~25% of keys should change buckets.
        // This is the key property of consistent hashing.
        let moved = (0..1000u64)
            .filter(|&k| jump_hash(k, 3) != jump_hash(k, 4))
            .count();
        // Theoretical: 1/4 = 25% move. Allow generous bounds.
        assert!(moved < 350, "too many keys moved on growth: {moved}");
    }

    #[test]
    fn affinity_bucket_is_none_for_no_session() {
        assert_eq!(affinity_bucket(None, 4), None);
    }

    #[test]
    fn affinity_bucket_is_deterministic_for_same_session() {
        let b1 = affinity_bucket(Some("session-abc"), 4);
        let b2 = affinity_bucket(Some("session-abc"), 4);
        assert_eq!(b1, b2);
    }

    #[test]
    fn different_sessions_may_land_on_different_buckets() {
        let n = 4;
        let buckets: std::collections::HashSet<usize> = (0..20)
            .filter_map(|i| affinity_bucket(Some(&format!("session-{i}")), n))
            .collect();
        // With 20 different sessions and 4 buckets, all 4 should be hit
        assert_eq!(buckets.len(), n, "all buckets should be reachable");
    }
}
