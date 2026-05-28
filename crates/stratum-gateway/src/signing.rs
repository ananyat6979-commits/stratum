//! Request signing for deterministic replay.
//!
//! Every request that enters STRATUM is assigned a `replay_key` at ingress.
//! This key is the primary identifier in the replay event log and must satisfy:
//!
//! 1. **Determinism**: identical inputs → identical key, always, forever
//! 2. **Collision resistance**: different requests → different keys with overwhelming probability  
//! 3. **Efficiency**: signing adds <10μs overhead (enforced by Criterion benchmark)
//!
//! The key is SHA-256(ingress_timestamp_ns_le || body_sha256 || node_id_utf8),
//! hex-encoded to 64 characters.

use sha2::{Digest, Sha256};

/// Computes the replay_key for an incoming request.
///
/// # Arguments
/// * `ingress_timestamp_ns` - Wall clock nanoseconds at ingress. Used for causal
///   ordering in replay. Must be the same value stored in `ReplayEvent.lamport_ts`.
/// * `body_sha256` - SHA-256 of the raw request body bytes. Callers are responsible
///   for computing this before calling. Separating body hashing from key computation
///   allows the body hash to be reused for content-addressed caching.
/// * `ingress_node_id` - Unique identifier for the ingress node. Prevents key
///   collisions when two nodes receive requests at identical nanosecond timestamps.
///
/// # Determinism guarantee
/// This function is pure. It has no side effects and no external dependencies.
/// Given identical inputs it always produces identical output.
/// This invariant is property-tested in `tests/signing_determinism.rs`.
///
/// # Cancellation safety
/// Synchronous. Cannot be cancelled.
///
/// # Example
/// ```
/// use stratum_gateway::signing::compute_replay_key;
///
/// use sha2::Digest;
/// let body_hash = sha2::Sha256::digest(b"hello world").into();
/// let key = compute_replay_key(1_700_000_000_000_000_000i64, &body_hash, "node-0");
/// assert_eq!(key.len(), 64); // hex-encoded SHA-256
/// ```
pub fn compute_replay_key(
    ingress_timestamp_ns: i64,
    body_sha256: &[u8; 32],
    ingress_node_id: &str,
) -> String {
    let mut hasher = Sha256::new();
    // Little-endian bytes for timestamp — consistent across architectures
    hasher.update(ingress_timestamp_ns.to_le_bytes());
    hasher.update(body_sha256);
    hasher.update(ingress_node_id.as_bytes());
    hex::encode(hasher.finalize())
}

/// Computes the SHA-256 hash of a request body.
///
/// Separated from `compute_replay_key` so the body hash can be reused
/// for content-addressed caching without re-hashing.
pub fn hash_body(body: &[u8]) -> [u8; 32] {
    Sha256::digest(body).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_key_is_64_hex_chars() {
        let body_hash = hash_body(b"test request body");
        let key = compute_replay_key(0, &body_hash, "node-0");
        assert_eq!(key.len(), 64);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn identical_inputs_produce_identical_key() {
        let body_hash = hash_body(b"determinism test");
        let key1 = compute_replay_key(1_700_000_000_000i64, &body_hash, "node-0");
        let key2 = compute_replay_key(1_700_000_000_000i64, &body_hash, "node-0");
        assert_eq!(key1, key2, "signing must be deterministic");
    }

    #[test]
    fn different_timestamps_produce_different_keys() {
        let body_hash = hash_body(b"collision test");
        let key1 = compute_replay_key(1_000, &body_hash, "node-0");
        let key2 = compute_replay_key(1_001, &body_hash, "node-0");
        assert_ne!(key1, key2, "different timestamps must produce different keys");
    }

    #[test]
    fn different_nodes_produce_different_keys() {
        let body_hash = hash_body(b"node collision test");
        let key1 = compute_replay_key(0, &body_hash, "node-0");
        let key2 = compute_replay_key(0, &body_hash, "node-1");
        assert_ne!(key1, key2, "different node IDs must produce different keys");
    }

    #[test]
    fn different_bodies_produce_different_keys() {
        let hash1 = hash_body(b"request A");
        let hash2 = hash_body(b"request B");
        let key1 = compute_replay_key(0, &hash1, "node-0");
        let key2 = compute_replay_key(0, &hash2, "node-0");
        assert_ne!(key1, key2, "different bodies must produce different keys");
    }
}