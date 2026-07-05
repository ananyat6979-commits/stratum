//! Per-worker cache-hit prediction index. Rust port of
//! services/cache-oracle/src/stratum_oracle/cache_hit_index.py.
//!
//! Lives in stratum-router (not cache-oracle) because cache_hit_prob
//! is a (request, worker) pair signal that must be computed locally
//! and synchronously at routing time, see embedding.rs's module
//! doc for the full architectural reasoning. This is NOT reachable
//! over HTTP; api.py's /signals endpoint permanently reports
//! cache_hit_prob: 0.0, cache_hit_prob_is_real: false on the wire,
//! and SemanticRouter overwrites that placeholder with this index's
//! real local computation. See ADR-009.
//!
//! # Why brute-force linear scan, not an approximate index
//! Same reasoning as ADR-008 (the Python reference's IndexFlatL2
//! choice): at max_entries=200 per worker, a linear scan over 64-dim
//! f32 vectors is sub-microsecond in Rust, there is no scale problem
//! to solve with an approximate/compressed index structure. Building
//! IVF-PQ-equivalent machinery here would be solving a problem that
//! does not exist at this data volume. Revisit only if profiling
//! against real request volume shows this scan is measurably load-bearing
//! in the routing hot path, which is not expected below ~10,000 entries
//! per worker (same trigger as ADR-008).
//!
//! # insert() is called after routing, never inside route()
//! SemanticRouter::route() only calls query(). insert() is called by
//! the caller (gateway or whatever dispatches the actual request)
//! after a routing decision is made and the request is sent, this
//! keeps route() itself free of any mutation, preserving its
//! determinism and keeping the write path explicit and separate from
//! the read (scoring) path.

use std::collections::VecDeque;

use crate::embedding::{cosine_similarity, embed, EMBEDDING_DIM};

pub const DEFAULT_MAX_ENTRIES: usize = 200;

/// Cosine similarity threshold above which a nearest-neighbor match is
/// considered a likely cache hit.
///
/// CALIBRATION NOTE: the Python reference implementation's near-duplicate
/// test ("What is 2+2?" vs "What is 2 + 2?") measures ~0.639 similarity
/// under this embedding scheme, barely above this threshold. This means
/// near-duplicate prompts differing by whitespace/punctuation have very
/// little margin above the hit/no-hit cliff below. This threshold was
/// NOT independently re-derived for the Rust port; it was carried over
/// from the Python reference's empirical calibration since both use the
/// same embedding algorithm (see embedding.rs's faithful-port comment).
/// If the embedding logic ever diverges between the two, this threshold
/// needs re-validation, not silent inheritance. See ADR-009.
pub const SIMILARITY_HIT_THRESHOLD: f32 = 0.6;

/// Per-worker index of recently-routed prompt embeddings.
pub struct CacheHitIndex {
    max_entries: usize,
    entries: VecDeque<[f32; EMBEDDING_DIM]>,
}

impl CacheHitIndex {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            entries: VecDeque::with_capacity(max_entries),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }

    /// Record that this prompt was routed to this worker.
    ///
    /// Evicts the oldest entry (FIFO) if at max_entries capacity.
    /// Degenerate (empty/near-empty) prompts that embed to a zero
    /// vector are silently not indexed, nothing meaningful to
    /// compare against later.
    pub fn insert(&mut self, prompt: &str) {
        let vec = embed(prompt);
        if vec.iter().all(|&x| x == 0.0) {
            return;
        }

        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(vec);
    }

    /// Estimate cache_hit_prob for a new prompt against this worker's
    /// recently-routed prompt history.
    ///
    /// Returns a value in [0.0, 1.0]. 0.0 if the index is empty, the
    /// query prompt is degenerate, or the best match's similarity is
    /// below SIMILARITY_HIT_THRESHOLD.
    pub fn query(&self, prompt: &str) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }

        let query_vec = embed(prompt);
        if query_vec.iter().all(|&x| x == 0.0) {
            return 0.0;
        }

        let best_similarity = self
            .entries
            .iter()
            .map(|entry| cosine_similarity(&query_vec, entry))
            .fold(f32::NEG_INFINITY, f32::max);

        if best_similarity < SIMILARITY_HIT_THRESHOLD {
            return 0.0;
        }

        best_similarity.clamp(0.0, 1.0)
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }
}

impl Default for CacheHitIndex {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_on_empty_index_returns_zero() {
        let idx = CacheHitIndex::with_defaults();
        assert_eq!(idx.query("any prompt"), 0.0);
    }

    #[test]
    fn size_starts_at_zero() {
        let idx = CacheHitIndex::with_defaults();
        assert_eq!(idx.size(), 0);
    }

    #[test]
    fn querying_inserted_prompt_returns_high_probability() {
        let mut idx = CacheHitIndex::with_defaults();
        idx.insert("What is the capital of France?");
        let prob = idx.query("What is the capital of France?");
        assert!(prob > 0.9, "exact match should score near 1.0, got {prob}");
    }

    #[test]
    fn querying_similar_prompt_returns_moderate_probability() {
        let mut idx = CacheHitIndex::with_defaults();
        idx.insert("Explain how neural networks work");
        let prob = idx.query("Explain how neural networks function");
        assert!(
            prob > 0.5,
            "similar prompt should score above threshold, got {prob}"
        );
    }

    #[test]
    fn querying_unrelated_prompt_returns_zero() {
        let mut idx = CacheHitIndex::with_defaults();
        idx.insert("What is the capital of France?");
        let prob = idx.query("Write a recursive Fibonacci function in Rust");
        assert_eq!(prob, 0.0, "unrelated prompt should score 0.0, got {prob}");
    }

    #[test]
    fn size_increments_on_insert() {
        let mut idx = CacheHitIndex::with_defaults();
        idx.insert("prompt one");
        idx.insert("prompt two");
        assert_eq!(idx.size(), 2);
    }

    #[test]
    fn multiple_entries_nearest_neighbor_is_correct() {
        let mut idx = CacheHitIndex::with_defaults();
        idx.insert("What is the capital of France?");
        idx.insert("Write a Python sorting function");
        idx.insert("Explain quantum entanglement");

        let prob = idx.query("What is the capital city of France?");
        assert!(prob > 0.5);
    }

    #[test]
    fn eviction_at_capacity_maintains_max_size() {
        let mut idx = CacheHitIndex::new(3);
        idx.insert("prompt one");
        idx.insert("prompt two");
        idx.insert("prompt three");
        idx.insert("prompt four"); // evicts "prompt one"

        assert_eq!(idx.size(), 3);
    }

    #[test]
    fn evicted_entry_no_longer_matches() {
        let mut idx = CacheHitIndex::new(2);
        idx.insert("a completely unique first prompt about astronomy");
        idx.insert("second prompt");
        idx.insert("third prompt"); // evicts the astronomy prompt

        let prob = idx.query("a completely unique first prompt about astronomy");
        assert!(
            prob < 0.9,
            "evicted entry should not still produce a near-exact match, got {prob}"
        );
    }

    #[test]
    fn empty_string_insert_does_not_index() {
        let mut idx = CacheHitIndex::with_defaults();
        idx.insert("");
        assert_eq!(idx.size(), 0, "empty prompt should not be indexed");
    }

    #[test]
    fn empty_string_query_returns_zero() {
        let mut idx = CacheHitIndex::with_defaults();
        idx.insert("some real prompt");
        assert_eq!(idx.query(""), 0.0);
    }

    #[test]
    fn default_max_entries_matches_module_constant() {
        let idx = CacheHitIndex::with_defaults();
        assert_eq!(idx.max_entries, DEFAULT_MAX_ENTRIES);
    }
}
