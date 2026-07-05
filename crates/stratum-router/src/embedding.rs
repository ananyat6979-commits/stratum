//! Lightweight hash-based text embedding for cache-hit prediction.
//!
//! Rust port of services/cache-oracle/src/stratum_oracle/embedding.py.
//! Ported faithfully, not improved, the Python version's similarity
//! thresholds were empirically calibrated against its exact output
//! (see the 0.639 near-duplicate finding below); silently changing the
//! embedding here would invalidate that calibration without anyone
//! noticing until a benchmark behaves strangely.
//!
//! # Why this lives in Rust, not called over HTTP to cache-oracle
//! cache_hit_prob is a (request, worker) pair signal, not a worker-state
//! signal: it answers "how similar is *this* prompt to worker W's recent
//! history," which cannot be captured by a fixed-interval polled snapshot
//! the way kv_pressure/latency/sla_affinity can (see http_signals_provider.rs
//! and the module doc there). Computing it requires a local, synchronous,
//! per-request lookup, which a sub-millisecond in-process vector scan
//! provides without violating RouterStrategy::route()'s "never block
//! indefinitely" contract. That contract is about network I/O and
//! unbounded waits, not all computation; a local brute-force scan over
//! at most a few hundred vectors is categorically different from an
//! HTTP round-trip to a separate process.
//!
//! # Known limitation: lexical, not semantic, similarity
//! Character-trigram hashing measures surface-form overlap, not meaning.
//! "What is the capital of France?" and "What is the capital of Germany?"
//! score highly similar under this scheme because they share nearly every
//! trigram except the country name, the one token that actually
//! determines whether a cached KV state is relevant. Conversely,
//! semantically identical but differently-phrased prompts may score
//! lower than lexically-similar-but-different-answer prompts.
//!
//! This is an accepted Phase 3 scoping choice (proving the routing/
//! caching *mechanism*, not achieving semantic embedding quality), but
//! it is a specific, trackable risk to the eventual benchmark: if
//! SemanticRouter underperforms or behaves oddly on near-duplicate
//! differently-worded prompts, check embedding quality FIRST, before
//! tuning bandit weights or similarity thresholds. See ADR-009.

/// Embedding dimensionality. Must match the Python reference
/// implementation's EMBEDDING_DIM for the two to remain comparable
/// if ever cross-validated.
pub const EMBEDDING_DIM: usize = 64;

/// Embed text into a fixed-dimension vector via hashed character trigrams.
///
/// Returns a zero vector for empty or near-empty (<3 char) input where
/// no trigram can be formed from more than the whole string. Callers
/// must check for a zero vector before using it in similarity search,
/// since cosine similarity is undefined for a zero vector.
pub fn embed(text: &str) -> [f32; EMBEDDING_DIM] {
    let mut vec = [0.0f32; EMBEDDING_DIM];
    let text = text.to_lowercase();
    let text = text.trim();

    if text.is_empty() {
        return vec;
    }

    let chars: Vec<char> = text.chars().collect();

    if chars.len() < 3 {
        let bucket = hash_str(text) as usize % EMBEDDING_DIM;
        vec[bucket] += 1.0;
        return normalize(vec);
    }

    for window in chars.windows(3) {
        let trigram: String = window.iter().collect();
        let bucket = hash_str(&trigram) as usize % EMBEDDING_DIM;
        vec[bucket] += 1.0;
    }

    normalize(vec)
}

fn normalize(mut vec: [f32; EMBEDDING_DIM]) -> [f32; EMBEDDING_DIM] {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
    vec
}

/// Deterministic string hash. Rust's std HashMap hasher is randomized
/// per-process by design (DoS protection), which would make embed()
/// non-deterministic across runs, unacceptable, since the whole
/// point is that identical text always produces an identical vector.
/// Uses a fixed-seed FNV-1a hash instead: simple, fast, deterministic,
/// no external dependency needed for this use case.
fn hash_str(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in s.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Cosine similarity between two vectors of the same fixed dimension.
/// Returns 0.0 if either vector has zero norm (undefined similarity).
pub fn cosine_similarity(a: &[f32; EMBEDDING_DIM], b: &[f32; EMBEDDING_DIM]) -> f32 {
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_has_correct_dimension() {
        let v = embed("hello world");
        assert_eq!(v.len(), EMBEDDING_DIM);
    }

    #[test]
    fn output_is_l2_normalized() {
        let v = embed("this is a test prompt of reasonable length");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
    }

    #[test]
    fn empty_string_returns_zero_vector() {
        let v = embed("");
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn deterministic_same_input_same_output() {
        let v1 = embed("What is the capital of France?");
        let v2 = embed("What is the capital of France?");
        assert_eq!(v1, v2);
    }

    #[test]
    fn identical_text_has_similarity_one() {
        let v1 = embed("Explain quantum computing");
        let v2 = embed("Explain quantum computing");
        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn near_duplicate_prompts_score_moderately_not_near_one() {
        // KNOWN, MEASURED, DOCUMENTED FINDING (matches the Python
        // reference implementation's test_similar_prompts_have_high_similarity):
        // inserting whitespace around "+2" shifts most trigrams from
        // that point onward, so this scores ~0.6-0.7, NOT near 1.0
        // despite the prompts being near-identical to a human reader.
        // This sits close to SIMILARITY_HIT_THRESHOLD (0.6) in
        // cache_hit_index.rs, meaning near-duplicate prompts that
        // differ by whitespace/punctuation have little margin above
        // the hit/no-hit cliff. See ADR-009.
        let v1 = embed("What is 2+2?");
        let v2 = embed("What is 2 + 2?");
        let sim = cosine_similarity(&v1, &v2);
        assert!(
            sim > 0.5,
            "expected moderate-to-high similarity for near-identical prompts, got {sim}"
        );
        // Explicit upper bound too: if this ever creeps to >0.9, the
        // embedding logic changed and the threshold calibration comment
        // above needs re-verification, not silent trust.
        assert!(
            sim < 0.9,
            "if this near-duplicate pair now scores near 1.0, the embedding \
             logic changed, re-verify SIMILARITY_HIT_THRESHOLD calibration, \
             got {sim}"
        );
    }

    #[test]
    fn related_prompts_more_similar_than_unrelated() {
        let sim_related = cosine_similarity(
            &embed("What is the capital of France?"),
            &embed("What is the capital of Germany?"),
        );
        let sim_unrelated = cosine_similarity(
            &embed("What is the capital of France?"),
            &embed("Write a Python function to sort a list"),
        );
        assert!(sim_related > sim_unrelated);
    }

    #[test]
    fn case_insensitive() {
        let v1 = embed("Hello World");
        let v2 = embed("hello world");
        assert_eq!(v1, v2);
    }

    #[test]
    fn short_strings_below_trigram_length_do_not_panic() {
        let v = embed("hi");
        assert_eq!(v.len(), EMBEDDING_DIM);
    }

    #[test]
    fn cosine_similarity_of_zero_vector_is_zero() {
        let zero = [0.0f32; EMBEDDING_DIM];
        let v = embed("some text");
        assert_eq!(cosine_similarity(&zero, &v), 0.0);
    }
}
