# ADR-009: cache_hit_prob Is Computed Locally in Rust, Never Over HTTP

**Status**: Accepted
**Date**: 2026-07-06
**Deciders**: Project owner

## Context

The original Phase 3 plan (per the module doc comments in `http_signals_provider.rs` and the earlier `api.py` design) treated `cache_hit_prob` as one of four worker-state signals served by cache-oracle's `GET /signals` endpoint and consumed via `HttpSignalsProvider`'s fixed-interval poll, identically to `kv_pressure`, predicted_latency_ms`, and `sla_affinity`.

This was a category error, caught during design review before any code wired `CacheHitIndex` into `api.py`. `kv_pressure` and the other two signals are genuinely worker-state: they describe a worker's current condition independent of any specific incoming request, so a periodic snapshot poll correctly represents them. `cache_hit_prob` is structurally different: it answers "how similar is *this specific incoming prompt* to worker W's recently-routed prompt history", a (request, worker) pair signal, not a worker-only fact. A fixed-interval poll has no way to know what the next request's prompt will be, so any `cache_hit_prob` value served by `/signals` would necessarily be either a stale average or a placeholder, not an honest answer to the question the field name implies.

## Decision

`cache_hit_prob` is computed locally and synchronously inside `stratum-router`, by a Rust port of the original Python prototype
(`crates/stratum-router/src/embedding.rs` and `cache_hit_index.rs`, ported from `services/cache-oracle/src/stratum_oracle/embedding.py` and `cache_hit_index.py`). It is never fetched over HTTP.

Consequences for each side of the system:

**cache-oracle (`api.py`)**: `GET /signals` permanently reports `cache_hit_prob: 0.0` and `cache_hit_prob_is_real: false` for every worker. This is not a temporary placeholder pending future wiring, it is the final, correct state. `CacheHitIndex`/`embedding.py` in cache-oracle remain in the codebase as a validated reference prototype (21 passing tests, including the empirical similarity-threshold calibration finding below), not as dead code, they proved the mechanism and surfaced a real calibration issue before the Rust port inherited it silently.

**stratum-router (`SemanticRouter`)**: gained a second data-fetch path alongside `signals_provider: Arc<P>`, a `cache_hit_indices: Arc<RwLock<HashMap<String, CacheHitIndex>>>` field, one `CacheHitIndex` per worker, populated by a new public method `record_routing_outcome(worker_id, prompt)`. `route()` calls `local_cache_hit_prob()` to query this structure and explicitly overwrites whatever `cache_hit_prob` value arrived from `signals_provider` (always 0.0 per the above) with the local result.

`record_routing_outcome()` MUST be called by the caller (gateway or whatever dispatches the actual request) *after* `route()` returns and the request is dispatched, never from inside `route()` itself. This keeps `route()` free of mutation, preserving `RouterStrategy::route()`'s documented determinism contract, and keeps the write path (recording what was routed where) explicit and separate from the read path (scoring a new request against existing history).

**`RouterStrategy` trait signature changed**: `route()` now takes `prompt: &str` in addition to `replay_key: &str` and `workers`.
`RoundRobinRouter` ignores it (`_prompt`). This was necessary rather than parsing prompt content out of `replay_key` via a third string convention (joining the existing `session:` and `sla:` prefix conventions), prompt text can contain any character, including the colons those conventions use as delimiters, so string-convention smuggling would have been genuinely broken, not just inelegant.

## Why "never block indefinitely" is not violated

`RouterStrategy::route()`'s contract says it must never block indefinitely and is called on the request hot path. A brute-force cosine scan over at most `DEFAULT_MAX_ENTRIES` (200) 64-dimensional f32 vectors, entirely in-process with no I/O, is sub-microsecond in Rust, categorically different from the HTTP round-trip to a separate Python process that made polling necessary for the other three signals in the first place. The contract's intent is "no network calls, no unbounded waits" on the hot path, not "zero computation." A future reader seeing a local vector-similarity call inside `route()` should not assume it is a contract violation and move it behind a poll, that would reintroduce the exact category error this ADR fixes.

## Known limitation: lexical, not semantic, similarity

The embedding (character-trigram hashing, the "hashing trick") measures surface-form overlap, not meaning. "What is the capital of France?" and "What is the capital of Germany?" score highly similar because they share nearly every trigram except the country name, the one token that actually determines whether cached state is relevant.
This is an accepted Phase 3 scoping choice (proving the routing/ caching mechanism, not achieving semantic embedding quality; see `embedding.rs`'s module doc for the full reasoning against pulling in `sentence-transformers`).

**Trigger for revisiting**: if the Phase 3 benchmark shows
`SemanticRouter` behaving oddly or underperforming specifically on
near-duplicate, differently-worded prompts, check embedding quality
first, before tuning bandit weights or `SIMILARITY_HIT_THRESHOLD`.

## Known limitation: threshold calibration has thin margin

`SIMILARITY_HIT_THRESHOLD = 0.6` was empirically measured against the
Python prototype's near-duplicate test case ("What is 2+2?" vs "What
is 2 + 2?"), which scores ~0.639, barely above the threshold. The
Rust port carries this same threshold since it uses the same embedding
algorithm (verified in `embedding.rs`'s
`near_duplicate_prompts_score_moderately_not_near_one` test, which
asserts both a lower bound of 0.5 and an upper bound of 0.9, so a
future embedding change that shifts this score is caught rather than
silently trusted). This means near-duplicate prompts differing only
by whitespace or punctuation have very little margin above the
hit/no-hit cliff (`query()` returns exactly 0.0 below threshold, not
a gradual falloff). Not fixed now; flagged for the same benchmark-time
revisit as the lexical-similarity limitation above.

## Consequences

**Positive**: `cache_hit_prob` is now an honest signal end-to-end, `api.py`'s wire format says plainly it is not real
(`cache_hit_prob_is_real: false`), and `SemanticRouter` never trusts
that placeholder, always overwriting it with a genuinely computed
local value. No dishonest float silently answering the wrong question.

**Negative**: The embedding/index logic now exists in two languages
(Python prototype, Rust production). Any future improvement to the
embedding algorithm must be made in both, or the two will diverge, worth a code comment cross-reference in both files (already present)
so this isn't discovered accidentally.

**Negative**: `record_routing_outcome()` is not yet called by any
production caller as of this ADR, `SemanticRouter`'s cache-hit history will remain permanently empty until the gateway/dispatch path
is updated to call it after a real request is sent. This is a known,
tracked gap, not an oversight: recording is deliberately decoupled
from `route()` itself, and wiring the actual call site is separate
follow-up work, not part of this architectural decision.

## Revisit Trigger

Before the Phase 3 benchmark (SemanticRouter vs RoundRobinRouter) is
used to make any claim involving cache-hit-aware routing quality:
confirm `record_routing_outcome()` is actually being called by a real
caller (otherwise every `cache_hit_prob` will be 0.0 throughout the
benchmark, silently reducing the four-signal design to three), and
revisit the two known limitations above if benchmark results look odd
specifically on near-duplicate prompt pairs.