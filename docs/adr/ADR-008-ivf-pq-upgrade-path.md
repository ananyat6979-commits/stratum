# ADR-008: IndexFlatL2 Now, IndexIVFPQ Later- Explicit Upgrade Trigger

**Status**: Accepted
**Date**: 2026-07-04
**Deciders**: Project owner

## Context

`CacheHitIndex` (services/cache-oracle/src/stratum_oracle/cache_hit_index.py)
needs a nearest-neighbor search structure over per-worker prompt
embeddings to estimate `cache_hit_prob`. The original Phase 3 design
(consistent with the blueprint's stated architecture) called for
FAISS IVF-PQ, matching the semantic router's ADR-002 mention of
"IVF-PQ chosen over HNSW for 5x memory reduction at 3% recall loss."

That tradeoff analysis is real and correct, for the data volume it
was evaluated against. It does not automatically transfer to
`CacheHitIndex`'s actual per-worker scale, which is materially
different from whatever corpus size the original IVF-PQ vs. HNSW
comparison assumed.

## Decision

Use `IndexFlatL2` (exact brute-force nearest-neighbor search) for
`CacheHitIndex`, not `IndexIVFPQ`, until per-worker index size
empirically justifies the switch.

## Rationale

IVF-PQ's value proposition, approximate search with compressed
memory footprint, pays off when the alternative (exact search) is
too slow or too memory-hungry. Neither condition holds yet:

- **Speed**: `IndexFlatL2` is O(n) per query. At `max_entries=200`
  per worker, this is sub-millisecond, not a bottleneck against any
  other latency in the routing path (network round-trip to
  cache-oracle alone dominates).
- **Memory**: 200 vectors × 64 dimensions × 4 bytes (float32) = ~51KB
  per worker. Trivial regardless of worker count at any realistic
  cluster size for this project.
- **Training requirement**: `IndexIVFPQ` requires training on
  representative data before it can index anything meaningfully.
  There is no representative corpus at this stage. Training it now
  would mean either synthetic filler data (invalidates any reported
  recall/latency numbers) or an undertrained index that provides
  none of IVF-PQ's actual benefits while adding real complexity.

Choosing `IndexIVFPQ` now would be premature optimization for a
constraint that doesn't exist, at the cost of a training-data
requirement that can't be honestly satisfied yet.

## Upgrade Trigger

Revisit this decision if/when either becomes true, measured in
production rather than assumed:

1. Per-worker index size grows to a scale where `IndexFlatL2`'s O(n)
   query cost becomes measurable relative to other routing-path
   latency (rough estimate: >10,000 entries per worker, though this
   should be confirmed against real benchmark numbers when the
   question becomes live, not assumed from this ADR alone)
2. Memory footprint across all worker indices becomes a real
   operational concern (unlikely before (1), given the size
   calculation above)

If either trigger fires: swap `CacheHitIndex`'s internal FAISS index
type from `IndexFlatL2` to `IndexIVFPQ`. The `insert`/`query` public
interface does not change, only the index construction and the
addition of a training step (accumulate enough vectors, call
`.train()`, then switch from insert-only to the trained index).

## Consequences

**Positive**: `CacheHitIndex` is correct and honestly-scoped today,
with no undertrained-index risk and no synthetic-data dishonesty in
any benchmark claims. The upgrade path is small and well-defined when
it's actually needed.

**Negative**: If per-worker index size unexpectedly grows past the
trigger threshold faster than anticipated, there's a real (if
currently sub-millisecond) query-cost migration to do. Given the
FIFO eviction cap at `max_entries=200`, this is not expected to
happen without a deliberate configuration change raising that cap
significantly first, which itself would be a natural trigger point
to revisit this ADR.

**Neutral**: This decision is specific to `CacheHitIndex`. It does
not contradict or revise ADR-002's IVF-PQ vs. HNSW analysis for
whatever corpus that ADR was evaluating. Those are different index
instances at potentially different scales, and each index choice
should be justified against its own actual data volume.