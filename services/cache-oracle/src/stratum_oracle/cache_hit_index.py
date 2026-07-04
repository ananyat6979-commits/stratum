"""
Per-worker cache-hit prediction index using FAISS.

WHY IndexFlatL2, NOT IndexIVFPQ (yet)
=======================================
IVF-PQ (inverted file + product quantization) is designed to make
approximate nearest-neighbor search fast and memory-efficient at
scale, typically justified starting in the range of ~10^5-10^6+
vectors. It requires a training step on representative data before
it can index anything (FAISS's own guidance: at least `nlist` training
points, ideally substantially more, to build a meaningful coarse
quantizer and PQ codebooks).

At Phase 3's actual expected scale, one index per worker, each
holding recently-routed prompts, realistically dozens to low hundreds
of entries per worker. IVF-PQ's approximation and compression
tradeoffs solve a problem this deployment does not have. Constructing
an IndexIVFPQ now would mean either training it on synthetic filler
data (dishonest, any recall/latency numbers reported would not
reflect real usage) or leaving it functionally undertrained, which
defeats the point of using it.

IndexFlatL2 is exact brute-force search: O(n) per query, no training
required, correct by construction. At n in the hundreds, this is
both fast enough (sub-millisecond) and exactly correct, there is no
benefit to trading exactness for approximation at this scale.

UPGRADE PATH (see ADR-008)
===========================
If/when per-worker index size grows into the range where IVF-PQ's
tradeoffs become justified (measured, not assumed, track actual
index size in production and revisit if it crosses ~10,000+ entries
per worker), swap CacheHitIndex's internal FAISS index type. The
insert/query interface below does not change; only the index
construction does.

CACHE ENTRY EVICTION
======================
Each worker's index has a maximum size (default 200 entries). When
full, the oldest entry is evicted (FIFO), this approximates "recently
routed prompts are more likely to reflect current KV cache state than
prompts routed long ago," without requiring real KV cache eviction
telemetry (which Phase 3 does not have, see ADR-007's related gap
for KV pressure).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import faiss
import numpy as np

from .embedding import embed, EMBEDDING_DIM

DEFAULT_MAX_ENTRIES = 200

# Cosine similarity threshold above which a nearest-neighbor match is
# considered a likely cache hit. Below this, cache_hit_prob is reported
# as low regardless of nearest-neighbor distance, an unrelated prompt
# should not be treated as a cache hit just because it's the "closest"
# of a bad set of candidates.
SIMILARITY_HIT_THRESHOLD = 0.6


@dataclass
class CacheHitIndex:
    """
    Per-worker index of recently-routed prompt embeddings.

    One instance per worker. The router (via cache-oracle's API) queries
    this to estimate cache_hit_prob for a new request: if the new
    prompt is similar to a recently-routed prompt on this worker, the
    worker's KV cache likely still holds relevant prefix state.

    Uses IndexFlatL2 (exact brute-force search), see module docstring
    for why this is the correct choice at Phase 3's scale, not IVF-PQ.
    """

    max_entries: int = DEFAULT_MAX_ENTRIES
    _index: faiss.IndexFlatL2 = field(init=False)
    _entry_order: deque = field(init=False)  # tracks insertion order for FIFO eviction
    _next_id: int = field(init=False, default=0)

    def __post_init__(self):
        self._index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self._entry_order = deque()

    def insert(self, prompt: str) -> None:
        """
        Record that this prompt was routed to this worker.

        Call this after a routing decision is made and the request is
        sent to the worker, this is what populates the "recently
        routed prompts" the next request's cache-hit prediction checks
        against.

        Evicts the oldest entry (FIFO) if at max_entries capacity.
        """
        vec = embed(prompt)
        if np.linalg.norm(vec) == 0:
            # Empty or degenerate prompt, nothing meaningful to index
            return

        if self._index.ntotal >= self.max_entries:
            self._evict_oldest()

        self._index.add(vec.reshape(1, -1))
        self._entry_order.append(self._next_id)
        self._next_id += 1

    def query(self, prompt: str) -> float:
        """
        Estimate cache_hit_prob for a new prompt against this worker's
        recently-routed prompt history.

        Returns:
            A value in [0.0, 1.0]. 0.0 if the index is empty, the query
            prompt is degenerate, or the nearest neighbor's similarity
            is below SIMILARITY_HIT_THRESHOLD. Otherwise, the cosine
            similarity to the nearest neighbor (higher = more likely
            this worker's KV cache holds relevant state).
        """
        if self._index.ntotal == 0:
            return 0.0

        query_vec = embed(prompt)
        if np.linalg.norm(query_vec) == 0:
            return 0.0

        # k=1: only the single nearest neighbor matters for cache-hit
        # prediction we care whether ANY recent prompt on this worker
        # is similar enough to suggest warm KV cache state, not an
        # aggregate over multiple neighbors.
        distances, _ = self._index.search(query_vec.reshape(1, -1), k=1)
        l2_distance = float(distances[0][0])

        # Convert L2 distance (on L2-normalized vectors) to cosine similarity.
        # For unit vectors: ||a-b||^2 = 2 - 2*cos(a,b), so cos(a,b) = 1 - d^2/2
        similarity = 1.0 - (l2_distance / 2.0)

        if similarity < SIMILARITY_HIT_THRESHOLD:
            return 0.0

        return max(0.0, min(1.0, similarity))

    def _evict_oldest(self) -> None:
        """
        FAISS's IndexFlatL2 does not support removing arbitrary vectors
        by insertion order directly in a way that's efficient for this
        use case, so eviction is implemented by rebuilding the index
        without the oldest entry. This is O(n) per eviction, acceptable
        at max_entries=200 scale (sub-millisecond), and avoids adding
        FAISS's more complex ID-based removal API for a problem this
        small.
        """
        if not self._entry_order:
            return

        self._entry_order.popleft()
        # Rebuild: FAISS doesn't expose "remove first N vectors" directly,
        # but reconstruct_n lets us pull all vectors, drop the first,
        # and rebuild. At n<=200 this is fast enough to not matter.
        all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
        remaining = all_vectors[1:]  # drop the oldest (index 0)

        self._index = faiss.IndexFlatL2(EMBEDDING_DIM)
        if len(remaining) > 0:
            self._index.add(remaining)

    @property
    def size(self) -> int:
        return self._index.ntotal