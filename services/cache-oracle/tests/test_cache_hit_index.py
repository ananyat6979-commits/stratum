"""
Tests for CacheHitIndex.

Verifies:
1. Empty index returns 0.0 cache_hit_prob (no data, no false positives)
2. Querying with a previously-inserted prompt returns high cache_hit_prob
3. Querying with an unrelated prompt returns low/zero cache_hit_prob
4. FIFO eviction at capacity works correctly
5. Degenerate inputs (empty strings) are handled without crashing
"""

import pytest

from stratum_oracle.cache_hit_index import CacheHitIndex, DEFAULT_MAX_ENTRIES


class TestEmptyIndex:
    def test_query_on_empty_index_returns_zero(self):
        idx = CacheHitIndex()
        assert idx.query("any prompt") == 0.0

    def test_size_starts_at_zero(self):
        idx = CacheHitIndex()
        assert idx.size == 0


class TestInsertAndQuery:
    def test_querying_inserted_prompt_returns_high_probability(self):
        idx = CacheHitIndex()
        idx.insert("What is the capital of France?")
        prob = idx.query("What is the capital of France?")
        assert prob > 0.9, f"exact match should score near 1.0, got {prob}"

    def test_querying_similar_prompt_returns_moderate_probability(self):
        idx = CacheHitIndex()
        idx.insert("Explain how neural networks work")
        prob = idx.query("Explain how neural networks function")
        assert prob > 0.5, f"similar prompt should score above threshold, got {prob}"

    def test_querying_unrelated_prompt_returns_zero(self):
        idx = CacheHitIndex()
        idx.insert("What is the capital of France?")
        prob = idx.query("Write a recursive Fibonacci function in Rust")
        assert prob == 0.0, f"unrelated prompt should score 0.0, got {prob}"

    def test_size_increments_on_insert(self):
        idx = CacheHitIndex()
        idx.insert("prompt one")
        idx.insert("prompt two")
        assert idx.size == 2

    def test_multiple_entries_nearest_neighbor_is_correct(self):
        idx = CacheHitIndex()
        idx.insert("What is the capital of France?")
        idx.insert("Write a Python sorting function")
        idx.insert("Explain quantum entanglement")

        # Query close to the first entry, should still find it despite
        # two other unrelated entries also being in the index
        prob = idx.query("What is the capital city of France?")
        assert prob > 0.5


class TestEviction:
    def test_eviction_at_capacity_maintains_max_size(self):
        idx = CacheHitIndex(max_entries=3)
        idx.insert("prompt one")
        idx.insert("prompt two")
        idx.insert("prompt three")
        idx.insert("prompt four")  # should evict "prompt one"

        assert idx.size == 3

    def test_evicted_entry_no_longer_matches(self):
        idx = CacheHitIndex(max_entries=2)
        idx.insert("a completely unique first prompt about astronomy")
        idx.insert("second prompt")
        idx.insert("third prompt")  # evicts the astronomy prompt

        prob = idx.query("a completely unique first prompt about astronomy")
        # After eviction, this exact prompt should no longer be in the
        # index, similarity should drop since it's no longer indexed
        # (though it may still partially match "second"/"third" prompt
        # trigrams by chance, so we check it's not a near-1.0 match)
        assert prob < 0.9, "evicted entry should not still produce a near-exact match"


class TestDegenerateInputs:
    def test_empty_string_insert_does_not_crash(self):
        idx = CacheHitIndex()
        idx.insert("")
        assert idx.size == 0, "empty prompt should not be indexed"

    def test_empty_string_query_returns_zero(self):
        idx = CacheHitIndex()
        idx.insert("some real prompt")
        prob = idx.query("")
        assert prob == 0.0

    def test_default_max_entries_matches_module_constant(self):
        idx = CacheHitIndex()
        assert idx.max_entries == DEFAULT_MAX_ENTRIES