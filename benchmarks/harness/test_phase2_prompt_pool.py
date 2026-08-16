"""
Tests for phase2_prompt_pool.py. Fast, no network, no live services,
this is pure generation logic that a multi-day real-inference run will
depend on, worth checking directly before trusting it inside something
that expensive to re-run if wrong.
"""

from __future__ import annotations

import pytest

from phase2_prompt_pool import (
    BASE_PROMPTS,
    MAX_REPETITIONS,
    MIN_REPETITIONS,
    generate_stream,
    infinite_stream,
)


class TestGenerateStream:
    def test_every_base_prompt_appears_at_least_once(self):
        stream = generate_stream(seed=1)
        prompts_seen = {entry.prompt for entry in stream}
        assert prompts_seen == set(BASE_PROMPTS)

    def test_every_prompt_repetition_count_within_bounds(self):
        stream = generate_stream(seed=1)
        counts: dict[str, int] = {}
        for entry in stream:
            counts[entry.prompt] = counts.get(entry.prompt, 0) + 1
        for prompt, count in counts.items():
            assert MIN_REPETITIONS <= count <= MAX_REPETITIONS, f"{prompt!r}: {count}"

    def test_repetition_index_is_correct_per_prompt(self):
        stream = generate_stream(seed=1)
        seen_indices: dict[str, list[int]] = {}
        for entry in stream:
            seen_indices.setdefault(entry.prompt, []).append(entry.repetition_index)
        for prompt, indices in seen_indices.items():
            assert sorted(indices) == list(range(len(indices))), f"{prompt!r}: {sorted(indices)}"

    def test_same_seed_produces_identical_stream(self):
        stream_a = generate_stream(seed=42)
        stream_b = generate_stream(seed=42)
        assert stream_a == stream_b

    def test_different_seeds_produce_different_orderings(self):
        stream_a = generate_stream(seed=1)
        stream_b = generate_stream(seed=2)
        prompts_a = [e.prompt for e in stream_a]
        prompts_b = [e.prompt for e in stream_b]
        assert prompts_a != prompts_b

    def test_consecutive_identical_prompts_are_rare_across_many_seeds(self):
        # A single seed's exact count is a noisy, low-sample statistic:
        # random.shuffle on a real multiset (27 prompts, each appearing
        # 2-4 times, ~27-108 total entries) will sometimes place two
        # occurrences of the same prompt adjacently just by chance, an
        # earlier version of this test asserted a hard ceiling on one
        # specific seed's count without checking what a random shuffle
        # actually produces on average, and failed on the very next
        # seed tried (seed=1 produced 4, the arbitrary ceiling was 3).
        # This checks the design intent (spread, not clustered) against
        # the AVERAGE across many independent seeds instead, which is
        # the property that actually matters for the real benchmark:
        # not that any single pass is duplicate-free, but that
        # clustering isn't a systematic property of generate_stream().
        n_seeds = 200
        duplicate_counts = []
        total_entries = None
        for seed in range(n_seeds):
            stream = generate_stream(seed=seed)
            total_entries = len(stream)
            duplicates = sum(
                1 for i in range(1, len(stream)) if stream[i].prompt == stream[i - 1].prompt
            )
            duplicate_counts.append(duplicates)

        mean_duplicates = sum(duplicate_counts) / n_seeds
        # A full pass has ~27-108 entries; even a handful of adjacent
        # duplicates per pass would mean roughly 5-15% of all adjacent
        # pairs are same-prompt, well above what a real shuffle of this
        # multiset should produce if spreading is working as intended.
        # 3.0 as a mean ceiling is generous relative to what an
        # unbiased shuffle of ~77-100 entries across 27 distinct values
        # should produce, while still catching a real regression (e.g.
        # generate_stream silently reverting to clustering repeats
        # instead of shuffling the full entry list).
        assert mean_duplicates <= 3.0, (
            f"mean {mean_duplicates:.2f} consecutive duplicate prompts per "
            f"pass across {n_seeds} seeds ({total_entries}-entry passes), "
            f"expected shuffling to keep this low on average"
        )


class TestInfiniteStream:
    def test_yields_more_entries_than_one_generate_stream_pass(self):
        gen = infinite_stream(seed=1)
        one_pass_length = len(generate_stream(seed=1))
        collected = [next(gen) for _ in range(one_pass_length + 10)]
        assert len(collected) == one_pass_length + 10

    def test_second_pass_is_not_identical_to_first_pass(self):
        gen = infinite_stream(seed=1)
        one_pass_length = len(generate_stream(seed=1))
        first_pass = [next(gen) for _ in range(one_pass_length)]
        second_pass = [next(gen) for _ in range(one_pass_length)]
        assert first_pass != second_pass