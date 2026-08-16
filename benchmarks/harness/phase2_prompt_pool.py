"""
Deliberately mixed prompt pool for Phase 2's routing-quality benchmark
(Option B from the design discussion: realistic, partially-repeated
traffic, not heavily-repeated best-case traffic).

WHY THIS SHAPE
==============
SemanticRouter's cache-hit index (see stratum-router's
semantic_router.rs, record_outcome/route) can only show a measurable
effect if the request stream actually contains real repetition for it
to learn from. A uniform-random, never-repeating stream gives it
nothing to act on, that would be an unfair test of a mechanism whose
entire premise is recognizing similar prompts. But heavily,
artificially repeated traffic (a handful of prompts, repeated
constantly) is unrealistically favorable, closer to a best-case demo
than an honest measurement: see the design discussion this file's
commit message links to for why Option A (favorable) was rejected in
favor of this.

This pool: 27 distinct prompts, varying meaningfully in length/topic
(short factual, medium reasoning, longer multi-step), each appearing
2-4 times across a full pass through the generated stream, spread out
rather than clustered, so most short windows of the stream look novel
to the cache-hit index and only a minority of requests are genuine
repeats, closer to a realistic mixed workload than either extreme.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

BASE_PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Name three primary colors.",
    "What year did World War 2 end?",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What is the largest planet in our solar system?",
    "What is the speed of light in a vacuum, approximately?",
    "Explain the difference between weather and climate in one sentence.",
    "Summarize why the sky appears blue.",
    "Explain what a prime number is, briefly.",
    "Describe the water cycle in two sentences.",
    "Explain the difference between a virus and bacteria briefly.",
    "What causes seasons to change on Earth?",
    "Briefly explain how a rainbow forms.",
    "What is the difference between a comet and an asteroid?",
    "Walk through the steps of photosynthesis briefly.",
    "Explain why ice floats on water, step by step.",
    "Describe, step by step, how a plant grows from a seed.",
    "Explain the basic idea behind supply and demand.",
    "Walk through the water treatment process at a high level.",
    "Explain how a refrigerator keeps food cold, step by step.",
    "Describe the process of how rain forms, step by step.",
    "Explain the basic steps of how a bill becomes a law.",
    "Walk through how a computer boots up, at a high level.",
]

MIN_REPETITIONS = 2
MAX_REPETITIONS = 4


@dataclass(frozen=True)
class StreamEntry:
    """One request in the generated stream: the prompt text and which
    repetition index it is (0-based) among that prompt's total
    occurrences, useful for later analysis of whether SemanticRouter's
    behavior differs on first-occurrence vs. repeat requests.
    """

    prompt: str
    repetition_index: int
    prompt_id: int


def generate_stream(seed: int) -> list[StreamEntry]:
    """Generates one full, shuffled pass through BASE_PROMPTS with each
    prompt repeated a random count in [MIN_REPETITIONS, MAX_REPETITIONS],
    spread across the stream via a full shuffle rather than clustered
    repeats, so consecutive requests are very rarely the same prompt by
    construction, that would trivially favor the cache-hit mechanism
    in a way real traffic wouldn't.

    Deterministic given the same seed, so a resumed run (see
    phase2_runner.py) can regenerate the exact same stream and continue
    from a known position rather than needing the stream itself
    persisted separately.
    """
    rng = random.Random(seed)
    entries: list[StreamEntry] = []
    for prompt_id, prompt in enumerate(BASE_PROMPTS):
        n_reps = rng.randint(MIN_REPETITIONS, MAX_REPETITIONS)
        for rep_idx in range(n_reps):
            entries.append(StreamEntry(prompt=prompt, repetition_index=rep_idx, prompt_id=prompt_id))
    rng.shuffle(entries)
    return entries


def infinite_stream(seed: int):
    """Yields StreamEntry values forever, cycling through fresh
    generate_stream() passes (each with a different derived seed, so
    repeated passes aren't byte-identical to each other) once one pass
    is exhausted. mSPRT's total observation count isn't known in
    advance (see the power check), so the runner needs a source that
    doesn't run out.
    """
    pass_index = 0
    while True:
        for entry in generate_stream(seed=seed + pass_index):
            yield entry
        pass_index += 1