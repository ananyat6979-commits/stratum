"""
Lightweight hash-based text embedding for cache-hit prediction.

WHY NOT A REAL EMBEDDING MODEL
================================
Phase 3's goal is proving the IVF-PQ routing/caching architecture --
index construction, nprobe recall/latency tradeoff, per-worker prompt
similarity tracking -- not achieving state-of-the-art semantic
similarity. A hashing-based bag-of-words vector is fast (no model
inference), has zero download/dependency weight beyond numpy, and is
sufficient to prove the mechanism: similar prompts (by shared token
n-grams) produce similar vectors, which is what the IVF-PQ index
needs to demonstrate correct nearest-neighbor retrieval.

Upgrading to a real embedding model (sentence-transformers or similar)
later is a drop-in replacement for embed() alone -- the IVF-PQ index
construction and query logic do not change based on how vectors are
produced.

METHOD: hashed character n-gram bag-of-words
==============================================
1. Extract character trigrams from the lowercased prompt text
2. Hash each trigram to a bucket in [0, DIM)
3. Increment that bucket's count
4. L2-normalize the resulting vector

This is the "hashing trick" (Weinberger et al. 2009), commonly used
for memory-efficient bag-of-words without maintaining a vocabulary.
Character n-grams (vs word n-grams) are used so short prompts and
prompts with typos/variations still produce meaningfully similar
vectors, word-level hashing would treat "What is 2+2?" and
"what is 2 + 2?" as more different than character trigrams do.
"""

from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 64


def embed(text: str) -> np.ndarray:
    """
    Embed text into a fixed-dimension vector via hashed character trigrams.

    Args:
        text: The prompt text to embed.

    Returns:
        A float32 numpy array of shape (EMBEDDING_DIM,), L2-normalized.
        Returns a zero vector for empty input (callers should check
        np.linalg.norm(result) > 0 before using it for similarity
        search, since a zero vector has undefined cosine similarity).
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    text = text.lower().strip()

    if len(text) < 3:
        # Too short for trigrams -- hash the whole string as one token
        if text:
            bucket = hash(text) % EMBEDDING_DIM
            vec[bucket] += 1.0
        return vec

    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        bucket = hash(trigram) % EMBEDDING_DIM
        vec[bucket] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec