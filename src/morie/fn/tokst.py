"""Token frequency and entropy statistics."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ._containers import DescriptiveResult


def token_stats(
    token_ids: list[int] | np.ndarray,
    vocab_size: int = 32000,
) -> DescriptiveResult:
    """Compute token frequency distribution and entropy statistics.

    :param token_ids: Sequence of token IDs.
    :param vocab_size: Vocabulary size for coverage calculation.
    :return: DescriptiveResult with entropy and coverage stats.
    """
    token_ids = np.asarray(token_ids)
    n = len(token_ids)
    if n == 0:
        raise ValueError("token_ids must not be empty")

    counts = Counter(token_ids.tolist())
    probs = np.array(list(counts.values()), dtype=np.float64) / n
    entropy = -float(np.sum(probs * np.log2(probs + 1e-30)))

    unique = len(counts)
    coverage = unique / vocab_size if vocab_size > 0 else 0.0

    sorted_counts = sorted(counts.values(), reverse=True)
    top_10_mass = sum(sorted_counts[:10]) / n if len(sorted_counts) >= 10 else 1.0

    return DescriptiveResult(
        name="token_stats",
        value=entropy,
        extra={
            "entropy_bits": entropy,
            "unique_tokens": unique,
            "total_tokens": n,
            "coverage": coverage,
            "top_10_mass": top_10_mass,
            "most_common": counts.most_common(10),
        },
    )


def cheatsheet() -> str:
    return "token_stats(token_ids, vocab_size) -> token frequency/entropy"


tokst = token_stats
