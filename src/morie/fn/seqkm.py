# morie.fn — function file (hadesllm/morie)
"""K-mer frequency counting. 'There is always a bigger fish.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kmer_frequency(
    sequence: str,
    k: int = 3,
) -> DescriptiveResult:
    """
    Count k-mer frequencies in a sequence.

    Extracts all overlapping subsequences of length *k* and returns
    their frequency distribution.

    :param sequence: Input sequence string.
    :param k: K-mer length. Default 3.
    :return: DescriptiveResult with frequency dict and entropy.
    :raises ValueError: If k <= 0 or k > sequence length.

    References
    ----------
    Compeau, P. E. C., Pevzner, P. A., & Tesler, G. (2011). How to
    apply de Bruijn graphs to genome assembly. *Nature Biotechnology*,
    29(11), 987-991.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")
    if k > len(sequence):
        raise ValueError(f"k ({k}) exceeds sequence length ({len(sequence)}).")

    freq: dict[str, int] = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        freq[kmer] = freq.get(kmer, 0) + 1

    total = sum(freq.values())
    probs = np.array(list(freq.values())) / total
    entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))

    sorted_freq = dict(sorted(freq.items(), key=lambda x: -x[1]))

    return DescriptiveResult(
        name="K-mer Frequency",
        value=float(entropy),
        extra={
            "frequencies": sorted_freq,
            "k": k,
            "n_unique": len(freq),
            "n_total": total,
            "entropy": entropy,
            "most_common": next(iter(sorted_freq)) if sorted_freq else "",
        },
    )


short = kmer_frequency


def cheatsheet() -> str:
    return "kmer_frequency({}) -> K-mer frequency counting. 'There is always a bigger fish.' -"
