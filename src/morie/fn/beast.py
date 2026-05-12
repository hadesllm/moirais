# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Mutation rate estimation (Jukes-Cantor). 'Oh my stars and garters!' -- Beast"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def jukes_cantor_rate(
    sequences: list[str],
    *,
    time: float = 1.0,
) -> DescriptiveResult:
    r"""Estimate pairwise evolutionary distances using the Jukes-Cantor model.

    For each pair of aligned nucleotide sequences, computes the proportion
    of differing sites :math:`p` and the JC69 corrected distance
    :math:`d = -\\frac{3}{4} \\ln(1 - \\frac{4}{3} p)`.

    Parameters
    ----------
    sequences : list of str
        Two or more aligned nucleotide sequences (same length).
    time : float
        Divergence time for rate calculation (distance / time).

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``distance_matrix`` (n x n), ``rate_matrix``,
        ``mean_distance``, ``n_pairs``.
    """
    if len(sequences) < 2:
        raise ValueError("Need at least 2 sequences")
    L = len(sequences[0])
    for s in sequences:
        if len(s) != L:
            raise ValueError("All sequences must have equal length")
    if L == 0:
        raise ValueError("Sequences must be non-empty")
    if time <= 0:
        raise ValueError("time must be positive")

    n = len(sequences)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diffs = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
            p = diffs / L
            if p >= 0.75:
                d = np.inf
            else:
                d = -0.75 * np.log(1 - (4.0 / 3.0) * p)
            D[i, j] = d
            D[j, i] = d

    R = D / time
    finite_mask = np.isfinite(D) & (D > 0)
    mean_d = float(D[finite_mask].mean()) if finite_mask.any() else 0.0
    n_pairs = n * (n - 1) // 2

    return DescriptiveResult(
        name="jukes_cantor_rate",
        value={
            "distance_matrix": D,
            "rate_matrix": R,
            "mean_distance": mean_d,
            "n_pairs": n_pairs,
        },
        extra={"n_sequences": n, "seq_length": L, "time": time},
    )


beast = jukes_cantor_rate


def cheatsheet() -> str:
    return "jukes_cantor_rate({}) -> Mutation rate estimation (Jukes-Cantor). 'Oh my stars and ga"
