# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Borda count election method."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def borda_count(rankings) -> DescriptiveResult:
    """Borda count voting from a preference ranking matrix.

    Each voter ranks candidates 1..k (1 = most preferred). Borda score
    for each candidate = sum of (k - rank) across voters.

    :param rankings: (n_voters x n_candidates) matrix of rankings.
    :return: DescriptiveResult with winner index and scores.

    References
    ----------
    Armstrong (2014), Ch 1.

    .. epigraph:: 'Knowledge itself is power. — Francis Bacon'
    """
    R = np.asarray(rankings, dtype=float)
    if R.ndim != 2:
        raise ValueError("rankings must be 2D (voters x candidates).")
    n_voters, n_cand = R.shape
    scores = np.zeros(n_cand)
    for j in range(n_cand):
        scores[j] = (n_cand - R[:, j]).sum()
    winner = int(np.argmax(scores))
    return DescriptiveResult(
        name="borda_count",
        value=winner,
        extra={"scores": scores.tolist(), "n_voters": n_voters, "n_candidates": n_cand},
    )


borda = borda_count


def cheatsheet() -> str:
    return "borda_count({}) -> Borda count election method."
