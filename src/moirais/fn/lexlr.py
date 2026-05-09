# moirais.fn — function file (hadesllm/moirais)
"""Lexicographic rank aggregation.

'Some people can read War and Peace and come away thinking it is a simple adventure story.' -- Lex Luthor"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lexico_rank(
    rankings: np.ndarray | list[list[int]],
) -> DescriptiveResult:
    """Lexicographic rank aggregation from multiple ranked lists.

    Given *m* rankers each ranking *n* items, computes an aggregate ranking
    using Borda count and lexicographic tie-breaking.

    Parameters
    ----------
    rankings : array (m, n)
        Each row is a ranking (1-based) of n items by one ranker.

    Returns
    -------
    DescriptiveResult
        ``value`` = aggregate ranking (0-indexed item order, best first).
    """
    R = np.asarray(rankings, dtype=int)
    if R.ndim != 2:
        raise ValueError("rankings must be a 2-D array (m rankers x n items)")
    m, n = R.shape
    if n < 2:
        raise ValueError("Need at least 2 items to rank")
    borda = np.zeros(n)
    for i in range(m):
        borda += n - R[i]
    sort_keys = [(-borda[j],) + tuple(R[:, j]) for j in range(n)]
    agg_order = sorted(range(n), key=lambda j: sort_keys[j])
    kendall_sum = 0.0
    count = 0
    for r in range(m):
        for i in range(n):
            for j in range(i + 1, n):
                ai, aj = agg_order.index(i), agg_order.index(j)
                ri, rj = R[r, i], R[r, j]
                if (ai < aj and ri > rj) or (ai > aj and ri < rj):
                    kendall_sum += 1
                count += 1
    avg_kendall = kendall_sum / (m * count) if count > 0 else 0.0
    return DescriptiveResult(
        name="Lexicographic rank aggregation",
        value=agg_order,
        extra={
            "n_items": n,
            "n_rankers": m,
            "borda_scores": borda.tolist(),
            "avg_kendall_tau_distance": round(avg_kendall, 4),
        },
    )


lexlr = lexico_rank


def cheatsheet() -> str:
    return "lexico_rank({}) -> Lexicographic rank aggregation. 'Some people can read War an"
