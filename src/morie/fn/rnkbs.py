# morie.fn — function file (hadesllm/morie)
"""Rank-based test for randomness (Gibbons Ch 3.5).

Mann's rank test for trend: count the number of pairs (i, j) with
i < j and X_i > X_j (an "inversion").  This is equivalent to a
Kendall-tau test of x against time index t = 1..n.

Standardised statistic Z under H0 (no trend):
    E[T] = n(n-1)/4,  Var[T] = n(n-1)(2n+5)/72.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["rank_based_test"]


def rank_based_test(x):
    """Mann's rank test for randomness (trend).

    Parameters
    ----------
    x : array-like
        Sequential observations.

    Returns
    -------
    RichResult with payload:
        statistic   : Kendall's tau of x vs. time index
        p_value     : two-sided p-value
        n           : sample size
        inversions  : number of inverted pairs
        z           : standardised normal approximation
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 3:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "n": n,
            "inversions": 0, "z": np.nan,
            "method": "Mann's rank test for randomness",
        })
    t = np.arange(1, n + 1, dtype=float)
    res = stats.kendalltau(t, x)
    # Count inversions directly for the inversions payload
    inv = 0
    for i in range(n - 1):
        inv += int(np.sum(x[i + 1:] < x[i]))
    return RichResult(payload={
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n": n,
        "inversions": int(inv),
        "z": float(res.statistic) * np.sqrt(9.0 * n * (n - 1) / (2.0 * (2 * n + 5))),
        "method": "Mann's rank test for randomness (Kendall tau vs time)",
    })


def cheatsheet():
    return "rnkbs: Mann's rank test for randomness (trend)"


# CANONICAL TEST
# >>> rank_based_test([1, 2, 3, 4, 5])
# Perfect upward trend: tau = 1, inversions = 0, p_value very small
