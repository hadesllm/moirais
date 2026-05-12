# morie.fn — function file (hadesllm/morie)
"""Two-sample coverage probability (Gibbons Ch 2.11.2).

Given samples X (size m) and Y (size n), the placement coverages
of Y_(j) among the ordered X_(1) < ... < X_(m) sum to n (every Y
falls in one of the m+1 X-intervals).  This routine returns the
vector of block frequencies ``b_i`` of Y in each of the m+1
intervals.  Under H0 (X,Y from same continuous F) the expected
block frequency is ``n/(m+1)``.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["two_sample_coverage"]


def two_sample_coverage(x, y):
    """Two-sample placement coverage.

    Parameters
    ----------
    x, y : array-like
        Independent univariate samples.

    Returns
    -------
    RichResult with payload:
        block_freq      : (m+1,) vector, count of Y in each X-interval
        block_prop      : block_freq / n
        expected_prop   : 1/(m+1) under H0
        m, n            : sample sizes
        cumulative      : sum(block_freq) = n (sanity)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = int(x.size)
    n = int(y.size)
    if m < 1 or n < 1:
        return RichResult(payload={
            "block_freq": np.array([]),
            "block_prop": np.array([]),
            "expected_prop": np.nan,
            "m": m, "n": n,
            "cumulative": 0,
            "method": "Two-sample coverage probability",
        })
    xs = np.sort(x)
    # Edges: (-inf, X_(1)], (X_(1), X_(2)], ..., (X_(m), +inf)
    # Use np.searchsorted with side='right' to count Y in each interval.
    idx = np.searchsorted(xs, y, side="right")  # 0..m
    block_freq = np.bincount(idx, minlength=m + 1).astype(int)
    block_prop = block_freq / float(n) if n > 0 else np.zeros(m + 1)
    return RichResult(payload={
        "block_freq": block_freq,
        "block_prop": block_prop,
        "expected_prop": 1.0 / (m + 1.0),
        "m": m,
        "n": n,
        "cumulative": int(block_freq.sum()),
        "method": "Two-sample coverage probability",
    })


def cheatsheet():
    return "cov2s: Two-sample coverage probability (block frequencies)"


# CANONICAL TEST
# >>> two_sample_coverage([1,3,5], [2,4,6])
# block_freq = [0,1,1,1] (Y=2 between X=1..3, Y=4 between 3..5, Y=6 after 5)
