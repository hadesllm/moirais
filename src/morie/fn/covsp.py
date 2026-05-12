# morie.fn — function file (hadesllm/morie)
"""One-sample coverage probability (Gibbons Ch 2.11.1).

For an ordered sample X_(1) < ... < X_(n) the coverages
``U_i = F(X_(i)) - F(X_(i-1))`` are i.i.d. Beta(1, n) under the
null that F is the assumed continuous distribution.  This routine
computes the empirical placement coverages plus the cumulative
coverage ``F(X_(s)) - F(X_(r))`` for the extreme order statistics
``r=1, s=n``.

Returns each coverage's expected value E[U_i] = 1/(n+1).
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["one_sample_coverage"]


def one_sample_coverage(x):
    """One-sample coverage estimates.

    Parameters
    ----------
    x : array-like
        Univariate sample (assumed F is continuous; ties broken
        arbitrarily).

    Returns
    -------
    RichResult with payload:
        coverages       : (n+1,) array of empirical coverages U_i
                          based on ranks ``i/(n+1)``
        cumulative      : F(X_(n)) - F(X_(1)) on the rank scale
                          = (n-1)/(n+1)
        expected        : E[U_i] = 1/(n+1)
        n               : sample size
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        return RichResult(payload={
            "coverages": np.array([]),
            "cumulative": np.nan,
            "expected": np.nan,
            "n": n,
            "method": "One-sample coverage probability",
        })
    xs = np.sort(x)
    # Empirical placements on the (n+1)-interval grid: rank/(n+1)
    ranks = (np.arange(1, n + 1)) / (n + 1.0)
    # Successive coverages U_i for i = 1..n+1 (n+1 intervals)
    coverages = np.diff(np.concatenate([[0.0], ranks, [1.0]]))
    cumulative = float(ranks[-1] - ranks[0])  # = (n-1)/(n+1)
    expected = 1.0 / (n + 1.0)
    return RichResult(payload={
        "coverages": coverages,
        "cumulative": cumulative,
        "expected": float(expected),
        "n": n,
        "sample_min": float(xs[0]),
        "sample_max": float(xs[-1]),
        "method": "One-sample coverage probability",
    })


def cheatsheet():
    return "covsp: One-sample coverage probability"


# CANONICAL TEST
# >>> one_sample_coverage([1, 2, 3, 4, 5])
# n=5, cumulative=(5-1)/(5+1)=0.6667, expected=1/6=0.1667
