# morie.fn — function file (hadesllm/morie)
"""Percentile modified rank test (Gibbons Ch 8.3.3).

Gastwirth's "percentile modified rank" test for location: only the
top-``q`` and bottom-``q`` quantiles of pooled ranks contribute; ranks
in the middle get score 0.  This down-weights the contribution of
observations near the centre, gaining power against heavy-tailed
alternatives.

Score function:
    a_i = max(R_i - (1-q)(N+1), 0) - max(q(N+1) - R_i, 0)

with q in (0, 0.5).  Statistic T = sum_{i in X} a_i.  Variance under
H0 is m*n/(N*(N-1)) * sum a_i^2.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["percentile_modified_rank"]


def percentile_modified_rank(x, y, q: float = 0.25):
    """Percentile-modified rank test (Gastwirth).

    Parameters
    ----------
    x, y : array-like
    q : float in (0, 0.5)
        Tail fraction to retain.  Default 0.25.

    Returns
    -------
    RichResult with payload:
        statistic, p_value, z, n, m, q, method
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m, n = int(x.size), int(y.size)
    N = m + n
    if m < 2 or n < 2:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "z": np.nan,
            "n": N, "m": m, "q": float(q),
            "method": "Percentile-modified rank test",
        })
    if not (0.0 < q < 0.5):
        raise ValueError("q must lie strictly between 0 and 0.5")
    pooled = np.concatenate([x, y])
    R = stats.rankdata(pooled)
    upper_cut = (1.0 - q) * (N + 1.0)
    lower_cut = q * (N + 1.0)
    a = np.maximum(R - upper_cut, 0.0) - np.maximum(lower_cut - R, 0.0)
    T = float(a[:m].sum())
    sum_a2 = float((a ** 2).sum())
    Var_T = (m * n / float(N * (N - 1))) * sum_a2
    z = T / np.sqrt(Var_T) if Var_T > 0 else np.nan
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return RichResult(payload={
        "statistic": T,
        "p_value": float(p),
        "z": float(z),
        "n": N,
        "m": m,
        "q": float(q),
        "method": "Percentile-modified rank (Gastwirth) test",
    })


def cheatsheet():
    return "pctmr: Percentile-modified rank (Gastwirth) test"


# CANONICAL TEST
# >>> percentile_modified_rank([1,2,3,4,5], [6,7,8,9,10], q=0.25)
# Clear location shift -> z strongly negative -> small p-value
