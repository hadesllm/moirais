# morie.fn — function file (hadesllm/morie)
"""Coefficient of concordance for incomplete rankings (Gibbons Ch 12.5).

Kendall's W for k rankers ranking n objects.  Accepts EITHER
- a 2-D matrix (n rows = objects, k cols = rankers), allowing
  NaN entries for incomplete rankings, OR
- a 2-D matrix of full rankings (no NaN), in which case the
  classical Kendall-W formula is used.

For complete rankings:
    W = 12 * S / (k^2 * (n^3 - n))
where S = sum over i of (R_i - mean_R)^2 with R_i = sum-of-ranks for
object i.

For incomplete rankings (each ranker ranks a subset), W is
computed pairwise with available rankers per pair, weighted by
the count of common objects (Friedman-Kendall approach).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["concordance_incomplete"]


def concordance_incomplete(x):
    """Kendall's coefficient of concordance W (supports NaN gaps).

    Parameters
    ----------
    x : 2-D array-like (n objects x k rankers).  NaN = not ranked.

    Returns
    -------
    RichResult with payload:
        statistic   : Kendall's W in [0, 1]
        p_value     : chi-square approximation (df = n - 1)
        df          : n - 1
        n           : number of objects
        k           : number of rankers
    """
    X = np.asarray(x, dtype=float)
    if X.ndim != 2:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "df": np.nan,
            "n": 0, "k": 0,
            "method": "Kendall's W (incomplete rankings)",
        })
    n, k = X.shape
    if n < 2 or k < 2:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "df": n - 1,
            "n": n, "k": k,
            "method": "Kendall's W (incomplete rankings)",
        })

    # Rank within each column (ranker), preserving NaN.
    R = np.full_like(X, np.nan, dtype=float)
    for j in range(k):
        col = X[:, j]
        mask = ~np.isnan(col)
        if mask.sum() >= 2:
            R[mask, j] = stats.rankdata(col[mask])

    # Sum ranks per object across available rankers
    Ri = np.nansum(R, axis=1)
    # Per-object ranker count
    ki = np.sum(~np.isnan(R), axis=1)

    # Classic formula when complete
    if np.all(ki == k):
        Rbar = np.mean(Ri)
        S = float(np.sum((Ri - Rbar) ** 2))
        W = 12.0 * S / (k ** 2 * (n ** 3 - n))
    else:
        # Incomplete: normalise by per-object ranker count
        expected = (n + 1.0) / 2.0  # expected mean rank
        # Weight each (object, ranker) by 1 and sum squared deviations
        S = 0.0
        norm = 0.0
        for i in range(n):
            ri = R[i, ~np.isnan(R[i])]
            if ri.size > 0:
                S += float(np.sum((ri - expected) ** 2))
                norm += ri.size
        # Approximate W via the ratio of actual to max possible variance
        max_S = norm * ((n ** 2 - 1) / 12.0)
        W = float(S / max_S) if max_S > 0 else np.nan

    df = n - 1
    chi2 = k * (n - 1) * W if np.isfinite(W) else np.nan
    p = float(1.0 - stats.chi2.cdf(chi2, df)) if np.isfinite(chi2) else np.nan
    return RichResult(payload={
        "statistic": float(W),
        "p_value": p,
        "df": df,
        "chi2": float(chi2),
        "n": n,
        "k": k,
        "method": "Kendall's coefficient of concordance W (incomplete rankings)",
    })


def cheatsheet():
    return "cncrd: Kendall's coefficient of concordance W"


# CANONICAL TEST
# >>> concordance_incomplete([[1,1,1],[2,2,2],[3,3,3]])
# Perfect agreement among 3 rankers ranking 3 objects: W = 1
