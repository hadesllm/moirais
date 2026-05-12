# morie.fn — function file (hadesllm/morie)
"""Pearson contingency coefficient C (Gibbons Ch 14.2.1).

C = sqrt(chi2 / (chi2 + n))

Also reports Cramer's V = sqrt(chi2 / (n * (min(r, c) - 1))) and
the maximum attainable C = sqrt((min(r,c)-1)/min(r,c)) so users
can normalise.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["contingency_coefficient"]


def contingency_coefficient(x):
    """Pearson contingency coefficient and friends.

    Parameters
    ----------
    x : 2-D array-like
        Two-way contingency table of counts (rows x cols).

    Returns
    -------
    RichResult with payload:
        statistic   : Pearson's C
        cramers_v   : Cramer's V (chi-square based)
        chi2        : Pearson chi-square statistic
        p_value     : chi-square p-value (df = (r-1)(c-1))
        df          : degrees of freedom
        max_C       : theoretical upper bound for C given table size
        n           : grand total
    """
    X = np.asarray(x, dtype=float)
    if X.ndim != 2 or X.size == 0:
        return RichResult(payload={
            "statistic": np.nan, "cramers_v": np.nan,
            "chi2": np.nan, "p_value": np.nan, "df": np.nan,
            "max_C": np.nan, "n": 0,
            "method": "Pearson contingency coefficient",
        })
    chi2, p, df, _ = stats.chi2_contingency(X, correction=False)
    n_total = float(X.sum())
    C = np.sqrt(chi2 / (chi2 + n_total)) if n_total > 0 else np.nan
    r, c = X.shape
    mn = min(r, c)
    V = np.sqrt(chi2 / (n_total * (mn - 1))) if (n_total > 0 and mn > 1) else np.nan
    max_C = np.sqrt((mn - 1) / mn) if mn > 1 else np.nan
    return RichResult(payload={
        "statistic": float(C),
        "cramers_v": float(V),
        "chi2": float(chi2),
        "p_value": float(p),
        "df": int(df),
        "max_C": float(max_C),
        "n": int(n_total),
        "method": "Pearson contingency coefficient",
    })


def cheatsheet():
    return "cntgc: Pearson contingency coefficient C"


# CANONICAL TEST
# >>> contingency_coefficient([[10, 0], [0, 10]])
# Perfect 2x2 association: chi2 = 20, C = sqrt(20/30) = 0.8165
