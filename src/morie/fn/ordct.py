# morie.fn — function file (hadesllm/morie)
"""Linear-by-linear association test for ordered categories
(Gibbons Ch 14.6.1).

For an r x c contingency table with ordered row scores u_i and
ordered column scores v_j (defaults: 1..r and 1..c), the
correlation-style statistic is

    M^2 = (n - 1) * r^2

where r is the Pearson correlation between the row- and column-
score pairs implied by the cell counts.  Under H0 of
independence, M^2 ~ chi^2_1.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["ordered_categories"]


def ordered_categories(x, row_scores=None, col_scores=None):
    """Linear-by-linear association test for ordered categorical data.

    Parameters
    ----------
    x : 2-D array-like
        r x c contingency table (rows ordered, cols ordered).
    row_scores : array-like or None
        Length-r row scores u_i.  Default: 1..r.
    col_scores : array-like or None
        Length-c col scores v_j.  Default: 1..c.

    Returns
    -------
    RichResult with payload:
        statistic, p_value, df=1, n, r, correlation, method
    """
    X = np.asarray(x, dtype=float)
    if X.ndim != 2 or X.size == 0:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "df": 1,
            "n": 0, "correlation": np.nan,
            "method": "Linear-by-linear association",
        })
    r, c = X.shape
    n_total = float(X.sum())
    if r < 2 or c < 2 or n_total < 2:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "df": 1,
            "n": int(n_total), "correlation": np.nan,
            "method": "Linear-by-linear association",
        })
    u = np.arange(1, r + 1, dtype=float) if row_scores is None else np.asarray(row_scores, dtype=float)
    v = np.arange(1, c + 1, dtype=float) if col_scores is None else np.asarray(col_scores, dtype=float)
    # Reconstruct sample of (u_i, v_j) pairs weighted by counts
    U = np.repeat(np.repeat(u[:, None], c, axis=1).ravel(),
                  X.ravel().astype(int))
    V = np.repeat(np.tile(v, r),
                  X.ravel().astype(int))
    if U.size < 2 or np.std(U) == 0 or np.std(V) == 0:
        return RichResult(payload={
            "statistic": 0.0, "p_value": 1.0, "df": 1,
            "n": int(n_total), "correlation": 0.0,
            "method": "Linear-by-linear association",
        })
    rho = float(np.corrcoef(U, V)[0, 1])
    M2 = (n_total - 1.0) * rho ** 2
    p = float(1.0 - stats.chi2.cdf(M2, 1))
    return RichResult(payload={
        "statistic": float(M2),
        "p_value": p,
        "df": 1,
        "n": int(n_total),
        "correlation": rho,
        "method": "Linear-by-linear (Mantel-Haenszel) trend test",
    })


def cheatsheet():
    return "ordct: Linear-by-linear association for ordered categories"


# CANONICAL TEST
# >>> ordered_categories([[10,2,1],[3,8,2],[1,2,10]])
# Diagonal-heavy: strong positive correlation, large M^2, small p
