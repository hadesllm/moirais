"""Sukhatme test for scale (Gibbons Ch 9.7).

Two-sample scale test assuming equal medians.  Counts for each
X_i the number of Y_j with |Y_j| > |X_i|, normalised.  Equivalent
to a Mann-Whitney U on absolute values, after centring both
samples at the pooled median.

Returns the standardised statistic and its normal-approx p-value.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["sukhatme_test"]


def sukhatme_test(x, y):
    """Sukhatme two-sample scale test.

    Parameters
    ----------
    x, y : array-like
        Independent samples (assumed equal medians; the function
        centres at the pooled median if medians differ).

    Returns
    -------
    RichResult with payload:
        statistic, p_value, U, n, m, method
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m, n = int(x.size), int(y.size)
    if m < 2 or n < 2:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "U": np.nan,
            "n": m + n, "m": m,
            "method": "Sukhatme scale test",
        })
    pooled_med = float(np.median(np.concatenate([x, y])))
    ax = np.abs(x - pooled_med)
    ay = np.abs(y - pooled_med)
    res = stats.mannwhitneyu(ax, ay, alternative="two-sided")
    # Standardised Z
    N = m + n
    E_U = m * n / 2.0
    Var_U = m * n * (N + 1.0) / 12.0
    z = (res.statistic - E_U) / np.sqrt(Var_U) if Var_U > 0 else np.nan
    return RichResult(payload={
        "statistic": float(z),
        "p_value": float(res.pvalue),
        "U": float(res.statistic),
        "n": N,
        "m": m,
        "method": "Sukhatme scale test (Mann-Whitney on |·-median|)",
    })


def cheatsheet():
    return "sukht: Sukhatme two-sample scale test"


# CANONICAL TEST
# >>> sukhatme_test([0,0,0,0,0], [-10,-5,5,10,0])
# X has zero spread, Y has large spread -> z strongly negative, small p
