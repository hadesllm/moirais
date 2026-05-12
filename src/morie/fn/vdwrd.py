"""Van der Waerden two-sample normal-scores test (Gibbons Ch 8.3.2).

Replaces ranks with ``Phi^{-1}(R_i / (N + 1))`` and sums over the
first sample.  Asymptotically equivalent to Terry-Hoeffding; the
two differ only in the boundary correction used to invert the
ranks back to the normal scale.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["van_der_waerden_test"]


def van_der_waerden_test(x, y):
    """Van der Waerden two-sample normal-scores location test.

    Parameters
    ----------
    x, y : array-like
        Independent samples.

    Returns
    -------
    RichResult with payload:
        statistic, p_value, z, n, m, method
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m, n = int(x.size), int(y.size)
    N = m + n
    if m < 2 or n < 2:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "z": np.nan,
            "n": N, "m": m,
            "method": "Van der Waerden normal-scores test",
        })
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled)
    scores = stats.norm.ppf(ranks / (N + 1.0))
    T = float(scores[:m].sum())
    sum_s2 = float((scores ** 2).sum())
    Var_T = (m * n / float(N * (N - 1))) * sum_s2
    z = T / np.sqrt(Var_T) if Var_T > 0 else np.nan
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return RichResult(payload={
        "statistic": T,
        "p_value": float(p),
        "z": float(z),
        "n": N,
        "m": m,
        "method": "Van der Waerden normal-scores test",
    })


def cheatsheet():
    return "vdwrd: Van der Waerden normal-scores test"


# CANONICAL TEST
# >>> van_der_waerden_test([1,2,3,4,5], [6,7,8,9,10])
# Same idea as Terry-Hoeffding; T strongly negative -> small p-value
