"""Terry-Hoeffding (Fisher-Yates) normal-scores location test (Gibbons Ch 8.3.1).

Two-sample location test that replaces ranks by expected normal
order statistics ``a_i = E[Z_(i)]`` where Z_(i) is the i-th order
statistic of an iid standard normal sample of size N = m + n.

Statistic for the first (X) sample:
    T = sum_{i in X-ranks} a_i
Under H0 with no ties: E[T] = 0, Var[T] = (m*n/(N*(N-1))) * sum a_i^2.

Approximated normal scores via inverse-CDF of midranks
(``Phi^{-1}((R_i - 3/8)/(N + 1/4))``) -- the Blom approximation,
identical to R's ``qnorm(ppoints(N))``.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["terry_hoeffding_test"]


def terry_hoeffding_test(x, y):
    """Two-sample Terry-Hoeffding (Fisher-Yates) test.

    Parameters
    ----------
    x, y : array-like
        Independent univariate samples.

    Returns
    -------
    RichResult with payload:
        statistic, p_value, z, df=None, n, m, method
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m, n = int(x.size), int(y.size)
    N = m + n
    if m < 2 or n < 2:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "z": np.nan,
            "n": N, "m": m,
            "method": "Terry-Hoeffding (Fisher-Yates) normal-scores test",
        })
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled)
    # Blom normal scores
    a = stats.norm.ppf((ranks - 3.0 / 8.0) / (N + 1.0 / 4.0))
    a_x = a[:m]
    T = float(a_x.sum())
    sum_a2 = float((a ** 2).sum())
    E_T = 0.0  # since scores sum to ~0
    Var_T = (m * n / float(N * (N - 1))) * sum_a2
    z = (T - E_T) / np.sqrt(Var_T) if Var_T > 0 else np.nan
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return RichResult(payload={
        "statistic": T,
        "p_value": float(p),
        "z": float(z),
        "n": N,
        "m": m,
        "method": "Terry-Hoeffding (Fisher-Yates) normal-scores test",
    })


def cheatsheet():
    return "thfdt: Terry-Hoeffding (Fisher-Yates) normal-scores test"


# CANONICAL TEST
# >>> terry_hoeffding_test([1,2,3,4,5], [6,7,8,9,10])
# X has all bottom-half ranks -> large negative T -> small p-value
