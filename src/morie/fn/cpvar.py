# morie.fn -- function file (hadesllm/morie)
"""Confine yourself to the present. -- Marcus Aurelius"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def variance_changepoint(y: np.ndarray, alpha: float = 0.05, cdf=None) -> DescriptiveResult:
    """
    Detect a single variance change point in a time series.

    Tests each candidate split point and selects the one maximising the
    likelihood-ratio statistic for a change in variance (assuming normal
    data with constant mean within segments).

    :param y: 1-D time series.
    :type y: numpy.ndarray
    :param alpha: Significance level. Default 0.05.
    :type alpha: float
    :return: DescriptiveResult with change point and variance estimates.
    :rtype: DescriptiveResult

    References
    ----------
    Inclan C. & Tiao G.C. (1994). Use of Cumulative Sums of Squares for
    Retrospective Detection of Changes of Variance. *Journal of the
    American Statistical Association*, 89(427), 913-923.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 6:
        raise ValueError(f"Need >= 6 observations, got {n}.")
    total_var = float(np.var(y, ddof=1))
    best_stat = 0.0
    best_tau = 1
    for tau in range(3, n - 2):
        v1 = np.var(y[:tau], ddof=1)
        v2 = np.var(y[tau:], ddof=1)
        if v1 <= 0 or v2 <= 0:
            continue
        ll_seg = -(tau / 2) * np.log(v1) - ((n - tau) / 2) * np.log(v2)
        ll_null = -(n / 2) * np.log(total_var)
        lr = 2.0 * (ll_seg - ll_null)
        if lr > best_stat:
            best_stat = lr
            best_tau = tau
    p_value = float(1.0 - _st.chi2.cdf(best_stat, df=1))
    significant = p_value < alpha
    var1 = float(np.var(y[:best_tau], ddof=1))
    var2 = float(np.var(y[best_tau:], ddof=1))
    return DescriptiveResult(
        name="variance_changepoint",
        value=best_tau,
        extra={
            "changepoint": best_tau,
            "variance_before": var1,
            "variance_after": var2,
            "lr_statistic": float(best_stat),
            "p_value": p_value,
            "significant": significant,
        },
    )


cpvar = variance_changepoint


def cheatsheet() -> str:
    return "variance_changepoint({}) -> Variance change-point detection. 'Be careful not to choke on"
