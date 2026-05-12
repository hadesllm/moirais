# morie.fn -- function file (hadesllm/morie)
"""Chow structural break test."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def chow_test(y: np.ndarray, X: np.ndarray, break_point: int, cdf=None) -> DescriptiveResult:
    """
    Chow test for a structural break at a known break point.

    Compares RSS of pooled regression to sum of sub-sample RSSes.

    :param y: (n,) dependent variable.
    :param X: (n, k) regressor matrix (include intercept if needed).
    :param break_point: Index of the break point.
    :return: DescriptiveResult with F-statistic and p-value.
    :raises ValueError: If break point invalid.

    References
    ----------
    Chow G.C. (1960). Tests of equality between sets of coefficients
    in two linear regressions. *Econometrica*, 28(3), 591-605.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if break_point < k or break_point > n - k:
        raise ValueError(f"break_point must be in [{k}, {n - k}], got {break_point}.")
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
    rss_full = float(np.sum((y - X @ beta_full) ** 2))
    beta1 = np.linalg.lstsq(X[:break_point], y[:break_point], rcond=None)[0]
    rss1 = float(np.sum((y[:break_point] - X[:break_point] @ beta1) ** 2))
    beta2 = np.linalg.lstsq(X[break_point:], y[break_point:], rcond=None)[0]
    rss2 = float(np.sum((y[break_point:] - X[break_point:] @ beta2) ** 2))
    df1 = k
    df2 = n - 2 * k
    if df2 < 1 or (rss1 + rss2) <= 0:
        raise ValueError("Insufficient degrees of freedom.")
    f_stat = ((rss_full - rss1 - rss2) / df1) / ((rss1 + rss2) / df2)
    p_val = 1 - stats.f.cdf(f_stat, df1, df2)
    return DescriptiveResult(
        name="chow_test",
        value=float(f_stat),
        extra={
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "df1": df1,
            "df2": df2,
            "break_point": break_point,
            "rss_full": rss_full,
            "rss1": rss1,
            "rss2": rss2,
            "n": n,
        },
    )


chows = chow_test


def cheatsheet() -> str:
    return "chow_test({}) -> Chow structural break F-test."
