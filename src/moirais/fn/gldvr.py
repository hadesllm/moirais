# moirais.fn — function file (hadesllm/moirais)
"""Goldfeld-Quandt heteroscedasticity test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def goldfeld_quandt_test(
    X,
    y,
    *,
    split_fraction: float = 0.4,
    sort_col: int = 0,
) -> TestResult:
    """Goldfeld-Quandt test for heteroscedasticity.

    Sorts by the specified column, omits the middle fraction,
    and compares residual variances in the two remaining groups.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    split_fraction : float
        Fraction of observations in each tail (default 0.4).
    sort_col : int
        Column index to sort by.

    Returns
    -------
    TestResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    order = np.argsort(X[:, sort_col])
    X, y = X[order], y[order]

    n_tail = int(n * split_fraction)
    if n_tail < p + 1:
        raise ValueError("split_fraction too small for the number of predictors.")

    X1, y1 = X[:n_tail], y[:n_tail]
    X2, y2 = X[-n_tail:], y[-n_tail:]

    r1 = y1 - X1 @ np.linalg.lstsq(X1, y1, rcond=None)[0]
    r2 = y2 - X2 @ np.linalg.lstsq(X2, y2, rcond=None)[0]

    rss1 = float(np.sum(r1**2))
    rss2 = float(np.sum(r2**2))
    df1 = n_tail - p
    df2 = n_tail - p

    f_stat = (rss2 / df2) / (rss1 / df1) if rss1 > 1e-12 else np.inf
    pval = sp_stats.f.sf(f_stat, df2, df1)

    return TestResult(
        test_name="Goldfeld-Quandt test",
        statistic=float(f_stat),
        p_value=float(pval),
        df=float(df2),
        n=n,
        method=f"split={split_fraction}, sort_col={sort_col}",
        extra={"rss_lower": rss1, "rss_upper": rss2},
    )


gldvr = goldfeld_quandt_test


def cheatsheet() -> str:
    return "goldfeld_quandt_test(X, y) -> Goldfeld-Quandt heteroscedasticity test."
