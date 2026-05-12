# morie.fn -- function file (hadesllm/morie)
"""Granger causality test."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def granger_test(y: np.ndarray, x: np.ndarray, max_lag: int = 4, cdf=None) -> DescriptiveResult:
    """
    Granger causality test: does *x* Granger-cause *y*?

    Compares a restricted AR(p) model of *y* on its own lags to an
    unrestricted model including lags of *x*. Uses an F-test.

    :param y: 1-D dependent series.
    :param x: 1-D candidate causal series (same length).
    :param max_lag: Maximum lag to test. Default 4.
    :return: DescriptiveResult with F-statistic, p-value, best lag.
    :raises ValueError: On length mismatch or insufficient data.

    References
    ----------
    Granger C.W.J. (1969). Investigating causal relations by
    econometric models and cross-spectral methods. *Econometrica*,
    37(3), 424-438.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if len(y) != len(x):
        raise ValueError(f"y and x must have same length, got {len(y)} and {len(x)}.")
    n = len(y)
    if n < max_lag + 5:
        raise ValueError(f"Need at least {max_lag + 5} observations, got {n}.")
    best_f = -1.0
    best_p = 1.0
    best_lag = 1
    for lag in range(1, max_lag + 1):
        T = n - lag
        dep = y[lag:]
        X_r = np.column_stack([np.ones(T)] + [y[lag - i - 1 : n - i - 1] for i in range(lag)])
        X_u = np.column_stack(
            [X_r] + [x[lag - i - 1 : n - i - 1] for i in range(lag)]
        )
        rss_r = float(np.sum((dep - X_r @ np.linalg.lstsq(X_r, dep, rcond=None)[0]) ** 2))
        rss_u = float(np.sum((dep - X_u @ np.linalg.lstsq(X_u, dep, rcond=None)[0]) ** 2))
        df1 = lag
        df2 = T - 2 * lag - 1
        if df2 < 1 or rss_u <= 0:
            continue
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_val = 1 - stats.f.cdf(f_stat, df1, df2)
        if f_stat > best_f:
            best_f = f_stat
            best_p = p_val
            best_lag = lag
    return DescriptiveResult(
        name="granger_test",
        value=best_f,
        extra={
            "f_statistic": best_f,
            "p_value": best_p,
            "best_lag": best_lag,
            "max_lag": max_lag,
            "n": n,
        },
    )


grang = granger_test


def cheatsheet() -> str:
    return "granger_test({}) -> Granger causality F-test."
