# moirais.fn — function file (hadesllm/moirais)
"""Hotelling's T-squared test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def hotelling_t2(X: np.ndarray, Y: np.ndarray | None = None, mu0: np.ndarray | None = None, cdf=None) -> DescriptiveResult:
    """Hotelling's T-squared test.

    One-sample: test if mean of X equals mu0.
    Two-sample: test if means of X and Y are equal.

    Parameters
    ----------
    X : ndarray (n, p)
        First sample.
    Y : ndarray (m, p), optional
        Second sample (two-sample test).
    mu0 : ndarray (p,), optional
        Hypothesised mean for one-sample test. Defaults to zero.

    Returns
    -------
    DescriptiveResult
        ``value`` is T2 statistic. ``extra`` has ``F``, ``p_value``, ``df1``, ``df2``.
    """
    Xa = np.asarray(X, dtype=np.float64)
    n, p = Xa.shape

    if Y is not None:
        Ya = np.asarray(Y, dtype=np.float64)
        m = Ya.shape[0]
        mx = Xa.mean(axis=0)
        my = Ya.mean(axis=0)
        d = mx - my

        Sx = np.cov(Xa, rowvar=False, ddof=1)
        Sy = np.cov(Ya, rowvar=False, ddof=1)
        Sp = ((n - 1) * Sx + (m - 1) * Sy) / (n + m - 2)
        Sp += np.eye(p) * 1e-10

        T2 = float(d @ np.linalg.inv(Sp / n + Sp / m) @ d)
        nu = n + m - 2
        F_val = T2 * (nu - p + 1) / (nu * p)
        df1, df2 = p, nu - p + 1
    else:
        if mu0 is None:
            mu0 = np.zeros(p)
        mu0 = np.asarray(mu0, dtype=np.float64)
        d = Xa.mean(axis=0) - mu0
        S = np.cov(Xa, rowvar=False, ddof=1) + np.eye(p) * 1e-10
        T2 = float(n * d @ np.linalg.inv(S) @ d)
        F_val = T2 * (n - p) / (p * (n - 1))
        df1, df2 = p, n - p

    df2 = max(df2, 1)
    p_value = float(1 - sp_stats.f.cdf(max(F_val, 0), df1, df2))

    return DescriptiveResult(
        name="HotellingT2",
        value=T2,
        extra={
            "F": float(F_val),
            "p_value": p_value,
            "df1": df1,
            "df2": df2,
        },
    )


hott2 = hotelling_t2


def cheatsheet() -> str:
    return "hotelling_t2({}) -> Hotelling's T-squared test."
