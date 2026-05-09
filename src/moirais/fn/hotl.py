# moirais.fn — function file (hadesllm/moirais)
"""Hotelling's T-squared test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def hotelling_t2(X1: np.ndarray, X2: np.ndarray, cdf=None) -> TestResult:
    """Hotelling's T-squared test for two multivariate groups.

    Parameters
    ----------
    X1 : (n1, p) array
    X2 : (n2, p) array

    Returns
    -------
    TestResult
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    n1, p = X1.shape
    n2 = X2.shape[0]
    if X2.shape[1] != p:
        raise ValueError("X1 and X2 must have same number of variables.")
    if n1 + n2 <= p + 1:
        raise ValueError("Need n1 + n2 > p + 1.")

    d = X1.mean(axis=0) - X2.mean(axis=0)
    S1 = np.cov(X1, rowvar=False)
    S2 = np.cov(X2, rowvar=False)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    if Sp.ndim == 0:
        Sp = Sp.reshape(1, 1)

    inv_Sp = np.linalg.inv(Sp + 1e-10 * np.eye(p))
    T2 = float((n1 * n2) / (n1 + n2) * d @ inv_Sp @ d)

    df1 = p
    df2 = n1 + n2 - p - 1
    F_stat = T2 * df2 / (df1 * (n1 + n2 - 2))
    p_val = float(1 - sp_stats.f.cdf(F_stat, df1, df2))

    return TestResult(
        test_name="Hotelling T2",
        statistic=float(T2),
        p_value=p_val,
        df=float(df1),
        method="Hotelling T-squared",
        n=n1 + n2,
        extra={"F": float(F_stat), "df1": df1, "df2": df2, "n1": n1, "n2": n2},
    )


hotl = hotelling_t2


def cheatsheet() -> str:
    return "hotelling_t2({}) -> Hotelling's T-squared test."
