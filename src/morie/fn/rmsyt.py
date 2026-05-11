# morie.fn — function file (hadesllm/morie)
"""Ramsey RESET specification test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def ramsey_reset(y: np.ndarray, X: np.ndarray, cdf=None, *, powers: list[int] | None = None) -> TestResult:
    """Ramsey RESET test for functional form misspecification.

    Parameters
    ----------
    y : (n,)
    X : (n, p)
    powers : list of int
        Powers of fitted values to include (default [2, 3]).

    Returns
    -------
    TestResult
    """
    if powers is None:
        powers = [2, 3]
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])

    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
    yhat = X_int @ beta
    resid_r = y - yhat
    ssr_r = np.sum(resid_r**2)
    df_r = n - X_int.shape[1]

    Z = np.column_stack([yhat**pw for pw in powers])
    X_aug = np.column_stack([X_int, Z])
    beta_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    resid_u = y - X_aug @ beta_aug
    ssr_u = np.sum(resid_u**2)
    df_u = n - X_aug.shape[1]

    q = len(powers)
    F_stat = ((ssr_r - ssr_u) / q) / (ssr_u / df_u) if df_u > 0 and ssr_u > 0 else 0.0
    p_val = float(1 - sp_stats.f.cdf(F_stat, q, df_u))

    return TestResult(
        test_name="RESET",
        statistic=float(F_stat),
        p_value=p_val,
        df=float(q),
        method="Ramsey RESET",
        n=n,
        extra={"df_num": q, "df_den": df_u, "powers": powers},
    )


rmsyt = ramsey_reset


def cheatsheet() -> str:
    return "ramsey_reset({}) -> Ramsey RESET specification test."
