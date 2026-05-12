# morie.fn -- function file (hadesllm/morie)
"""Ramsey RESET test for functional form."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def ramsey_reset_test(
    X,
    y,
    *,
    power: int = 3,
) -> TestResult:
    """Ramsey RESET test for omitted nonlinearities.

    Augments the OLS model with powers of fitted values
    (y-hat^2, ..., y-hat^power) and tests joint significance.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    power : int
        Maximum power of fitted values (default 3).

    Returns
    -------
    TestResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < p + power + 1:
        raise ValueError("Not enough observations.")

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid_r = y - y_hat
    rss_r = np.sum(resid_r**2)

    aug_cols = [X]
    for pw in range(2, power + 1):
        aug_cols.append((y_hat**pw).reshape(-1, 1))
    X_aug = np.hstack(aug_cols)
    q = X_aug.shape[1]

    beta_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    resid_u = y - X_aug @ beta_aug
    rss_u = np.sum(resid_u**2)

    df_num = power - 1
    df_den = n - q
    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den) if rss_u > 1e-12 else np.inf
    pval = sp_stats.f.sf(f_stat, df_num, df_den)

    return TestResult(
        test_name="Ramsey RESET test",
        statistic=float(f_stat),
        p_value=float(pval),
        df=float(df_num),
        n=n,
        method=f"power={power}",
        extra={"df_num": df_num, "df_den": df_den},
    )


ramsy = ramsey_reset_test


def cheatsheet() -> str:
    return "ramsey_reset_test(X, y) -> Ramsey RESET test."
