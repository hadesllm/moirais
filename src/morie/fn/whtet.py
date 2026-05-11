"""White's test for heteroscedasticity."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def white_heterosc_test(
    X,
    y,
) -> TestResult:
    """White's general test for heteroscedasticity.

    Regresses squared OLS residuals on X, X^2, and cross-products.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix.
    y : array-like, shape (n,)
        Response.

    Returns
    -------
    TestResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    e2 = resid**2

    Z_parts = [np.ones((n, 1)), X, X**2]
    for i in range(p):
        for j in range(i + 1, p):
            Z_parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
    Z = np.hstack(Z_parts)
    q = Z.shape[1]
    if n <= q:
        raise ValueError("Not enough observations for White's test.")

    gamma = np.linalg.lstsq(Z, e2, rcond=None)[0]
    fitted = Z @ gamma
    ss_tot = np.sum((e2 - np.mean(e2)) ** 2)
    ss_res = np.sum((e2 - fitted) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    lm = n * r2
    df = q - 1
    pval = sp_stats.chi2.sf(lm, df)

    return TestResult(
        test_name="White's test",
        statistic=float(lm),
        p_value=float(pval),
        df=float(df),
        n=n,
        method="nR^2 LM statistic",
        extra={"r_squared_aux": float(r2)},
    )


whtet = white_heterosc_test


def cheatsheet() -> str:
    return "white_heterosc_test(X, y) -> White's heteroscedasticity test."
