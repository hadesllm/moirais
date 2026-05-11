# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Breusch-Pagan test for heteroscedasticity."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def breusch_pagan(residuals: np.ndarray, X: np.ndarray, cdf=None) -> TestResult:
    """Breusch-Pagan test.

    Parameters
    ----------
    residuals : (n,) OLS residuals
    X : (n, p) original design matrix

    Returns
    -------
    TestResult
    """
    e = np.asarray(residuals, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(e)

    e2 = e**2
    sigma2 = e2.mean()
    g = e2 / sigma2

    X_int = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_int, g, rcond=None)[0]
    ghat = X_int @ beta
    ss_reg = np.sum((ghat - g.mean()) ** 2)
    bp = ss_reg / 2.0
    df = X.shape[1]
    p_val = float(1 - sp_stats.chi2.cdf(bp, df))

    return TestResult(
        test_name="Breusch-Pagan",
        statistic=float(bp),
        p_value=p_val,
        df=float(df),
        method="Breusch-Pagan",
        n=n,
    )


brpgn = breusch_pagan


def cheatsheet() -> str:
    return "breusch_pagan({}) -> Breusch-Pagan test for heteroscedasticity."
