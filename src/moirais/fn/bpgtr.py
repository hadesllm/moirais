# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Breusch-Pagan test for heteroscedasticity."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def breusch_pagan_test(
    X,
    y,
) -> TestResult:
    """Breusch-Pagan test for heteroscedasticity.

    Regresses squared OLS residuals on X and tests via LM statistic.

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
    if n < p + 2:
        raise ValueError("Need n > p+1 observations.")

    Xa = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
    resid = y - Xa @ beta
    sigma2 = np.sum(resid**2) / n

    e2 = resid**2 / sigma2
    gamma = np.linalg.lstsq(Xa, e2, rcond=None)[0]
    fitted = Xa @ gamma
    ss_reg = np.sum((fitted - np.mean(e2)) ** 2)
    lm = ss_reg / 2.0
    pval = sp_stats.chi2.sf(lm, p)

    return TestResult(
        test_name="Breusch-Pagan test",
        statistic=float(lm),
        p_value=float(pval),
        df=float(p),
        n=n,
        method="LM statistic",
    )


bpgtr = breusch_pagan_test


def cheatsheet() -> str:
    return "breusch_pagan_test(X, y) -> Breusch-Pagan heteroscedasticity test."
