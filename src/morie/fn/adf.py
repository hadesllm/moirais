# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Augmented Dickey-Fuller stationarity test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def adf_test(x, cdf=None, *, max_lag: int | None = None) -> TestResult:
    """ADF unit root test. H0: unit root (non-stationary).

    Parameters
    ----------
    x : array-like
        Time series observations.
    max_lag : int, optional
        Number of lagged difference terms. Default: floor((n-1)^(1/3)).

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        raise ValueError("Need at least 10 observations")
    if max_lag is None:
        max_lag = int(np.floor((n - 1) ** (1 / 3)))
    dx = np.diff(x)
    x_lag = x[max_lag:-1]
    dx_dep = dx[max_lag:]
    # Build regressor matrix: [x_{t-1}, dx_{t-1}, ..., dx_{t-p}, 1]
    regressors = [x_lag]
    for i in range(max_lag):
        regressors.append(dx[max_lag - i - 1 : n - 2 - i])
    regressors.append(np.ones(len(dx_dep)))
    X = np.column_stack(regressors)
    beta = np.linalg.lstsq(X, dx_dep, rcond=None)[0]
    resid = dx_dep - X @ beta
    se_beta = np.sqrt(np.sum(resid**2) / (len(dx_dep) - X.shape[1]) * np.abs(np.diag(np.linalg.pinv(X.T @ X))))
    adf_stat = beta[0] / se_beta[0] if se_beta[0] > 0 else 0
    # Left-tail test using normal approximation
    p_val = float(stats.norm.cdf(adf_stat))
    return TestResult(
        test_name="ADF",
        statistic=float(adf_stat),
        p_value=p_val,
        n=n,
        method=f"ADF test (lag={max_lag})",
        extra={"lag": max_lag, "gamma": float(beta[0])},
    )


adf = adf_test


def cheatsheet() -> str:
    return "adf_test({}) -> Augmented Dickey-Fuller stationarity test."
