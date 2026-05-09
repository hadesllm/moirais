# moirais.fn — function file (hadesllm/moirais)
"""Engle-Granger cointegration test."""

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def cointegration_test(x: np.ndarray, y: np.ndarray, max_lag: int = 1, cdf=None) -> TestResult:
    """
    Engle-Granger two-step cointegration test.

    Step 1: Regress y on x. Step 2: ADF test on residuals.
    Rejection indicates cointegration.

    :param x: (n,) first I(1) series.
    :param y: (n,) second I(1) series.
    :param max_lag: Lag order for ADF on residuals.
    :return: TestResult with ADF statistic and approximate p-value.

    References
    ----------
    Engle RF, Granger CWJ (1987). Co-integration and error correction.
    Econometrica, 55(2), 251-276.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dr = np.diff(resid)
    m = len(dr)
    start = max_lag
    Y_adf = dr[start:]
    T = len(Y_adf)
    cols = [resid[start : start + T]]
    for lag in range(1, max_lag + 1):
        cols.append(dr[start - lag : start - lag + T])
    X_adf = np.column_stack([np.ones(T)] + cols)
    b = np.linalg.lstsq(X_adf, Y_adf, rcond=None)[0]
    fitted = X_adf @ b
    rss = float(np.sum((Y_adf - fitted) ** 2))
    se = np.sqrt(rss / (T - X_adf.shape[1]) * np.diag(np.linalg.pinv(X_adf.T @ X_adf)))
    gamma = b[1]
    t_stat = float(gamma / se[1]) if se[1] > 0 else 0.0
    cv_5pct = -3.37
    pval_approx = float(sp_stats.norm.cdf(t_stat - cv_5pct + sp_stats.norm.ppf(0.05)))
    pval_approx = np.clip(pval_approx, 0, 1)
    return TestResult(
        test_name="cointegration",
        statistic=t_stat,
        p_value=float(pval_approx),
        df=float(T - X_adf.shape[1]),
        method="Engle-Granger ADF",
        n=n,
        extra={
            "gamma": float(gamma),
            "cointegrating_beta": float(beta[1]),
            "intercept": float(beta[0]),
            "cv_5pct": cv_5pct,
        },
    )


coint = cointegration_test


def cheatsheet() -> str:
    return "cointegration_test({}) -> Engle-Granger cointegration test."
